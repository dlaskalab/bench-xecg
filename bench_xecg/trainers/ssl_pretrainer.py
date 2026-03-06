import lightning as L
import numpy as np
import torch
import lightning

from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import Perceptron, LogisticRegression

from .common import configure_optimizer_teacher_student, configure_optimizers
from ..utils.loss_utils import masked_mse_loss, masked_mae_loss, gradient_loss, masked_min_max_loss, masked_cosine_loss, SimDINOv2Loss
from ..utils.lejepa.epps_pulley import SIGReg
from ..utils.plot_utils import plot_reconstruction, plot_local_views, plot_latent_space


# define the LightningModule
class PretrainedNetwork(L.LightningModule):
    def __init__(
            self, 
            model, 
            len_train_dataset,
            config,
            ptb_xl_train_dataloader=None,
            ptb_xl_val_dataloader=None,
            mit_bih_train_dataloader=None,
            mit_bih_val_dataloader=None,
        ):
        super().__init__()
        self.lr = config.lr
        self.layerwise_lr_decay = config.layerwise_lr_decay
        self.reconstruction_lr = config.reconstruction_lr
        self.model = model
        self.batch_size = config.batch_size
        self.optimizer = config.optimizer
        self.wd = config.wd
        self.final_wd = config.final_wd if hasattr(config, 'final_wd') else config.wd
        self.use_scheduler = config.use_scheduler
        self.patch_size = config.patch_size
        self.epochs = config.epochs
        self.loss_type = config.loss_type
        self.mask_ratio = config.mask_ratio
        self.len_train_dataset = len_train_dataset
        self.num_epochs_warmup = config.num_epochs_warmup
        self.sched_decay_factor = config.sched_decay_factor
        self.grad_loss_lambda = config.grad_loss_lambda
        self.grad_clip = config.grad_clip
        self.min_max_loss_lambda = config.min_max_loss_lambda
        self.pretraining_strategy = config.strategy
        self.start_train_head_at_epoch = config.start_train_head_at_epoch
        self.lambda_loss =  config.lambda_loss
        self.devices = config.devices
        self.sampling_freq = config.sampling_freq
        self.use_teacher_student = False
        self.plot_samples = config.plot_samples
        self.sync_dist = config.devices >= 2 if config.devices is not None else False
        self.lambda_code_rate = config.lambda_code_rate

        self.validation_step_outputs = [] 

        if self.pretraining_strategy == 'sim_dino_v2':
            self.sim_dino_loss = SimDINOv2Loss(eps=0.05)
            self.automatic_optimization=False
            self.use_teacher_student = True


        if self.pretraining_strategy.startswith('lejepa'):
            self.lejepa_loss = SIGReg().to('cuda')
            if self.pretraining_strategy == 'lejepa_masked':
                self.automatic_optimization=False

        if self.use_teacher_student:
            self.ema_0 = config.ema_0
            self.ema_1 = config.ema_1
            self.model.init_teacher()
            self.teacher_temp = config.teacher_temp
            self.stud_temp = config.stud_temp


        self.ptb_xl_train_dataloader = ptb_xl_train_dataloader
        self.ptb_xl_val_dataloader = ptb_xl_val_dataloader

        self.mit_bih_train_dataloader = mit_bih_train_dataloader
        self.mit_bih_val_dataloader = mit_bih_val_dataloader

    def configure_model(self):
        # Ensure model is properly initialized before DDP
        torch.cuda.synchronize()
        return self.model

    def training_step(self, batch, _):
        losses = self.reconstruct_batch(batch, step='train')
        rec_loss, pretraining_loss = losses['reconstruction_loss'], losses['pretraining_loss']
        
        # only for teacher student architectures
        if not self.automatic_optimization:
            opt_core, opt_head = self.optimizers()
            sched_core, sched_head = self.lr_schedulers()

            train_head = self.current_epoch >= self.start_train_head_at_epoch
            
            opt_core.zero_grad(set_to_none=True)
            self.manual_backward(pretraining_loss, retain_graph=False)
            self.clip_gradients(opt_core, gradient_clip_val=self.grad_clip, gradient_clip_algorithm="norm")

            opt_core.step()
            sched_core.step()

            if train_head:
                opt_head.zero_grad(set_to_none=True)
                self.manual_backward(rec_loss)
                opt_head.step()
                sched_head.step()

            if self.use_teacher_student:
                self.update_teacher()
                self.update_scheduled_weight_decay(opt_core)
            return
        else:
            return rec_loss
        
    def update_scheduled_weight_decay(self, opt_core):
        # linear decay
        steps_per_epoch = np.ceil(self.len_train_dataset / self.batch_size)
        total_steps = steps_per_epoch * self.epochs
        step = self.global_step 
        if self.use_teacher_student:  step = step // 2 # this because i do two steps in the training loop
        wd = self.wd + (self.final_wd - self.wd) * (step / total_steps)
    
        for param_group in opt_core.param_groups:
            param_group['weight_decay'] = wd 

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)
    
    @torch.no_grad()
    def update_teacher(self):
        steps_per_epoch = np.ceil(self.len_train_dataset / self.batch_size)
        # with teacher-student, the number of training steps is counted twice because of two grad steps
        # so i have to divide the global_step by two
        num_training_steps = steps_per_epoch * self.epochs * 2 
        beta = self.ema_0 + self.global_step * (self.ema_1 - self.ema_0) / num_training_steps
        beta = min(max(beta, 0.0), 1.0) # bound to max 1.0
        self.log('teacher_beta', beta, prog_bar=False, sync_dist=self.sync_dist)
        self.update_module(self.model._teacher, self.model, beta)
        
    def update_module(self, teacher_module, student_module, beta):
        student_params = dict(student_module.named_parameters())
        for name_t, param_t in teacher_module.named_parameters():
            if name_t not in student_params:
                raise KeyError(f"{name_t} not found in student")
            param_s = student_params[name_t]
            param_t.data = param_t.data * beta + (1.0 - beta) * param_s.data

    def update_param(self, teacher_param, student_param, beta):
        teacher_param.data = teacher_param.data * beta + (1.0 - beta) * student_param.data

    def validation_step(self, batch, _):
        losses = self.reconstruct_batch(batch, step='val')
        pretraining_loss = losses['pretraining_loss']

        if 'embeddings' in losses.keys():
            embeddings = losses['embeddings']
            embedding_mean_over_views = embeddings.detach().cpu().mean(0)
            self.validation_step_outputs.append(embedding_mean_over_views)

        return pretraining_loss
    
    def test_step(self, batch, _):
        loss = self.reconstruct_batch(batch, step='test')
        return loss
    
    def on_train_epoch_end(self):
        """
        When the training loop ends, some representative plots from different classes are saved on wandb
        """
        if self.logger is None or not self.plot_samples:
            return super().on_validation_epoch_end()

        self.log_sample_plots(self.trainer.train_dataloader, 0, -42, stage_name='train')

        return super().on_train_epoch_end()

    def on_validation_epoch_start(self):
        # Clear storage at start of validation
        self.validation_step_outputs = []
        super().on_validation_epoch_start()

    def on_validation_epoch_end(self):
        """
        When the validation loop ends, some representative plots from different classes are saved on wandb
        """
        # if self.global_step <= 1:
        #    return super().on_validation_epoch_end()
        
        # self.eval_model_downstream_mit_bih()
        self.eval_model_downstream_ptb_xl()

        if self.logger is None or not self.plot_samples:
            return super().on_validation_epoch_end()
        
        self.log_sample_plots(self.trainer.val_dataloaders, 115, -25, stage_name='val')

        if len(self.validation_step_outputs) > 0:

            effective_rank, total_variance, path = plot_latent_space(torch.cat(self.validation_step_outputs, dim=0), self.current_epoch, self.logger.log_dir)
            self.log('val_effective_rank', effective_rank, prog_bar=True)
            self.log('val_embedding_variance', total_variance, prog_bar=False)
            self.logger.log_image(key="latent_space_analysis", images=[path])
            self.validation_step_outputs = []

        return super().on_validation_epoch_end()

    def log_sample_plots(self, dataloader, fixed_idx1, fixed_idx2, stage_name=''):
        # save the plots of the reconstruction for some samples
        sample_1 = dataloader.dataset[fixed_idx1]
        sample_2 = dataloader.dataset[fixed_idx2]

        # get two random samples from the training dataset
        idx_3 = np.random.randint(0, len(dataloader.dataset))
        idx_4 = np.random.randint(0, len(dataloader.dataset))

        sample_3 = dataloader.dataset[idx_3]
        sample_4 = dataloader.dataset[idx_4]

        log_dir = self.logger.log_dir if self.logger.log_dir is not None else self.logger.experiment.dir

        local_views = plot_local_views(sample_4, self.patch_size, self.sampling_freq, self.device, log_dir, self.current_epoch, 'local_views_sample_s')
        if isinstance(self.logger, lightning.pytorch.loggers.WandbLogger):
            self.logger.log_image(key=f"local_views_{stage_name}", images=[local_views])

        if self.pretraining_strategy == 'sim_dino_v2' or self.pretraining_strategy == 'lejepa_masked':
            img_1 = plot_reconstruction(sample_1, self.model, self.patch_size, self.sampling_freq, self.device, log_dir, self.current_epoch, 'sample_s', training_strategy=self.pretraining_strategy)
            img_2 = plot_reconstruction(sample_2, self.model, self.patch_size, self.sampling_freq, self.device, log_dir, self.current_epoch, 'sample_v', training_strategy=self.pretraining_strategy)
            img_3 = plot_reconstruction(sample_3, self.model, self.patch_size, self.sampling_freq, self.device, log_dir, self.current_epoch, 'sample_t', training_strategy=self.pretraining_strategy)
            img_4 = plot_reconstruction(sample_4, self.model, self.patch_size, self.sampling_freq, self.device, log_dir, self.current_epoch, 'sample_n', training_strategy=self.pretraining_strategy)
            if isinstance(self.logger, lightning.pytorch.loggers.WandbLogger):
                self.logger.log_image(key=f"reconstructions_{stage_name}", images=[img_1, img_2, img_3, img_4])

    def reconstruct_batch(self, batch, step):
        if self.pretraining_strategy == 'sim_dino_v2':
            return self.reconstruct_batch_sim_dino_v2(batch, step)
        elif self.pretraining_strategy.startswith('lejepa'):
            return self.reconstruct_batch_lejepa(batch, step)
        else:
            raise ValueError(f"Pretraining strategy {self.pretraining_strategy} not implemented yet")
        
    def reconstruct_batch_lejepa(self, batch, step):
        global_signals = batch["global_signals"]
        local_signals = batch["local_signals"]

        n_global_views, bs, seq_len, n_channels = global_signals.shape

        global_out = self.model(global_signals.reshape(-1, seq_len, n_channels), masking=False, reconstruct=False)
        global_out_cls = global_out['cls'].reshape(n_global_views, bs, global_out['cls'].shape[-1])  # [n_global_views, bs dim]
        # print(f'Global out CLS shape: {global_out_cls.shape}') [n_global_views, bs, dim]

        n_local_views, bs, seq_len, n_channels = local_signals.shape
        local_out = self.model(local_signals.reshape(-1, seq_len, n_channels), masking=False, reconstruct=False)
        local_out_cls = local_out['cls'].reshape(n_local_views, bs, local_out['cls'].shape[-1])  # [n_local_views, bs dim]
        # print(f'Local out CLS shape: {local_out_cls.shape}') [n_local_views, bs, dim]

        all_view_cls = torch.cat([global_out_cls, local_out_cls], dim=0)  # [global_views + local_views, bs, dim]
        # print(f'All view CLS shape: {all_view_cls.shape}')

        global_mean_cls = global_out_cls.mean(dim=0)  # [bs, dim]
        similarity = (global_mean_cls.unsqueeze(0) - all_view_cls).square().mean()
        self.log(f"{step}_similarity_loss", similarity.item(), prog_bar=True, sync_dist=self.sync_dist)

        sigreg = torch.mean(torch.stack([self.lejepa_loss(samples) for samples in all_view_cls]))
        self.log(f"{step}_sigreg_loss", sigreg.item(), prog_bar=True, sync_dist=self.sync_dist)

        lejepa_loss = (1 - self.lambda_loss) * similarity + self.lambda_loss * sigreg

        self.log(f"{step}_lejepa_loss", lejepa_loss.item(), prog_bar=True, sync_dist=self.sync_dist)


        return {'reconstruction_loss': None, 'pretraining_loss': lejepa_loss, 'embeddings': all_view_cls}

    def reconstruct_batch_sim_dino_v2(self, batch, step):
        global_signals = batch["global_signals"]
        local_signals = batch["local_signals"]
        batch_size, _, num_leads = global_signals[0].shape

        global_genders = batch["global_genders"]
        global_ages = batch["global_ages"]
        local_ages = batch["local_ages"]
        local_genders = batch["local_genders"]

        global_out = [self.model(x, masking=True, age=age, gender=gender) for x, age, gender in zip(global_signals, global_ages, global_genders)]
        global_out_teacher = [self.model.teacher_fwd(x, age=age, gender=gender) for x, age, gender in zip(global_signals, global_ages, global_genders)]
        local_out = [self.model(x, masking=False, reconstruct=False, age=age, gender=gender) for x, age, gender in zip(local_signals, local_ages, local_genders)]

        # compute the loss and use the gradients only when it is needed
        teacher_student_loss = None

        # padding_masks contains the padded part of the signal that should be excluded from the loss calculation
        padding_masks = [(sig != 0.).flip(1).cumsum(dim=1).flip(1) == 0 for sig in global_signals]
        padding_masks_patched = [m.view(batch_size, m.shape[1] // self.patch_size, self.patch_size, num_leads).sum(dim=-1) == num_leads for m in padding_masks]
        combined_padding_mask = torch.stack(padding_masks_patched, dim=1).max(dim=-1)[0] # [bs, n_global_views, seq_len // patch_size]

        # mask for the loss calculation, true values need to be included in the loss, false values need to be excluded
        masks = [out['mask'] for out in global_out]
        seq_len = global_signals[0].shape[1]
        n_patches = seq_len // self.patch_size
        num_age_genger_tokens = [out['mask'].shape[1] - n_patches for out in global_out]
        combined_mask = torch.stack(masks, dim=1).max(dim=-1)[0] # [bs, n_global_views, num_tokens]
        # i do not want to predict where masking is applied to masked tokens
        combined_mask[:, :, num_age_genger_tokens[0]:] = combined_mask[:, :, num_age_genger_tokens[0]:] * ~combined_padding_mask
        combined_mask = combined_mask.flatten()

        cls_tok_stud_g = torch.stack([g['cls'] for g in global_out] + [l['cls'] for l in local_out], dim=0)
        cls_tok_teacher_g = torch.stack([g['cls'] for g in global_out_teacher], dim=0)
    
        compression_term, expansion_term = self.sim_dino_loss(cls_tok_stud_g, cls_tok_teacher_g)
        self.log(f"{step}_compression_term", compression_term.item(), prog_bar=False, sync_dist=self.devices == 2)
        self.log(f"{step}_expansion_term", expansion_term.item(), prog_bar=False, sync_dist=self.devices == 2)

        stud_embeddings = torch.cat([out['patches'] for out in global_out], dim=1).flatten(0, 1)
        teacher_embeddings = torch.cat([out_t['patches'] for out_t in global_out_teacher], dim=1).flatten(0, 1)

        # simplified dino uses only the mse between the embeddigns
        patch_loss = masked_cosine_loss(stud_embeddings, teacher_embeddings, reduction='mean', mask=combined_mask)
        self.log(f"{step}_patch_loss", patch_loss.item(), prog_bar=True, sync_dist=self.devices == 2)

        rank_me = [self.rank_me(out['cls']) for out in global_out]
        self.log(f"{step}_rank_me", (sum(rank_me) / len(rank_me)).item(), prog_bar=True)

        teacher_student_loss = compression_term + self.lambda_code_rate * expansion_term + patch_loss
        self.log(f"{step}_dino_loss", teacher_student_loss.item(), prog_bar=True, sync_dist=self.devices == 2)

        #  = [self.rank_me(out['cls']) for out in global_out]
        # self.log(f"{step}_rank_me", (sum(rank_me) / len(rank_me)).item(), prog_bar=True, sync_dist=self.devices == 2)

        # log norm of output
        with torch.no_grad():
            norm = torch.norm(global_out[0]['patches'], dim=-1)
            norm = norm.mean()
            self.log(f"{step}_norm_emb", norm.item(), prog_bar=False, sync_dist=self.devices == 2)

            # log the mean cosine similarity between all samples in the batch
            cos_sim = torch.nn.functional.cosine_similarity(global_out[0]['cls'].unsqueeze(1), global_out[1]['cls'].unsqueeze(0), dim=-1)
            # zero the diagonal
            cos_sim2 = torch.nn.functional.cosine_similarity(global_out[1]['cls'].unsqueeze(1), global_out[0]['cls'].unsqueeze(0), dim=-1)
            mask = torch.eye(cos_sim.shape[0], device=cos_sim.device).bool()
            cos_sim_diff = (cos_sim[~mask].mean() + cos_sim2[~mask].mean()) / 2
            cos_sim_same = (cos_sim2[mask].mean() + cos_sim[mask].mean()) / 2
            self.log(f"{step}_cos_sim_different_samples", cos_sim_diff.item(), prog_bar=False, sync_dist=self.devices == 2)
            self.log(f"{step}_cos_sim_same_samples", cos_sim_same.mean().item(), prog_bar=False, sync_dist=self.devices == 2)
        
        nrmse, mse, mae, grad, min_max = self.calculate_metrics_reconstruction(global_out[0]['reconstruction'], global_signals[0], mask = None) #  out['mask'])
        nrmse2, mse2, mae2, grad2, min_max2 = self.calculate_metrics_reconstruction(global_out[1]['reconstruction'], global_signals[1], mask=None) #, out2['mask'])
        nrmse, mse, mae, grad, min_max = (nrmse + nrmse2) / 2, (mse + mse2) / 2, (mae + mae2) / 2, (grad + grad2) / 2, (min_max + min_max2) / 2

        loss = torch.tensor(0.0, device=self.device)

        if 'mae' in self.loss_type: loss += mae
        elif 'mse' in self.loss_type: loss += mse
        if 'grad' in self.loss_type: loss += grad * self.grad_loss_lambda
        if 'min_max' in self.loss_type: loss += min_max * self.min_max_loss_lambda

        self.log(f"{step}_loss", loss.item(), prog_bar=True, sync_dist=self.devices == 2)
        self.log(f"{step}_mse", mse.item(), prog_bar=False, sync_dist=self.devices == 2)
        self.log(f"{step}_mae", mae.item(), prog_bar=False, sync_dist=self.devices == 2)
        self.log(f"{step}_grad", grad.item(), prog_bar=False, sync_dist=self.devices == 2)
        if 'min_max' in self.loss_type: self.log(f"{step}_min_max", min_max.item(), prog_bar=False)
        
        self.log(f"{step}_nrmse", nrmse.mean().item(), prog_bar=False, sync_dist=self.devices == 2)

        return {'reconstruction_loss': loss, 'pretraining_loss': teacher_student_loss}
    
    @torch.no_grad()
    def rank_me(self, tensor, eps=1e-8):
        if not torch.isfinite(tensor).all():
            return torch.tensor(0.0, device=tensor.device)
        try:
            _, S, _ = torch.linalg.svd(tensor, full_matrices=False)  # shape: (min(N, D),)

            # Normalize singular values to get a probability distribution
            S_norm = S / (S.sum() + eps)

            # Entropy of the distribution
            entropy = -torch.sum(S_norm * torch.log(S_norm + eps))

            # Effective rank
            rank_me = torch.exp(entropy)
            return rank_me / tensor.shape[0]
        except:
            return torch.tensor(0.0, device=self.device)
        
    def reconstruction_head_step(self, global_outs, global_signals, step):
        nrmses, meses, maes, grads, min_maxs = [], [], [], [], []
        for global_out, global_signal in zip(global_outs, global_signals):
            nrmse, mse, mae, grad, min_max = self.calculate_metrics_reconstruction(global_out, global_signal, mask = None) # out['mask'])
            nrmses.append(nrmse)
            meses.append(mse)
            maes.append(mae)
            grads.append(grad)
            min_maxs.append(min_max)

        nrmse, mse, mae, grad, min_max = (sum(nrmses) / len(nrmses), sum(meses) / len(meses), sum(maes) / len(maes), sum(grads) / len(grads), sum(min_maxs) / len(min_maxs))

        loss = torch.tensor(0.0, device=self.device)

        if 'mae' in self.loss_type: loss += mae
        elif 'mse' in self.loss_type: loss += mse
        if 'grad' in self.loss_type: loss += grad * self.grad_loss_lambda
        if 'min_max' in self.loss_type: loss += min_max * self.min_max_loss_lambda

        self.log(f"{step}_loss", loss.item(), prog_bar=True, sync_dist=self.sync_dist)
        self.log(f"{step}_mse", mse.item(), prog_bar=False, sync_dist=self.sync_dist)
        self.log(f"{step}_mae", mae.item(), prog_bar=False, sync_dist=self.sync_dist)
        self.log(f"{step}_grad", grad.item(), prog_bar=False, sync_dist=self.sync_dist)
        if 'min_max' in self.loss_type: self.log(f"{step}_min_max", min_max.item(), prog_bar=False, sync_dist=self.sync_dist)
        
        self.log(f"{step}_nrmse", nrmse.mean().item(), prog_bar=False, sync_dist=self.sync_dist)
        return loss
    
    def calculate_metrics_reconstruction(self, rec, target, mask):
        batch_size, tokens_num, channels = target.shape
        if mask is not None:
            patched_mask = mask.view(batch_size, tokens_num // self.patch_size, self.patch_size)
            mask = mask.view(batch_size, tokens_num, 1).repeat_interleave(channels, dim=-1)
        else:
            patched_mask = None

        nrmse = np.inf
    
        if 'min_max' in self.loss_type:
            min_max = masked_min_max_loss(rec, target, patch_size=self.patch_size, mask=patched_mask)

        if 'mae' in self.loss_type:
            mae = masked_mae_loss(rec, target, mask=mask)
        else:
            with torch.no_grad(): mae = masked_mae_loss(rec, target, mask=mask)

        if 'grad' in self.loss_type:
            grad = gradient_loss(rec, target, mask=mask)
        else:
            with torch.no_grad(): grad = gradient_loss(rec, target, mask=mask)

        if 'mse' in self.loss_type:
            mse = masked_mse_loss(rec, target, reduction='mean', mask=mask)
        else:
            with torch.no_grad(): mse = masked_mse_loss(rec, target, reduction='mean', mask=mask)
   
        # calculate the normalized root squared error only for the first token prediction
        with torch.no_grad():
            nrmse = torch.sqrt(mse) / (target.max() - target.min())

        return nrmse, mse, mae, grad, min_max

    def next_token_prediction(self, batch):
        x = self.pad(batch["signal"])

        reconstruction, out_teacher, last_emb = self.model(x)

        x = x[:, self.patch_size:].squeeze()
        reconstruction = reconstruction[:, :-self.patch_size]

        if self.use_teacher_student:
            out_teacher = out_teacher[:, 1:, :] # [bs, seq_len -1, num_hiddens]
            last_emb = last_emb[:, :-1, :] # [bs, seq_len -1, num_hiddens]
            return x, reconstruction, out_teacher, last_emb
        
        return x, reconstruction, None, None
    
    def get_feature_data(self, dataloader, feature_classification=False):
        all_features = {}
        all_labels = []
        for batch in dataloader:
            signal = batch["signals"]
            age = batch["ages"].to(self.device) if batch["ages"] is not None else None
            gender = batch["genders"].to(self.device) if batch["genders"] is not None else None
            self.model.eval()
            features = self.model.get_features(signal.to(self.device), feature_classification=feature_classification, age=age, gender=gender)
            for k, v in features.items():
                if k not in all_features:
                    all_features[k] = []
                
                if feature_classification:
                    all_features[k].append(v.reshape(-1, v.shape[-1]).detach().cpu())
                else:
                    all_features[k].append(v.detach().cpu())

            if feature_classification:
                all_labels.append(batch['labels'].reshape(-1).detach().cpu())
            else:
                all_labels.append(batch['class_labels'].detach().cpu())

        output = {}

        for k in all_features:
            output[k] = torch.cat(all_features[k]).numpy()
        output["label"] = torch.cat(all_labels).numpy()

        return output

    def evaluate_on_model_type(self, train_data: dict[str, np.array], val_data: dict[str, np.array], model_name: str, task_name: str, model_class, model_config: dict):
        y_train = train_data["label"]
        y_val = val_data["label"]

        print(f"Evaluating {model_name} on {task_name} task")

        # all key that are not 'label'
        feature_types = [k for k in train_data if k != "label"]
        for feature_type in feature_types:
            x_train = train_data[feature_type]
            x_val = val_data[feature_type]

            model = model_class(**model_config)
            model = OneVsRestClassifier(model, n_jobs=-1)
            model.fit(x_train, y_train)

            val_pred = model.predict(x_val)
            train_pred = model.predict(x_train)

            f1_val = f1_score(y_val, val_pred, average='macro')
            f1_train = f1_score(y_train, train_pred, average='macro')

            self.log(f'{task_name}/train_{feature_type}_{model_name}_f1', f1_train)
            self.log(f'{task_name}/val_{feature_type}_{model_name}_f1', f1_val)

    def eval_model_downstream_ptb_xl(self):
        """
        Evaluate the quality of model features using KNN on a classification task, e.g. PTB-XL superclasses.
        """
        if self.ptb_xl_train_dataloader is None or self.ptb_xl_val_dataloader is None:
            return
        
        with torch.no_grad():
            train_data_ptb_xl = self.get_feature_data(self.ptb_xl_train_dataloader, feature_classification=False)
            val_data_ptb_xl = self.get_feature_data(self.ptb_xl_val_dataloader, feature_classification=False)

        mlp_config = {
            "hidden_layer_sizes": [256, 128, 64],
            "random_state": 42,
            "max_iter": 64,
            "early_stopping": True,
        }

        linear_probe_config = {
            "random_state": 42,
            "C": 1.0,
            "max_iter": 256,
        }

        # self.evaluate_on_model_type(train_data_ptb_xl, val_data_ptb_xl, "mlp", 'ptb-xl', MLPClassifier, mlp_config)
        self.evaluate_on_model_type(train_data_ptb_xl, val_data_ptb_xl, "lp", 'ptb-xl', LogisticRegression, linear_probe_config)

        del train_data_ptb_xl
        del val_data_ptb_xl

    def eval_model_downstream_mit_bih(self):
        """
        Evaluate the quality of model features using KNN on a classification task, e.g. PTB-XL superclasses.
        """
        if self.mit_bih_train_dataloader is None or self.mit_bih_val_dataloader is None:
            return
        
        with torch.no_grad():
            train_data_mit_bih = self.get_feature_data(self.mit_bih_train_dataloader, feature_classification=True)
            val_data_mit_bih = self.get_feature_data(self.mit_bih_val_dataloader, feature_classification=True)

        # mlp_config = {
        #     "hidden_layer_sizes": [256, 128, 64],
        #     "random_state": 42,
        #     "max_iter": 16,
        #     "early_stopping": True,
        # }

        linear_probe_config = {
            "random_state": 42,
            "max_iter": 32,
            "early_stopping": True
        }

        # self.evaluate_on_model_type(train_data_mit_bih, val_data_mit_bih, "mlp", 'mit-bih', MLPClassifier, mlp_config)
        self.evaluate_on_model_type(train_data_mit_bih, val_data_mit_bih, "lp", 'mit-bih', Perceptron, linear_probe_config)

        del train_data_mit_bih
        del val_data_mit_bih
    
    def get_params(self):
        if self.layerwise_lr_decay > 0.:
            params = [ ]
            num_layers = len(self.model.core.model.blocks) + 1

            # Assign learning rates to each transformer layer
            for i, layer in enumerate(self.model.core.model.blocks):
                layer_lr = self.lr * (self.layerwise_lr_decay ** (num_layers - i - 1))  # Earlier layers get smaller LR
                layer_params = layer.parameters()
                params.append({"params": layer_params, "lr": layer_lr, "name": f"layer_{i}"})

            layer_lr = self.lr * (self.layerwise_lr_decay ** num_layers)
            params.append({"params": self.model.patch_embedding.parameters(), "lr": layer_lr, "name": "patch_embedding"})

            params.append({'params': self.model.mask_token, 'lr': self.lr, 'weight_decay': self.wd, 'name': 'mask_token'})

            if self.model.encoder_type =='large':
                params.append({'params': self.model.core.model.out_norm.parameters(), 'lr': self.lr, 'weight_decay': self.wd, 'name': 'ln2'})
            else:
                params.append({'params': self.model.core.model.post_blocks_norm.parameters(), 'lr': self.lr, 'weight_decay': self.wd, 'name': 'ln2'})

            if self.model.cls_type == 'token' or self.model.cls_type == 'token_2':
                params.append({'params': self.model.cls_token, 'lr': self.lr, 'weight_decay': self.wd, 'name': 'cls'})
            elif self.model.cls_type == 'attn_pool' or self.model.cls_type == 'lin_attn_pool':
                params.append({'params': self.model.attn_pool.parameters(), 'lr': self.lr, 'weight_decay': self.wd, 'name': 'cls'})

            if self.model.num_reg_tokens > 0:
                params.append({'params': self.reg_token, 'lr': self.lr, 'weight_decay': self.wd, 'name': 'reg_tokens'})
        else:
            params = [
                {'params': self.model.trainable_parameters(), 'lr': self.lf, 'weight_decay': self.wd},
            ]
        return params
    
    def get_lr(self):
        return self.lr
    
    def get_reconstruction_lr(self):
        return self.reconstruction_lr

    def configure_optimizers(self):
        if self.automatic_optimization == False:
            return configure_optimizer_teacher_student(self)
        else:
            return configure_optimizers(self)