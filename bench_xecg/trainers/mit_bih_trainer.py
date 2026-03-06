import os

from torch import nn
import torchmetrics
import numpy as np
import torch
import matplotlib.pyplot as plt
import lightning.pytorch as pl

from .common_trainer import CommonTrainerDownstream


class TrainingMIT_BIH(CommonTrainerDownstream):
    def __init__(self, model, config,  len_train_dataset, weights=None):
        super().__init__(model, config,  len_train_dataset, weights)
            
        self.plot_predictions = config.plot_predictions

        self.predict_no_hb = config.predict_no_hb
        if self.predict_no_hb:
            self.num_classes -= 1

        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, average='micro', ignore_index=-1, top_k=1)
        self.valid_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, average='micro', ignore_index=-1, top_k=1)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, average='micro', ignore_index=-1, top_k=1)
        self.test_acc_no_avg = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes, average=None, ignore_index=-1, top_k=1)
        self.train_f1 = torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='macro', ignore_index=-1, top_k=1)
        self.valid_f1 = torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average='macro', ignore_index=-1, top_k=1)
        self.test_f1 = torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average=None, ignore_index=-1, top_k=1)

        self.train_auroc = torchmetrics.AUROC(num_classes=self.num_classes, ignore_index=-1, task='multiclass')
        self.valid_auroc = torchmetrics.AUROC(num_classes=self.num_classes, ignore_index=-1, task='multiclass')
        self.test_auroc = torchmetrics.AUROC(num_classes=self.num_classes, ignore_index=-1, task='multiclass')

        self.single_hb = config.single_hb

        # add sensitivity and specificity for the first class
        self.val_spec = torchmetrics.Specificity(num_classes=self.num_classes, average=None, ignore_index=-1, task='multiclass', top_k=1)
        self.test_spec = torchmetrics.Specificity(num_classes=self.num_classes, average=None, ignore_index=-1, task='multiclass', top_k=1)
        self.val_recall = torchmetrics.Recall(num_classes=self.num_classes, average=None, ignore_index=-1, task='multiclass', top_k=1)
        self.test_recall = torchmetrics.Recall(num_classes=self.num_classes, average=None, ignore_index=-1, task='multiclass', top_k=1)
        self.val_precision = torchmetrics.Precision(num_classes=self.num_classes, average=None, ignore_index=-1, task='multiclass', top_k=1)
        self.test_precision = torchmetrics.Precision(num_classes=self.num_classes, average=None, ignore_index=-1, task='multiclass', top_k=1)

    def log_perclass_metric(self, metric, metric_name, step='val'):
        self.log(f"{step}_{metric_name}/N", metric[0], metric_attribute=metric_name)
        self.log(f"{step}_{metric_name}/S", metric[1], metric_attribute=metric_name)
        self.log(f"{step}_{metric_name}/V", metric[2], metric_attribute=metric_name)
        if self.num_classes == 5:
            self.log(f"{step}_{metric_name}/F", metric[3], metric_attribute=metric_name)
            self.log(f"{step}_{metric_name}/Q", metric[4], metric_attribute=metric_name)
        
        self.log(f"{step}_{metric_name}/mean", metric.mean(), metric_attribute=metric_name)

    def training_step(self, batch, _):
        loss_cls, preds, targets, logits = self.predict_batch(batch)
        self.log('train_loss', loss_cls.detach().item(), prog_bar=True)

        self.train_acc(preds.flatten(), targets.flatten())
        self.log('train_acc', self.train_acc, prog_bar=True)

        self.train_f1(preds.flatten(), targets.flatten())
        self.log('train_f1', self.train_f1, prog_bar=True)

        self.train_auroc(logits, targets.long())
        self.log("train_auroc", self.train_auroc)

        return loss_cls 
    
    def validation_step(self, batch, _):
        loss_cls, preds, targets, logits = self.predict_batch(batch)
        self.log('val_loss', loss_cls.detach().item(), prog_bar=True)

        self.valid_acc(preds.flatten(), targets.flatten())
        self.log('val_acc', self.valid_acc, prog_bar=True)

        self.valid_f1(preds.flatten(), targets.flatten())
        self.log('val_f1', self.valid_f1, prog_bar=True)

        # specificity
        specificity = self.val_spec(preds, targets)
        self.log_perclass_metric(specificity, 'specificity', step='val')

        # sensitivity
        recall = self.val_recall(preds, targets)
        self.log_perclass_metric(recall, 'sensitivity', step='val')

        # ppv
        ppv = self.val_precision(preds, targets)
        self.log_perclass_metric(ppv, 'ppv', step='val')

        # auroc
        self.valid_auroc(logits, targets.long())
        self.log('val_auroc', self.valid_auroc, prog_bar=True)

        return loss_cls
            
    def test_step(self, batch, _):
        loss_cls, preds, targets, logits = self.predict_batch(batch)
        self.log("test_loss", loss_cls.detach().item())

        self.test_acc(preds.flatten(), targets.flatten())
        self.log("test_acc", self.test_acc)

        acc = self.test_acc_no_avg(preds.flatten(), targets.flatten())
        self.log_perclass_metric(acc, 'acc', step='test')

        f1 = self.test_f1(preds.flatten(), targets.flatten())
        self.log_perclass_metric(f1, 'f1', step='test')

        specificity = self.test_spec(preds, targets)
        self.log_perclass_metric(specificity, 'specificity', step='test')

        recall = self.test_recall(preds, targets)
        self.log_perclass_metric(recall, 'sensitivity', step='test')

        ppv = self.test_precision(preds, targets)
        self.log_perclass_metric(ppv, 'ppv', step='test')

        # auroc  
        self.test_auroc(logits, targets.long())
        self.log("test_auroc", self.test_auroc)

        return loss_cls 
    
    def plot_predictions_if_needed(self, dataloader, step='train'):
        if self.plot_predictions:
            try:
                idx_1 = 0
                idx_2 = 21
                idx_3 = np.random.randint(0, len(dataloader.dataset))
                sample_1 = dataloader.dataset[idx_1]
                sample_2 = dataloader.dataset[idx_2]
                sample_3 = dataloader.dataset[idx_3]
                log_dir = self.logger.log_dir if self.logger is not None and self.logger.log_dir is not None else 'figs/'
                img_1 = self.plot_mit_bih_pred(sample_1, log_dir, f'mit_1_{step}')
                img_2 = self.plot_mit_bih_pred(sample_2, log_dir, f'mit_2_{step}')
                img_3 = self.plot_mit_bih_pred(sample_3, log_dir, f'mit_3_{step}')

                if isinstance(self.logger, pl.loggers.WandbLogger):
                    self.logger.log_image(key=f"reconstructions_{step}", images=[img_1, img_2, img_3])
            except Exception as e:
                # print stack trace
                import traceback
                traceback.print_exc()
                print(f"Error plotting R-peaks: {e}")
    
    def on_train_epoch_end(self):
        self.plot_predictions_if_needed(self.trainer.train_dataloader, step='train')
        super().on_train_epoch_end()

    def on_validation_epoch_end(self):
        self.plot_predictions_if_needed(self.trainer.val_dataloaders, step='val')
        super().on_test_epoch_end()


    def on_test_epoch_end(self):
        self.plot_predictions_if_needed(self.trainer.test_dataloaders, step='test')
        super().on_test_epoch_end()

    def predict_batch(self, batch):
        x = batch["signals"]
        targets = batch['labels'].long()
        age = batch['ages']
        gender = batch['genders']

        if self.linear_probing:
            self.model.set_eval_linear_probing()

        if self.predict_no_hb:
            targets = targets + 1

        if self.single_hb:
            cls = self.model(x, age=age, gender=gender) # [bs, 1, num_classes]
            targets = targets.squeeze()
            loss_cls = nn.functional.cross_entropy(cls, targets, weight=self.weights, ignore_index=-1)
            preds = torch.argmax(cls, dim=-1)
        else:
            cls = self.model(x, age=age, gender=gender).permute(0, 2, 1)
            #print('cls shape:', cls.shape)
            #print('target shape:', targets.shape)
            loss_cls = nn.functional.cross_entropy(cls, targets, weight=self.weights, ignore_index=-1)
            preds = torch.argmax(cls, dim=-2)

        if self.use_focal_loss:
            pt = torch.exp(-loss_cls)
            alpha = 2.
            gamma = .25
            loss_cls = (alpha * (1-pt)**gamma * loss_cls)

        if self.predict_no_hb:
            preds = preds - 1
            preds[preds == -1] = 0
            targets = targets - 1
            cls = cls[..., 1:, :]
        
        return loss_cls, preds, targets, cls

    def plot_mit_bih_pred(self, sample, logdir, name):
        with torch.no_grad():
            signal = torch.from_numpy(sample['signal'].copy()).to(self.device).unsqueeze(0).float()
            targets = torch.from_numpy(sample['label']).to(self.device).unsqueeze(0)

            predicted = self.model(signal)
            targets = targets.unfold(1, self.model.patch_size, self.model.patch_size).max(dim=-1)[0].long()

            # consider max 2000 time samples for plotting
            # if signal.shape[1] > max_length:
            #     signal = signal[:, :max_length, :]
            #    target = target[:, :max_length]

            # targets = target.unfold(1, model.patch_size, model.patch_size).max(dim=-1)[0].long()
            fig, ax = plt.subplots(figsize=(25, 5))

            to_plot = signal[:, :, 1].cpu().squeeze().numpy() if signal.ndim > 2 else signal.cpu().squeeze().numpy()
            min = to_plot.min()
            max = to_plot.max()

            ax.plot(to_plot, label='Original Signal')

            # Plot vertical lines at each patch
            for j in range(0, signal.shape[1], self.model.patch_size):
                ax.axvline(j, color='gray', linestyle='--', linewidth=0.5)

            # inside each patch plot the prediction above and the target below

            for i in range(targets.shape[1]):
                target_class = targets[0, i].item()
                if target_class == -1: continue

                patch_start = i * self.model.patch_size
                patch_end = patch_start + self.model.patch_size

                # get the max index of the prediction
                pred_class = torch.argmax(predicted[0, i]).item() 
                if self.predict_no_hb:
                    pred_class = pred_class - 1
                    
                ax.text((patch_start + patch_end) / 2, max - 0.1,
                        f'{get_label(pred_class)}',
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=18,
                        color='green',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
                # plot the target class below the patch
                ax.text((patch_start + patch_end) / 2, min + 0.1,
                        f'{get_label(target_class)}',  
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=18,
                        color='orange',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
                
                # color the patch background if the prediction is correct or not
                if target_class != -1:
                    ax.axvspan(patch_start, patch_end, color='green' if pred_class == target_class else 'red', alpha=0.1)
                
            # ax.set_title('MIT-BIH ECG Signal with Predictions and Targets')
            ax.set_xlabel('Timepoints', fontdict={'size': 24})
            ax.set_ylabel('Amplitude', fontdict={'size': 24})
            ax.tick_params(axis='x', labelsize=20)
            ax.tick_params(axis='y', labelsize=20)
            # ax.legend()

            # mkdir if it does not exist
            os.makedirs(f'{logdir}/epoch_{self.current_epoch}', exist_ok=True)

            path = f'{logdir}/epoch_{self.current_epoch}/r_peaks_{name}.png'
            plt.tight_layout()
            plt.savefig(path, dpi=300)
            plt.close()
            return path
    

def get_label(label):
    if label == 0: return 'N'
    if label == 1: return 'S'
    if label == 2: return 'V'
    if label == 3: return 'F'
    if label == 4: return 'Q'
    if label == -1: return ' '
    else: raise ValueError(f'Unknown label {label}')