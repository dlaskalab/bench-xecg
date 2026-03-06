import copy

import torch
from torch import nn

from .pooling import AttentionPooling, LinearAttentionPooling
from .base_model import BaseModel
from .utils import get_normalization_layer, get_xlstm, get_large_xlstm, get_patch_embedding, get_reconstruction_head,  get_transformer

class pretrainedxLSTM(BaseModel):
    def __init__(
            self, 
            num_channels,
            config,
            reconstruction=True
        ): 
        super(pretrainedxLSTM, self).__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.patch_size = config.patch_size
        self.bidirectional = config.bidirectional
        self.use_teacher_student = config.strategy == 'lejepa'
        self.mask_ratio = config.mask_ratio
        self.embedding_size = config.embedding_size
        self.cls_type = config.cls_type
        self.masking_type = config.masking_type
        self.encoder_type = config.encoder_type
        self.sampling_freq = config.sampling_freq
        self.linear_probing = config.linear_probing
        self.use_age_and_gender = config.use_age_and_gender
        self.keep_age_gender_tokens = config.keep_age_gender_tokens

        self.patch_embedding = get_patch_embedding(config.patch_embedding, config.patch_size, config.embedding_size, num_channels)

        if config.encoder_type == 'large':
            self.core = get_large_xlstm(config)
        elif config.encoder_type == 'transformer':
            self.core = get_transformer(config)
        else:
            self.core = get_xlstm(config)

        self.mask_token = nn.Parameter(torch.zeros(config.embedding_size))
        
        if self.cls_type == 'token' or self.cls_type == 'token_2':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embedding_size))
            nn.init.xavier_uniform_(self.cls_token, gain=1.0)
        elif self.cls_type == 'attn_pool':
            self.attn_pool = AttentionPooling(config.embedding_size, config.num_heads)
        elif self.cls_type == 'lin_attn_pool':
            self.attn_pool = LinearAttentionPooling(config.embedding_size)
            
        self.num_reg_tokens = config.num_reg_token
        if config.num_reg_token > 0:
            self.reg_token = nn.Parameter(torch.zeros(1, config.num_reg_token, config.embedding_size))
            nn.init.xavier_uniform_(self.reg_token, gain=1.0)

        if self.use_age_and_gender:
            # embedding for age buckets for every 5 years, from 18 to 90
            self.age_embedding = nn.Embedding(16, config.embedding_size, padding_idx=0)
            self.gender_embedding = nn.Embedding(3, config.embedding_size, padding_idx=0)
          
        if reconstruction:
            self.reconstruction = get_reconstruction_head(config.patch_size, config.embedding_size, num_channels)

        self.normalization_layer = get_normalization_layer(config, config.embedding_size)

        if self.linear_probing:
            # freezing model
            for name, param in self.named_parameters():
                if 'head' not in name:  # no freezing last layer
                    param.requires_grad = False

    def init_teacher(self):
        self._teacher = self.create_teacher_module()

    def create_teacher_module(self):
        param = copy.deepcopy(self)
        # remove reconstruction params
        for name in list(param._modules.keys()):
            if 'reconstruction' in name or 'teacher' in name:
                print(f'removing {name} from teacher network')
                del param._modules[name]

        for param_t in param.parameters():
            param_t.requires_grad = False

        param.eval()
        return param
    
    def pooling(self, out, padding_mask=None, pooling_type='avg'):
        cls = None
        if pooling_type == 'max':
            if padding_mask is None:
                cls = out.max(dim=1)[0]
            else:
                cls = out.masked_fill(padding_mask, -torch.inf).max(dim=1)[0]
        elif pooling_type == 'mean' or pooling_type == 'avg':
            if padding_mask is None:
                cls = out.mean(dim=1)
            else:
                cls = out.masked_fill(padding_mask, 0).sum(dim=1) / (out.shape[1] - padding_mask.sum(dim=1)).clamp(min=1)

        elif pooling_type == 'mix':
            if padding_mask is None:
                max_p = out.max(dim=1)[0]
                mean_p = out.mean(dim=1)
            else:
                max_p = out.masked_fill(padding_mask, -torch.inf).max(dim=1)[0]
                mean_p = out.masked_fill(padding_mask, 0).sum(dim=1) / (out.shape[1] - padding_mask.sum(dim=1)).clamp(min=1)
            cls = torch.cat([max_p, mean_p], dim=-1)
        elif pooling_type == 'token':
            cls = out[:, -1, :]
            out = out[:, :-1, :]
        elif pooling_type == 'token_2':
            cls_1 = out[:, 0, :]
            cls_2 = out[:, -1, :]
            out = out[:, 1:-1, :]
            cls = cls_1 + cls_2
        elif pooling_type == 'attn_pool' or pooling_type == 'lin_attn_pool':
            if padding_mask is None:
                cls = self.attn_pool(out).squeeze()
            else:
                cls = self.attn_pool(out.masked_fill(padding_mask, 0)).squeeze()  
        else:
            return cls, out

        cls = self.normalization_layer(cls)    
        return cls, out
    
    def forward_core(self, x, padding_mask=None):
        # add the [cls] and [reg] tokens
        if self.cls_type == 'token':
            x = self.add_cls_token(x)
        elif self.cls_type == 'token_2':
            x = self.add_cls_token_2(x)

        if self.num_reg_tokens > 0:
            x = self.add_reg_tokens(x)

        # pass to xlstm
        out = self.core(x, need_expansion=False) # [batch_size, embedding_dim]

        if self.num_reg_tokens > 0:
            out = self.remove_reg_tokens(out)

        cls, out = self.pooling(out, padding_mask, pooling_type=self.cls_type)
        return cls, out

    def forward(self, x, masking=True, reconstruct=True, age=None, gender=None):
        # padding_mask = self.get_padding_mask(x)

        # mask the signal if needed and get the patch embeddings
        x_emb, mask = self.embed_and_mask_signal_if_needed(x, masking, age=age, gender=gender)
        
        # forward on the core xlstm
        cls, out = self.forward_core(x_emb) #, padding_mask=padding_mask)

        if not self.keep_age_gender_tokens:
            out = self.remove_age_gender_embeddings(out, age=age, gender=gender)

        # reconstruct signal
        if reconstruct:
            if self.keep_age_gender_tokens:
                out_rec_input = self.remove_age_gender_embeddings(out, age=age, gender=gender)
                rec, _ = self.reconstruction(out_rec_input.clone().detach())
            else:
                rec, _ = self.reconstruction(out.clone().detach())

        tortn = {
            'patches': out,
            'cls': cls,
        }
        
        if masking: tortn['mask'] = mask
        if reconstruct: tortn['reconstruction'] = rec
        
        return tortn
    
    @torch.no_grad()
    def teacher_fwd(self, x, age=None, gender=None):
        return self._teacher(x, masking=False, reconstruct=False, age=age, gender=gender)
         
    def embed_and_mask_signal_if_needed(self, x, masking:bool=False, age=None, gender=None):
        # patching
        x_patches = self.patch_embedding(x)

        # add age and gender embeddings
        x_age_gen = self.get_age_gender_embeddings(x_patches, age=age, gender=gender)

        generated_mask = None
        if masking:
            # 4. Calculate mask based on the EMBEDDINGS, not the raw signal
            # We pass 'x' optionally just to calculate where the padding is
            generated_mask = self.get_random_mask(x_patches, raw_x=x) # True means "masked/replace with token"
            
            # 5. Apply the mask token
            # Expand mask token to match batch and sequence dimensions
            mask_token_expanded = self.mask_token.expand_as(x_patches)
            
            # Replace embeddings with mask_token where mask is True
            x_patches = torch.where(
                generated_mask, 
                mask_token_expanded, 
                x_patches
            )

            if x_age_gen is not None:
                age_gen_mask = self.get_random_mask_for_age_gender(x_age_gen)
                mask_token_expanded_age_gen = self.mask_token.expand_as(x_age_gen)
                x_age_gen = torch.where(
                    age_gen_mask,
                    mask_token_expanded_age_gen,
                    x_age_gen
                )

        # We always concatenate embeddings if aux tokens exist
        if x_age_gen is not None:
            x_final = torch.cat([x_age_gen, x_patches], dim=1)
            
            # 4. Concatenate Masks (Only if masking was active)
            if generated_mask is not None:
                # If we have age tokens, we must have an age mask (even if it's all False, 
                # but here it comes from the loop above).
                # Sanity check to ensure age_gen_mask exists if we are in this block
                if age_gen_mask is None:
                    # This happens if masking=True but x_age_gen was empty (logic handled by outer if)
                    # OR if logic failed. 
                    # If x_age_gen > 0 and masking=True, age_gen_mask is calculated above.
                    pass 
                
                final_mask = torch.cat([age_gen_mask, generated_mask], dim=1)
            else:
                final_mask = None
        else:
            x_final = x_patches
            final_mask = generated_mask

        return x_final, final_mask
    
    def get_random_mask(self, x_emb, raw_x=None):
        """
        Return a mask of the same shape as x_emb (Batch, Num_Patches, 1).
        Masked values (to be replaced) are set to TRUE.
        """
        batch_size, num_patches, _ = x_emb.shape
        
        # Determine padding based on raw_x if provided
        if raw_x is not None:
            # Reshape raw_x to [Batch, Num_Patches, Patch_Size, Channels]
            # to check if a specific patch consists entirely of padding (0s)
            x_reshaped = raw_x.reshape(batch_size, num_patches, self.patch_size, -1)
            # If the sum of absolute values in a patch is 0, it is padding
            is_padding = (x_reshaped.abs().sum(dim=(2, 3)) == 0).unsqueeze(-1)
        else:
            # Fallback if raw_x isn't passed (assume no padding)
            is_padding = torch.zeros(batch_size, num_patches, 1, device=x_emb.device, dtype=torch.bool)

        # Generate Random Mask (Shapes are now naturally [Batch, Num_Patches])
        if self.masking_type == 'random':
            rand = torch.rand(batch_size, num_patches, device=x_emb.device)
            mask = (rand < self.mask_ratio) # True for masked
            mask = mask.unsqueeze(-1)       # [Batch, Num_Patches, 1]
            
        elif self.masking_type == 'block':
            rand = torch.rand(batch_size, num_patches, device=x_emb.device)
            mask = (rand < self.mask_ratio / 4) 
            # after a masked patch, the next 3 patches are masked
            for i in range(1, 4):
                mask = mask | mask.roll(-1, dims=1)
            mask = mask.unsqueeze(-1)       # [Batch, Num_Patches, 1]
        
        else:
            # Fallback (no masking)
            mask = torch.zeros(batch_size, num_patches, 1, device=x_emb.device, dtype=torch.bool)

        # Do NOT mask positions that are actually padding (keep them as original embeddings/zeros)
        return mask & ~is_padding
    
    def get_random_mask_for_age_gender(self, x_age_gen):
        batch_size, num_tokens, _ = x_age_gen.shape
        # Vectorized implementation
        rand = torch.rand(batch_size, num_tokens, device=x_age_gen.device)
        mask = (rand < self.mask_ratio)
        return mask.unsqueeze(-1) # [Batch, Num_Tokens, 1]

    def get_age_token(self, age):
        # from 15 to 85+ in 5 year increments,
        age_bucket = ((age.clamp(15, 85) - 15) // 5).long()

        # Shift by 1 so valid data is in range [1, 15]
        age_bucket += 1

        # Fill NaNs with 0 (the designated 'missing' index)
        age_bucket = age_bucket.masked_fill(age.isnan(), 0)

        age_emb = self.age_embedding(age_bucket)
        return age_emb

    def remove_age_gender_embeddings(self, x, age=None, gender=None):
        if age is not None and self.use_age_and_gender:
            x = x[:, 1:, :]
        if gender is not None and self.use_age_and_gender:
            x = x[:, 1:, :]
        return x
    
    def get_gender_token(self, gender):
        # Female -> 0 -> 1, Male -> 1 -> 2, NaN -> 0
        gender_idx = gender.long() + 1
        gender_idx = gender_idx.masked_fill(gender.isnan(), 0)

        gender_emb = self.gender_embedding(gender_idx)
        return gender_emb


    def get_age_gender_embeddings(self, x, age=None, gender=None):
        # add age and gender embeddings
        if not self.use_age_and_gender:
            return x
        
        embs = torch.zeros(x.shape[0], 0, self.embedding_size, device=x.device)
        if age is not None:
            age_emb = self.get_age_token(age)
            # print("age_emb shape: ", age_emb.shape)
            # print("x shape before adding age embedding: ", x.shape)
            embs = torch.cat([age_emb.unsqueeze(1), embs], dim=1)

        if gender is not None:
            gender_emb = self.get_gender_token(gender)
            embs = torch.cat([gender_emb.unsqueeze(1), embs], dim=1)

        if embs.shape[1] == 0:
            return None
        else:
            return embs

    def add_reg_tokens(self, x):
        reg_tokens = self.reg_token.expand(x.shape[0], -1, -1)
        half = self.num_reg_tokens // 2
        return torch.cat([reg_tokens[:, :half, :], x, reg_tokens[:, half:, :]], dim=1)
    
    def remove_reg_tokens(self, x):
        half = self.num_reg_tokens // 2
        return x[:, half:-(self.num_reg_tokens - half), :]
    
    def add_cls_token(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat([cls_token, x], dim=1)
    
    def add_cls_token_2(self, x):
        cls_token_1 = self.cls_token.expand(x.shape[0], -1, -1)
        cls_token_2 = self.cls_token.expand(x.shape[0], -1, -1)
        return torch.cat([cls_token_1, x, cls_token_2], dim=1)
    
    def get_padding_mask(self, x):
        """
        Return a padding mask of shape [batch_size, num_patches, embedding_size]
        where masked values are set to TRUE
        """
        padding_mask = (x.abs().sum(dim=-1) == 0).unsqueeze(-1)
        num_patches = x.shape[-2] // self.patch_size
        padding_mask_patched = padding_mask.view(-1, num_patches, self.patch_size)[:, :, 0].unsqueeze(-1).expand(-1, -1, self.embedding_size)
        return padding_mask_patched

    
    def trainable_parameters(self):
        if self.use_teacher_student:
            return [param for name, param in self.named_parameters() if "teacher" not in name and 'reconstruction' not in name]
        
        return self.parameters()

    def get_features(self, x, feature_classification=False, age=None, gender=None):
        """
        This function should be the complete forward pass apart from the classification head.
        """
                # mask the signal if needed and get the patch embeddings
        x_emb, _ = self.embed_and_mask_signal_if_needed(x, age=age, gender=gender)

        cls, out = self.forward_core(x_emb)

        if not self.keep_age_gender_tokens:
            out = self.remove_age_gender_embeddings(out, age=age, gender=gender)

        if feature_classification:
            return {'feat': out}
        
        tortn = {}

        if self.cls_type != 'avg' and self.cls_type != 'mean':
            avg, _ = self.pooling(out, pooling_type='avg')
            tortn['avg'] = avg
        else:
            tortn['avg'] = cls
        
        if self.cls_type != 'max':
            max, _ = self.pooling(out, pooling_type='max')
            tortn['max'] = max
        else:
            tortn['max'] = cls

        if self.cls_type == 'attn_pool':
            tortn['attn_pool'] = cls

        if self.cls_type == 'token' or self.cls_type == 'token_2':
            tortn['token'] = cls
        
        return tortn

    
    def get_layers(self):
        """
        This function should return the layers of the model where to apply the layerwise decay
        """
        return self.core.model.blocks
    
    def additional_params(self, lr, last_layer_lr, wd):
        """
        This fucntion should return additional parameters used by a model (like classification token and so on...)
        """
        params = []
        params.append({"params": self.patch_embedding.parameters(), "lr": last_layer_lr, "name": "patch_embedding"})

        if self.encoder_type =='large':
            params.append({'params': self.core.model.out_norm.parameters(), 'lr': lr, 'weight_decay': wd, 'name': 'ln2'})
        else:
            params.append({'params': self.core.model.post_blocks_norm.parameters(), 'lr': lr, 'weight_decay': wd, 'name': 'ln2'})

        if self.cls_type == 'token' or self.cls_type == 'token_2':
            params.append({'params': self.cls_token, 'lr': lr, 'weight_decay': wd, 'name': 'cls'})
        elif self.cls_type == 'attn_pool' or self.cls_type == 'lin_attn_pool':
            params.append({'params': self.attn_pool.parameters(), 'lr': lr, 'weight_decay': wd, 'name': 'cls'})

        if self.use_age_and_gender:
            params.append({'params': self.age_embedding.parameters(), 'lr': lr, 'weight_decay': wd, 'name': 'age_emb'})
            params.append({'params': self.gender_embedding.parameters(), 'lr': lr, 'weight_decay': wd, 'name': 'gender_emb'})

        if self.num_reg_tokens > 0:
            params.append({'params': self.reg_token, 'lr': lr, 'weight_decay': wd, 'name': 'reg_tokens'})

        if hasattr(self.core, 'post_blocks_norm'):
            params.append({'params': self.core.post_blocks_norm, 'lr': lr, 'name': 'post_block_norm'})

        return params
        
