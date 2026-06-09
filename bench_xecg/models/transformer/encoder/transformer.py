# Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# vit_pytorch: https://github.com/lucidrains/vit-pytorch
# --------------------------------------------------------

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from bench_xecg.models.transformer.encoder.vit import TransformerBlock


__all__ = ['Transformer', 'vit_small', 'vit_base']


class Transformer(nn.Module):
    def __init__(self,
                 seq_len: int,
                 patch_size: int,
                 num_leads: int,
                 num_classes: Optional[int] = None,
                 width: int = 768,
                 depth: int = 12,
                 mlp_dim: int = 3072,
                 heads: int = 12,
                 dim_head: int = 64,
                 qkv_bias: bool = True,
                 drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0.,
                 drop_path_rate: float = 0,
                 use_final_layer_norm: bool = True):
        super().__init__()
        assert seq_len % patch_size == 0, 'The sequence length must be divisible by the patch size. But got ' \
                                          f'seq_len={seq_len} and patch_size={patch_size}.'
        self._repr_dict = {'seq_len': seq_len,
                           'patch_size': patch_size,
                           'num_leads': num_leads,
                           'num_classes': num_classes if num_classes is not None else 'None',
                           'width': width,
                           'depth': depth,
                           'mlp_dim': mlp_dim,
                           'heads': heads,
                           'dim_head': dim_head,
                           'qkv_bias': qkv_bias,
                           'drop_out_rate': drop_out_rate,
                           'attn_drop_out_rate': attn_drop_out_rate,
                           'drop_path_rate': drop_path_rate,
                           }
        self.width = width
        self.depth = depth

        # embedding layers
        num_patches = seq_len // patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, width))

        # transformer layers
        drop_path_rate_list = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.model = nn.Module()
        self.model.add_module('blocks', nn.ModuleList())

        for i in range(depth):
            block = TransformerBlock(input_dim=width,
                                     output_dim=width,
                                     hidden_dim=mlp_dim,
                                     heads=heads,
                                     dim_head=dim_head,
                                     qkv_bias=qkv_bias,
                                     drop_out_rate=drop_out_rate,
                                     attn_drop_out_rate=attn_drop_out_rate,
                                     drop_path_rate=drop_path_rate_list[i])
            self.model.blocks.append(block)

        self.dropout = nn.Dropout(drop_out_rate)
        self.model.post_blocks_norm = nn.LayerNorm(width) if use_final_layer_norm else nn.Identity()

    def forward(self, x, need_expansion=False):
        bs, n, emb = x.shape

        # cut the signal if needed
        if n >= self.pos_embedding.shape[1]:
            x = x[:, :self.pos_embedding.shape[1] -1, :]
            print(f'Input sequence length {n} is longer than the maximum sequence length {self.pos_embedding.shape[1]-1}. Cutting the input to match the maximum sequence length.')

        x = x + self.pos_embedding[:, 1:n + 1, :]

        x = self.dropout(x)
        for i in range(self.depth):
            x = self.model.blocks[i](x)

        return self.model.post_blocks_norm(x)


    def __repr__(self):
        print_str = f"{self.__class__.__name__}(\n"
        for k, v in self._repr_dict.items():
            print_str += f'    {k}={v},\n'
        print_str += ')'
        return print_str

    def training_params(self):
        return self.head.parameters()
    
    def set_eval_linear_probing(self):
        self.eval()
        self.head.train()


def vit_small(num_leads, num_classes=None, seq_len=2250, patch_size=75, **kwargs):
    model_args = dict(seq_len=seq_len,
                      patch_size=patch_size,
                      num_leads=num_leads,
                      num_classes=num_classes,
                      width=384,
                      depth=12,
                      heads=6,
                      mlp_dim=1536,
                      **kwargs)
    return Transformer(**model_args)


def vit_base(num_leads, num_classes=None, seq_len=2250, patch_size=75, **kwargs):
    model_args = dict(seq_len=seq_len,
                      patch_size=patch_size,
                      num_leads=num_leads,
                      num_classes=num_classes,
                      width=768,
                      depth=12,
                      heads=12,
                      mlp_dim=3072,
                      **kwargs)
    return Transformer(**model_args)