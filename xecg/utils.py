import torch
import numpy as np
import random
from torch import nn
from xlstm import FeedForwardConfig, mLSTMLayerConfig, mLSTMBlockConfig, sLSTMLayerConfig, sLSTMBlockConfig, xLSTMBlockStackConfig, xLSTMBlockStack
from xlstm.xlstm_large import xLSTMLargeConfig
from xlstm.xlstm_large.model import xLSTMLargeBlockStack
from bench_xecg.models.modules import *
import os
import bench_xecg.models.transformer.encoder as encoder


def get_normalization_layer(config, embedding_size=None):
    if config.cls_normalization == 'layer':
        return nn.LayerNorm(embedding_size)
    elif config.cls_normalization == 'batch':
        return nn.BatchNorm1d(embedding_size)
    elif config.cls_normalization == 'instance':
        return nn.InstanceNorm1d(embedding_size)
    else:
        return nn.Identity()
    

def get_patch_embedding(type, patch_size, num_hiddens, num_channels):
    if type == 'linear':
        print('using linear patch embedding')
        return LinearPatchEmbedding(patch_size=patch_size, num_hiddens=num_hiddens, num_channels=num_channels)
    if type == 'non_linear':
        print('using non-linear patch embedding')
        return NonLinearPatchEmbedding(patch_size=patch_size, num_hiddens=num_hiddens, num_channels=num_channels)
    if type == 'conv':
        print('using conv patch embedding')
        return ConvPatchEmbedding(patch_size=patch_size, num_hiddens=num_hiddens, num_channels=num_channels)
    if type == 'enriched':
        print('using enriched patch embedding')
        return EnrichedLinearPatchEmbedding(patch_size=patch_size, num_hiddens=num_hiddens, num_channels=num_channels, enrich_dim=num_hiddens //4, kernel_size=5)
    if type == 'sparse_conv':
        print('using sparse patch embedding')
        return SparseConvPatchEmbedding(patch_size=patch_size, num_hiddens=num_hiddens, num_channels=num_channels, out_channels=num_hiddens // 4)
    else:
        raise ValueError(f"Patch embedding {type} not supported")

def get_transformer(config):
    return encoder.__dict__['Transformer'](seq_len=1200, patch_size=config.patch_size, num_leads=12, drop_path_rate=config.drop_path_prob)

def get_reconstruction_head(patch_size, embedding_size, num_channels):
    return EmbedPatching(
        patch_size=patch_size, 
        num_hiddens=embedding_size, 
        num_channels=num_channels, 
        use_pre_head=True
    )


def get_xlstm(config):
    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4, 
                qkv_proj_blocksize=config.num_heads, 
                num_heads=config.num_heads,
                proj_factor=config.proj_factor
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                num_heads=config.num_heads,
                backend=config.backend if config.backend else "cuda",
                conv1d_kernel_size=4,
                bias_init="powerlaw_blockdependent",
                batch_size=config.batch_size,
            ),
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn=config.activation_fn),
        ),
        context_length=8000,
        num_blocks=len(config.xlstm_config),
        embedding_dim=config.embedding_size,
        slstm_at=[idx for idx, b in enumerate(config.xlstm_config) if b == 's'],
        dropout=config.dropout,

        add_post_blocks_norm=config.use_final_layer_norm
    )
    print('creating xlstm with slstm at: ', [idx for idx, b in enumerate(config.xlstm_config) if b == 's'])
    blocks = xLSTMBlockStack(cfg)
    return vanillaxLSTMWrapper(blocks, dropout=config.dropout, bidirectional=config.bidirectional, drop_path=config.drop_path_prob)

def get_large_xlstm(     
        config,
    ):
    xlstm_config = xLSTMLargeConfig(
        embedding_dim=config.embedding_size,
        num_heads=config.num_heads,
        num_blocks=len(config.xlstm_config),
        vocab_size=0,
        return_last_states=True,
        mode="train",
        chunkwise_kernel="chunkwise--triton_xl_chunk", # xl_chunk == TFLA kernels
        sequence_kernel="native_sequence__triton",
        step_kernel="triton",
        add_out_norm=config.use_final_layer_norm
    )

    blocks = xLSTMLargeBlockStack(xlstm_config)
    return mLSTMWrapper(blocks, dropout=config.dropout, bidirectional=config.bidirectional, drop_path=config.drop_path_prob)
