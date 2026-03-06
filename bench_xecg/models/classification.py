import torch
import torch.nn as nn

from .xlstm_model import pretrainedxLSTM
from .utils import get_normalization_layer

class xLSTMClassification(pretrainedxLSTM):
    def __init__(
            self, 
            config,
            num_classes,
            num_channels,
        ): 
        self.linear_probing = config.linear_probing
        super(xLSTMClassification, self).__init__(num_channels, config, reconstruction=False)

        self.head = nn.Sequential(
            get_normalization_layer(config, config.embedding_size),
            nn.Linear(config.embedding_size, num_classes)
        )

    def forward(self, x, age=None, gender=None):
        # padding_mask = self.get_padding_mask(x)

        if self.linear_probing:
            with torch.no_grad():
                x, _ = self.embed_and_mask_signal_if_needed(x, masking=False, age=age, gender=gender)
                cls, _ = self.forward_core(x) #, padding_mask)
        else:  
            x, _ = self.embed_and_mask_signal_if_needed(x, masking=False, age=age, gender=gender)
            cls, _ = self.forward_core(x) #, padding_mask)

        res = self.head(cls)
        return res
    
class xLSTMSleepApnea(pretrainedxLSTM):
    def __init__(
            self, 
            config,
            num_classes,
            num_channels,
        ): 
        self.linear_probing = config.linear_probing
        self.context_size = config.context_size
        self.window_size = config.window_size

        super(xLSTMSleepApnea, self).__init__(num_channels, config, reconstruction=False)

        self.head = nn.Sequential(
            get_normalization_layer(config, config.embedding_size),
            nn.Linear(config.embedding_size, num_classes)
        )

    def forward(self, x, age=None, gender=None):
        if self.linear_probing:
            with torch.no_grad():
                x, _ = self.embed_and_mask_signal_if_needed(x, masking=False, age=age, gender=gender)
                _, features = self.forward_core(x)
        else:  
            x, _ = self.embed_and_mask_signal_if_needed(x, masking=False, age=age, gender=gender)
            _, features = self.forward_core(x)

        # remove the context patches from the features
        if self.context_size > 0:
            context_patches = (self.context_size * self.sampling_freq) // self.patch_size // 2
            window_patches = (self.window_size * self.sampling_freq) // self.patch_size
            start = context_patches
            end = context_patches + window_patches
            # print(f"Features shape before removing context patches: {features.shape}")
            # print(f"Removing context patches: start {start}, end {end}")
            features = features[:, start:end, :]
            # print(f"Features shape after removing context patches: {features.shape}")

        out, _ = self.pooling(features)
        res = self.head(out)

        return res

class xLSTMFeatureClassification(pretrainedxLSTM):
    def __init__(
            self, 
            config,
            num_classes,
            num_channels,
        ): 
        self.linear_probing = config.linear_probing
        super(xLSTMFeatureClassification, self).__init__(num_channels, config, reconstruction=False)

        self.head = nn.Sequential(
            get_normalization_layer(config, config.embedding_size),
            nn.Linear(config.embedding_size, num_classes)
        )

    def forward(self, x, age=None, gender=None):
        if self.linear_probing:
            with torch.no_grad():
                x, _ = self.embed_and_mask_signal_if_needed(x, masking=False, age=age, gender=gender)
                _, features = self.forward_core(x)
        else:  
            x, _ = self.embed_and_mask_signal_if_needed(x, masking=False, age=age, gender=gender)
            _, features = self.forward_core(x)

        _, features = self.pooling(features)

        res = self.head(features)
        return res

