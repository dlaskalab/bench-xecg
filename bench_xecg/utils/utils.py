import os
from collections import Counter
from joblib import Parallel, delayed

import torch
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torch.utils.data import Subset
import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, CSVLogger
import lightning as pl
import torch.nn as nn

from bench_xecg.models.classification import xLSTMClassification, xLSTMFeatureClassification, xLSTMSleepApnea
from ..models.ecg_jepa.models import load_encoder
import bench_xecg.models.st_mem.encoder as encoder
from ..models.ecg_founder.finetune_model import ft_1lead_ECGFounder, ft_12lead_ECGFounder
from ..models.cpc.model import CPCWrapper


def get_base_model(config, feature_classification=False, sleep_apnea=False, model_base_path = ''):
    """
    Returns the base model according to the configuration.
    
    Args:
        config (ConfigDict): Global configuration.
        feature_classification (bool): Whether to use feature classification of signal level classification
        minute_aggregation (bool): Whether the head should aggregate 1 minute of signal (this is used for the sleep apnea task)
    Returns:
        nn.Module: The base model.
    """
    if config.use_st_mem:
        base_model = getattr(encoder, 'st_mem_vit_base')(
            seq_len=2250, 
            patch_size=75, 
            num_leads=12, 
            num_classes=config.num_classes, 
            linear_probing=config.linear_probing, 
            drop_path_rate=config.drop_path_prob, 
            feature_classification=feature_classification, 
            r_peaks_detection=config.r_peaks_detection,
            sleep_apnea=sleep_apnea,
            window_size=config.window_size,
            context_size=config.context_size
        )
        checkpoint = torch.load(model_base_path + './pretrained_models/st_mem/st_mem_vit_base_encoder.pth', weights_only=False)
        checkpoint_model = checkpoint['model']
        state_dict = base_model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Remove key {k} from pre-trained checkpoint")
                del checkpoint_model[k]
        msg = base_model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    elif config.use_ecg_jepa:
        ckpt_dir = model_base_path + './pretrained_models/ecg_jepa/multiblock_epoch100.pth'
        base_model = load_encoder(
            ckpt_dir=ckpt_dir, 
            config=config, 
            feature_classification=feature_classification, 
            sleep_apnea=sleep_apnea
        ) # dim is the dimension of the latent space
    elif config.use_ecg_founder:
        if len(config.leads) == 1:
            path = model_base_path + './pretrained_models/ecg_founder/1_lead_ECGFounder.pth'
            base_model = ft_1lead_ECGFounder('cuda', path, config.num_classes, linear_prob=config.linear_probing)
        else:
            path =  model_base_path + './pretrained_models/ecg_founder/12_lead_ECGFounder.pth'
            base_model = ft_12lead_ECGFounder('cuda', path, config.num_classes, linear_prob=config.linear_probing)
    elif config.use_ecg_cpc:
        base_model = CPCWrapper(config, model_base_path+ './pretrained_models/ecg_cpc/init_dict.yaml', feature_classification=feature_classification, sleep_apnea=sleep_apnea)
        base_model.load_weights_from_checkpoint('./pretrained_models/ecg_cpc/last_11597276_state_dict.ckpt')
    else:
        if sleep_apnea:
            base_model = xLSTMSleepApnea(config=config, num_classes=config.num_classes, num_channels=len(config.leads))
        elif feature_classification:
            base_model = xLSTMFeatureClassification(config=config, num_classes=config.num_classes, num_channels=len(config.leads))
        else:
            base_model = xLSTMClassification(config=config, num_classes=config.num_classes, num_channels=len(config.leads))
        
        if config.checkpoint is not None and config.checkpoint != '':   
            checkpoint = torch.load(config.checkpoint, weights_only=False)
            new_state_dict = {format_keys(k): v for k, v in checkpoint['state_dict'].items()}

            if config.backend == 'vanilla':
                for k, v in new_state_dict.items():
                    if "slstm_cell._recurrent_kernel_" in k:
                        new_state_dict[k] = v.permute(0, 2, 1)

            # remove the fc layer
            new_state_dict = {k: v for k, v in new_state_dict.items() if 'fc' not in k and 'head' not in k}
            message = base_model.load_state_dict(new_state_dict, strict=False) 
            print(message) 

    # this gives problem due to reshaping
    if config.compile_model:
        base_model.compile()

    return base_model

def change_positional_embedding_if_needed(model, config):
    """
    Change the positional embedding of the model to match the new sequence length.
    This is done by repeating the last positional embedding.

    Args:
        model (nn.Module): The model with the positional embedding to change.
        config (ConfigDict): Global configuration.
    Returns:
        nn.Module: The model with the changed positional embedding.
    """

    
    win_seq_len = (config.window_size * config.sampling_freq // config.patch_size)
    new_seq_len = (config.context_size * config.sampling_freq // config.patch_size) // 2

    print(f"Changing positional embedding to new sequence length {new_seq_len}")
    if config.use_st_mem:
        original_seq_len = model.pos_embedding.shape[1] - 2  # exclude cls token
        print(f'Original positional embedding shape: {model.pos_embedding.shape}')

        if new_seq_len * 2 + win_seq_len + 2 <= model.pos_embedding.shape[1]:
            print(f'No need to change positional embedding, current max sequence length is {model.core.pos_embedding.shape[1]-1}')
            return model
        
        # get all the positional embeddings except the last one (for cls token)
        left_pe = model.pos_embedding[:, :1, :] # position 0
        right_pe = model.pos_embedding[:, -1:, :] # last position

        # take the last -1 positional embedding and repeat it (the last is for cls tokens)
        repeated_pe_left = model.pos_embedding[:, :1, :].repeat(1, new_seq_len + (win_seq_len - original_seq_len) // 2, 1)
        print(f'repeated left pe shape: {repeated_pe_left.shape}')
        repeated_pe_right = model.pos_embedding[:, -1:, :].repeat(1, new_seq_len + (win_seq_len - original_seq_len) // 2, 1)
        print(f'repeated right pe shape: {repeated_pe_right.shape}')

        model.pos_embedding = nn.Parameter(torch.cat([
            left_pe, 
            repeated_pe_left, 
            model.pos_embedding[:, 1:-1, :], 
            repeated_pe_right, 
            right_pe
        ], dim=1))

        print(f'New positional embedding shape: {model.pos_embedding.shape}')

    elif config.use_ecg_founder:
        pass
    elif config.use_ecg_jepa:
        pass
    elif config.encoder_type == 'transformer':
        original_seq_len = model.core.pos_embedding.shape[1] - 2  # exclude cls token
        print(f'Original positional embedding shape: {model.core.pos_embedding.shape}')

        if new_seq_len * 2 + win_seq_len + 2 <= model.core.pos_embedding.shape[1]:
            print(f'No need to change positional embedding, current max sequence length is {model.core.pos_embedding.shape[1]-1}')
            return model
        
        # get all the positional embeddings except the last one (for cls token)
        left_pe = model.core.pos_embedding[:, :1, :] # position 0
        right_pe = model.core.pos_embedding[:, -1:, :] # last position

        # take the last -1 positional embedding and repeat it (the last is for cls tokens)
        repeated_pe_left = model.core.pos_embedding[:, :1, :].repeat(1, new_seq_len + (win_seq_len - original_seq_len) // 2, 1)
        print(f'repeated left pe shape: {repeated_pe_left.shape}')
        repeated_pe_right = model.core.pos_embedding[:, -1:, :].repeat(1, new_seq_len + (win_seq_len - original_seq_len) // 2, 1)
        print(f'repeated right pe shape: {repeated_pe_right.shape}')

        model.core.pos_embedding = nn.Parameter(torch.cat([
            left_pe, 
            repeated_pe_left, 
            model.core.pos_embedding[:, 1:-1, :], 
            repeated_pe_right, 
            right_pe
        ], dim=1))

    return model

def split_dataset_preserve_labels(dataset, split_ratio=0.1, key='class_label'):
    """
    Splits the dataset into a training set while preserving the label distribution using multilabel stratified shuffle split.

    Args:
        dataset (Dataset): The dataset to split.
        split_ratio (float): The ratio of the dataset to use for training.
        key (str): The key in the dataset samples that contains the multilabels.
    Returns:
        Subset: A subset of the original dataset containing the training samples.
    """
    print(f"Splitting dataset with {split_ratio} training data")
    multilabels = np.array([dataset[i][key] for i in range(len(dataset))])  # get multilabels
    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1 - split_ratio)
    train_idx, _ = next(splitter.split(np.zeros(len(multilabels)), multilabels))
    balanced_train_dataset = Subset(dataset, train_idx)
    print(f"Train dataset size: {len(balanced_train_dataset)}")
    return balanced_train_dataset

def format_keys(key):
    """ 
    Because of some model re-naming after saving weights this function is needed to load weights correctly.
    """
    if key.startswith('model.'):
        key = key[6:]

    key = key.replace('xlstm.model', 'core.model')  # Remove 'module.' prefix if present
        
    return key

def get_trainer(config, prj_string, wandb=False, run=None, version=None):
    """
    Define all the callbacks and loggers for the trainer.
    """
    callbacks = []
    if config.monitor_metric is not None:
        early_stopping = EarlyStopping(monitor=config.monitor_metric, check_finite=True, patience=config.patience, mode=config.monitor_mode)
        callbacks.append(early_stopping)

        if config.monitor_metric != 'val_loss':
            nan_stop = EarlyStopping(monitor='val_loss', check_finite=True, patience=config.epochs, mode='min')
            callbacks.append(nan_stop)

    csv_logger = CSVLogger(save_dir=f'logs_lightning/{prj_string}', name=config.wandb_group, version=version)

    if wandb:
        print(f"Using WandbLogger for project {prj_string} and run {run}")
        if config.monitor_metric is not None:
            checkpoint_callback = ModelCheckpoint(monitor=config.monitor_metric, mode=config.monitor_mode)
            callbacks.append(checkpoint_callback)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)

        gpu_tag = [f'CUDA_VISIBLE_DEVICES_{os.environ["CUDA_VISIBLE_DEVICES"]}'] if 'CUDA_VISIBLE_DEVICES' in os.environ else [f'CUDA_VISIBLE_DEVICES_{torch.cuda.current_device()}']
        print(f"Using GPU tag: {gpu_tag}")
        
        # Adding the tag to the logger
        wandb_logger = WandbLogger(project=prj_string, experiment=run, config=config, group=config.wandb_group, tags=gpu_tag)
        #  wand_logger.watch(model, log=None)
        trainer = pl.Trainer(max_epochs=config.epochs, logger=[wandb_logger, csv_logger], callbacks=callbacks, gradient_clip_val=config.grad_clip, precision=config.precision)
        # need to save the config file to a new file in the wandb directory
    else:
        print(f"Using default logger for project {prj_string} and run {run}")
        trainer = pl.Trainer(logger=csv_logger, max_epochs=config.epochs, callbacks=callbacks, gradient_clip_val=config.grad_clip, precision=config.precision)
    return trainer

def get_training_class_weights(train_dataset, do_not_consider_classes=[], label_key='label', n_jobs=-1):
    """
    Returns the class weights for the training dataset.
    Parallelized using joblib for faster processing.
    
    Args:
        train_dataset: Dataset to extract labels from
        do_not_consider_classes: Classes to exclude from weight calculation
        label_key: Key to access labels in dataset
        n_jobs: Number of parallel jobs (-1 uses all cores)
    """
    
    def extract_label(i):
        """Extract and process a single label."""
        return train_dataset[i][label_key]
    
    # Parallel label extraction
    labels = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(extract_label)(i) for i in range(len(train_dataset))
    )
    
    # Flatten multilabel if needed
    if len(labels[0]) > 1:
        from itertools import chain
        labels = list(chain.from_iterable(labels))
    
    # Filter and convert to items
    labels = [label.item() for label in labels if label not in do_not_consider_classes]
    
    # Calculate class weights
    class_counts = Counter(labels)
    print('Class count:', class_counts)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    class_weights = {cls: total_samples / (num_classes * count) for cls, count in class_counts.items()}
    
    weights = torch.tensor([class_weights[cls] for cls in range(num_classes)], dtype=torch.float32)
    print(f"Class Weights: {weights}")
    return weights

def get_training_class_weights_multilabel(train_dataset, label_key='label'):
    labels = [sample[label_key] for sample in train_dataset]
    classes_count = torch.zeros((len(labels[0]),))

    for label in labels:
        classes_count += torch.tensor(label)

    num_classes = classes_count.shape[0]
    total_samples = len(labels)

    class_weights =  total_samples / (classes_count * num_classes)
    print('Class weights: ', class_weights)
    return class_weights

