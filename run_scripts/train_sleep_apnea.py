import argparse
import random

import lightning as L
import torch
from torch.utils.data import DataLoader

import bench_xecg.utils.utils as utils
from bench_xecg.dataset.generic_utils import get_transforms
from bench_xecg.trainers.sleep_apnea_trainer import TrainingSleepApnea
from bench_xecg.config import parse_config

import bench_xecg.dataset.sleep_apnea as sleep_apnea

def check_window_size(config):
    assert config.window_size > 0, "Window size must be greater than 0"
    if config.window_size < 60:
        assert 60 % config.window_size == 0, "For window sizes less than 60, ensure that 60 is divisible by the window size."
    if config.window_size > 60:
        raise NotImplementedError("Window sizes greater than 60 seconds are not supported.")
    #else: assert config.window_size % 60 == 0, "For window sizes greater than or equal to 60, ensure that the window size is divisible by 60."
    
    if config.context_size > 0:
        assert config.context_size % 120 == 0, "Context size must be a multiple of 60."

import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file', type=str, default='configs/train_sleep_apnea_run_config.yaml', help='Path to the config file')

def train(config, run=None, wandb=False):
    # set deterministic training
    if config.deterministic: L.seed_everything(42)
    check_window_size(config)

    split_seed = 42 if config.deterministic else random.randrange(2**32)
    train_dataset = sleep_apnea.ECGSleepApneaDataset(config, split='train', augmentations=get_transforms(config), split_seed=split_seed)
    print(f"Train dataset size: {len(train_dataset)}")
    val_dataset = sleep_apnea.ECGSleepApneaDataset(config, split='val', augmentations=get_transforms(config, split='val'), split_seed=split_seed)
    print(f"Val dataset size: {len(val_dataset)}")

    if config.use_class_weights:
        weights = utils.get_training_class_weights(train_dataset, label_key='annotation').to('cuda')
        print(f"Using class weights: {weights}")
    else:
        weights = None

    config.max_length_signal = (config.window_size + config.context_size) * config.sampling_freq
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=sleep_apnea.make_collate_fn(config, split='train'), drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False, collate_fn=sleep_apnea.make_collate_fn(config, split='val'))

    test_dataset = sleep_apnea.ECGSleepApneaDataset(config, split='test', augmentations=get_transforms(config, split='test'))
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False, collate_fn=sleep_apnea.make_collate_fn(config, split='test'))

    # feature classification only if the signal is 1 minute long
    feature_classification = config.window_size % 60 == 0
    base_model = utils.get_base_model(config, feature_classification=feature_classification, sleep_apnea=True)
    base_model = utils.change_positional_embedding_if_needed(base_model, config)
            
    model = TrainingSleepApnea(model=base_model, config=config, len_train_dataset=len(train_dataset), weights=weights)

    trainer = utils.get_trainer(config, 'train-sleep-apnea_bis', wandb=wandb, run=run)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader, ckpt_path='best')

# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = parse_config(args.config_file, 'config_defaults/train_sleep_apnea_defaults.yaml')

    train(config, wandb=config.wandb_log)