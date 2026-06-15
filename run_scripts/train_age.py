from torch import utils
import lightning as L
import torch
from torch.utils.data import DataLoader


import bench_xecg.dataset.code_dataset as code
import bench_xecg.dataset.ptb_xl as ptbxl
import bench_xecg.dataset.mimic_iv as mimic_iv
import bench_xecg.dataset.cpsc2018 as cpsc2018

import bench_xecg.utils.utils as utils
from bench_xecg.dataset.generic_utils import get_transforms, make_collate_fn_task
from bench_xecg.trainers.regression_trainer import RegressionTrainer
from bench_xecg.config import parse_config

import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file', type=str, default='configs/train_age_run_config.yaml', help='Path to the config file')
parser.add_argument('--version', type=str, default=None)

def train(config, run=None, wandb=False, version=None):
    # set deterministic training
    if config.deterministic: L.seed_everything(42)
    
    code_dataset = code.ECGCODE15AgeDataset(config, split='train', global_augmentations=get_transforms(config))

    # split the dataset in val and train
    pct = 0.8
    train_dataset, val_dataset = torch.utils.data.random_split(code_dataset, [int(len(code_dataset) * pct), len(code_dataset) - int(len(code_dataset) * pct)])

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=make_collate_fn_task(config, key_label='age'))
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=make_collate_fn_task(config, key_label='age'))

    test_ptbxl = ptbxl.ECGPTBXLAgeDataset(config, split='all', global_augmentations=get_transforms(config, split='test'))
    test_mimic = mimic_iv.ECGMIMICDataset(config, split='all', global_augmentations=get_transforms(config, split='test'), downstream_task='age')
    test_cpsc = cpsc2018.ECGCPSC2018AgeDataset(config, split='all', global_augmentations=get_transforms(config, split='test'))
    test_ptbxl = DataLoader(test_ptbxl, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=make_collate_fn_task(config, key_label='age'))
    test_mimic = DataLoader(test_mimic, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=make_collate_fn_task(config, key_label='age'))
    test_cpsc = DataLoader(test_cpsc, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=make_collate_fn_task(config, key_label='age'))

    base_model = utils.get_base_model(config)

    log_every_n_steps = max(1, len(train_dataset) // (config.batch_size * 10))
    print(f"Logging every {log_every_n_steps} steps")

    map_idx_dataloader = {0: 'ptbxl', 1: 'mimic', 2: 'cpsc'}
    model = RegressionTrainer(model=base_model, config=config, len_train_dataset=len(train_dataset), map_idx_dataloader=map_idx_dataloader)

    trainer = utils.get_trainer(config, 'train-age', wandb=wandb, run=run, version=f'version_{version}' if version is not None else None)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=[test_ptbxl, test_mimic, test_cpsc], ckpt_path='best')

# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = parse_config(args.config_file, 'config_defaults/train_age_defaults.yaml')

    train(config, wandb=config.wandb_log, version=args.version)