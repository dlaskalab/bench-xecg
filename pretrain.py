import os

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
import torch
from torch.utils.data import DataLoader, Subset

from bench_xecg.dataset.dataset_preparation_utils import load_datasets
from bench_xecg.trainers.common import DelayedCheckpoint
import bench_xecg.dataset.ptb_xl as ptb_xl
from bench_xecg.dataset import mit_bih
from bench_xecg.models.xlstm_model import pretrainedxLSTM
import bench_xecg.dataset.generic_utils as generic_utils
from bench_xecg.trainers.ssl_pretrainer import PretrainedNetwork
from bench_xecg.config import parse_config


# argparse
import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file', type=str, default='configs/pretrain_run_config.yaml', help='Path to the config file')

def pretrain(config, run=None, wandb=False):
    max_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", config.num_workers))
    config.num_workers = min(config.num_workers, max_cpus)

    # set deterministic training
    if config.deterministic: L.seed_everything(42)

    train_dataset,val_dataset = load_datasets(config)

    # downstream ptb_xl dataset
    ptb_xl_train_dataset = ptb_xl.ECGPTBXLDataset(config, split='train', global_augmentations=None, local_augmentations=None)
    ptb_xl_val_dataset = ptb_xl.ECGPTBXLDataset(config, split='val', global_augmentations=None, local_augmentations=None)
    ptb_xl_train_dataloader = DataLoader(ptb_xl_train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=ptb_xl.make_collate_fn(config))
    ptb_xl_val_dataloader = DataLoader(ptb_xl_val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=ptb_xl.make_collate_fn(config))

    # downstream mit-bih dataset
    mit_bih_train_dataset = mit_bih.ECGMITBIHDataset(config, split='train', augmentations=None)
    mit_bih_val_dataset = mit_bih.ECGMITBIHDataset(config, split='val', augmentations=None)
    mit_bih_train_dataloader = DataLoader(mit_bih_train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=mit_bih.make_collate_fn(config))
    mit_bih_val_dataloader = DataLoader(mit_bih_val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=mit_bih.make_collate_fn(config))

    # keep only 10% of the dataset
    if config.debug: 
        train_dataset = Subset(train_dataset, range(0, len(train_dataset) // 100))
        config.xlstm_config = ['m', 'm', 'm']
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=generic_utils.make_collate_fn(config))
    len_train_dataset = len(train_dataset)

    # cat the two dataloaders
    # if config.debug: val_dataset = Subset(val_dataset, range(0, len(val_dataset) // 10))
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=generic_utils.make_collate_fn(config))
    base_model = pretrainedxLSTM(config=config, num_channels=len(config.leads))
    # base_model.compile()

    if config.checkpoint != None:
        model = PretrainedNetwork.load_from_checkpoint(
            checkpoint_path=config.checkpoint,
            model=base_model, 
            len_train_dataset=len_train_dataset, 
            config=config, 
            ptb_xl_train_dataloader=ptb_xl_train_dataloader, 
            ptb_xl_val_dataloader=ptb_xl_val_dataloader,
            mit_bih_train_dataloader=mit_bih_train_dataloader,
            mit_bih_val_dataloader=mit_bih_val_dataloader,
        )
    else:
        model = PretrainedNetwork(
            model=base_model, 
            len_train_dataset=len_train_dataset,
            config=config, 
            ptb_xl_train_dataloader=ptb_xl_train_dataloader, 
            ptb_xl_val_dataloader=ptb_xl_val_dataloader,
            mit_bih_train_dataloader=mit_bih_train_dataloader,
            mit_bih_val_dataloader=mit_bih_val_dataloader,
        )
        
    checkpoint_callback = DelayedCheckpoint(delay_epochs=config.monitor_delay_epochs, monitor=config.monitor_metric, mode=config.monitor_mode)
    early_stopping = EarlyStopping(monitor=config.monitor_metric, patience=config.patience, mode=config.monitor_mode)
    num_gpus = os.environ.get('NUM_GPUS', 1)
    print(f"Number of GPUs: {num_gpus}")

    if wandb:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        wand_logger = WandbLogger(project="pretrain-xLSTM", experiment=run, config=config)
        wand_logger.watch(model, log='gradients')
        trainer = L.Trainer(
            # num_sanity_val_steps=0,
            max_epochs=config.epochs, 
            logger=wand_logger, 
            callbacks=[checkpoint_callback, early_stopping, lr_monitor], 
            # gradient_clip_val=config.grad_clip,
            accelerator='gpu',
            devices=num_gpus,
            strategy='ddp_find_unused_parameters_true' if num_gpus > 1 else 'auto', 
            sync_batchnorm=True if num_gpus > 1 else False,
            precision=config.precision
        )
    else:
        trainer = L.Trainer(
            # num_sanity_val_steps=0,
            logger=False,
            max_epochs=config.epochs, 
            callbacks=[checkpoint_callback, early_stopping], 
            gradient_clip_val=config.grad_clip,
            accelerator='gpu',
            devices=num_gpus,
            strategy='ddp_find_unused_parameters_true' if num_gpus > 1 else 'auto',
            sync_batchnorm=True if num_gpus > 1 else False,
            precision=config.precision
        )

    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    #test_dataset = mit_bih.ECGMITBIHDataset(config, split='test')
    #test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=generic_utils.collate_fn, num_workers=config.num_workers)
    #trainer.test(model=model, dataloaders=test_dataloader)


# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = parse_config(args.config_file, 'config_defaults/pretrain_config_defaults.yaml')
    pretrain(config, wandb=config.wandb_log)
