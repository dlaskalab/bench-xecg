import argparse

from torch import utils
import lightning as pl
import torch
from torch.utils.data import DataLoader

from bench_xecg.trainers.r_peaks_trainer import TrainingRPeak
import bench_xecg.dataset.intense_exercise as intense_exercise
import bench_xecg.utils.utils as utils
from bench_xecg.dataset.generic_utils import get_transforms
from bench_xecg.config import parse_config, set_num_classes_r_peaks

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file', type=str, default='configs/train_high_intensity_run_config.yaml', help='Path to the config file')

def train(config, run=None, wandb=False):
    # set deterministic training
    if config.deterministic: pl.seed_everything(42)

    config = set_num_classes_r_peaks(config)


    if config.use_class_weights:
        param = config.sampling_freq if config.patch_size < 5 else config.patch_size
        weights = torch.tensor([1/param, (param-1)/param]).to('cuda')
        print(f'Using class weights for r-peaks detection: {weights}')
    else:
        weights = None


    train_dataset =  intense_exercise.ECGHighIntensity(config, split='train', global_augmentations=get_transforms(config))
    print(f"Train dataset size: {len(train_dataset)}")
    val_dataset = intense_exercise.ECGHighIntensity(config, split='val', global_augmentations=get_transforms(config, split='val'))
    print(f"Val dataset size: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    test_dataset = intense_exercise.ECGHighIntensity(config, split='test', global_augmentations=get_transforms(config, split='test'))
    print(f"Test dataset size: {len(test_dataset)}")

    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    base_model = utils.get_base_model(config, feature_classification=True)

    model = TrainingRPeak(model=base_model, config=config, len_train_dataset=len(train_dataset), weights=weights)

    trainer = utils.get_trainer(config, "train-exercise-r-peak", wandb=wandb, run=run)
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model=model, dataloaders=test_dataloader, ckpt_path='best')

# if main
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()
    config = parse_config(args.config_file, 'config_defaults/train_high_intensity_defaults.yaml')

    train(config, wandb=config.wandb_log)