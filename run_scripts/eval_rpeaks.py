
import argparse
from pathlib import Path

from torch import utils
import lightning as L
import torch
from torch.utils.data import DataLoader

from bench_xecg import dataset
import bench_xecg.dataset.mit_bih as mit_bih
import bench_xecg.dataset.intense_exercise as exercise
import bench_xecg.dataset.mimic_iv as mimic_iv
import bench_xecg.dataset.cpsc2018 as cpsc2018
from bench_xecg.utils.eval_utils import find_checkpoint
from bench_xecg.config import parse_config, set_num_classes_r_peaks

import bench_xecg.utils.utils as utils
from bench_xecg.dataset.generic_utils import get_transforms
from bench_xecg.trainers.r_peaks_trainer import TrainingRPeak

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file',     type=str, default='configs/train_mit_bih_defaults.yaml')
parser.add_argument('--config_file_default',     type=str, default='config_defaults/train_mit_bih_defaults.yaml')
parser.add_argument('--experiment_list', type=str, nargs='+', required=True, help='List of run IDs, e.g. xlstm_abc123 resnet_xyz456')
parser.add_argument('--ckpt_root',       type=str, default='train-age', help='Root folder that contains <run_id>/checkpoints/')
parser.add_argument('--results_root',    type=str, default='lightning_logs/train-age', help='Root folder for results CSVs')
parser.add_argument('--batch_size',      type=int, default=2048, help='Batch')
parser.add_argument('--dataset', type=str, default='mit-bih', help='Dataset to evaluate: mit_bih or exercise')

def build_test_dataloader(config, batch_size, dataset):
    transforms = get_transforms(config, split='test')
    if dataset == 'mit-bih':
        ds = mit_bih.ECGMITBIHDataset(config, split='test', augmentations=transforms)
    elif dataset == 'exercise':
        ds = exercise.ECGHighIntensity(config, split='test', global_augmentations=transforms)
    else:
        raise ValueError(f"Dataset {dataset} not supported, only 'mit_bih' or 'exercise' are valid")

    return DataLoader(ds, num_workers=config.num_workers, batch_size=batch_size, collate_fn=mit_bih.make_collate_fn(config))


def evaluate_run(run_id: str, config, ckpt_root: str, base_model, test_dataloader, dataset=''):
    ckpt_path = find_checkpoint(ckpt_root, run_id)
    print(f"\n{'='*60}")
    print(f"Run : {run_id}")
    print(f"Ckpt: {ckpt_path}")
    print(f"{'='*60}")

    print(config)
    
    model = TrainingRPeak(
        model=base_model,
        config=config,
        len_train_dataset=1,          # not used during test
    )

    trainer = utils.get_trainer(config, f'train-{dataset}-rpeak', wandb=False, version=run_id)
    trainer.test(model=model, dataloaders=test_dataloader, ckpt_path=ckpt_path)

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args   = parser.parse_args()
    config = parse_config(args.config_file, args.config_file_default)
    config.max_length_signal = config.win_len * 2 + config.context_len * 2

    config = set_num_classes_r_peaks(config)

    base_model = utils.get_base_model(config, feature_classification=(not config.single_hb))
    test_dataloader  = build_test_dataloader(config, args.batch_size, args.dataset)

    for run_id in args.experiment_list:
        evaluate_run(
            run_id          = run_id,
            config          = config,
            ckpt_root       = args.ckpt_root,
            base_model      = base_model,
            test_dataloader = test_dataloader,
            dataset         = args.dataset
        )
