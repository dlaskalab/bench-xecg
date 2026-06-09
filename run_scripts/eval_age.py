import os
import glob
import argparse
from pathlib import Path

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

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file',     type=str, default='configs/train_age_run_config.yaml')
parser.add_argument('--experiment_list', type=str, nargs='+', required=True, help='List of run IDs, e.g. xlstm_abc123 resnet_xyz456')
parser.add_argument('--ckpt_root',       type=str, default='train-age', help='Root folder that contains <run_id>/checkpoints/')
parser.add_argument('--results_root',    type=str, default='lightning_logs/train-age', help='Root folder for results CSVs')
parser.add_argument('--batch_size',      type=int, default=2048, help='Batch')

def find_checkpoint(ckpt_root: str, run_id: str) -> str:
    """Return the single .ckpt file inside train-age/<run_id>/checkpoints/."""
    pattern = os.path.join(ckpt_root, run_id, 'checkpoints', '*.ckpt')
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No checkpoint found for run '{run_id}' (pattern: {pattern})")
    if len(matches) > 1:
        # Prefer 'best' in the name, otherwise take the last modified file
        best = [m for m in matches if 'best' in os.path.basename(m).lower()]
        matches = best if best else sorted(matches, key=os.path.getmtime)
    return matches[0]

def build_test_dataloaders(config, batch_size):
    """Build the three test dataloaders (PTB-XL, MIMIC-IV, CPSC-2018)."""
    transforms = get_transforms(config, split='test')
    collate_fn = make_collate_fn_task(config, key_label='age')
    loader_kw  = dict(batch_size=batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=collate_fn)

    dl_ptbxl = DataLoader(ptbxl.ECGPTBXLAgeDataset(config, split='all', global_augmentations=transforms), **loader_kw)
    dl_mimic = DataLoader(mimic_iv.ECGMIMICDataset(config, split='all', global_augmentations=transforms, downstream_task='age'), **loader_kw)
    dl_cpsc  = DataLoader(cpsc2018.ECGCPSC2018AgeDataset(config, split='all', global_augmentations=transforms), **loader_kw)

    return [dl_ptbxl, dl_mimic, dl_cpsc]

def evaluate_run(run_id: str, config, ckpt_root: str, results_root: str, base_model, test_dataloaders):
    ckpt_path = find_checkpoint(ckpt_root, run_id)
    print(f"\n{'='*60}")
    print(f"Run : {run_id}")
    print(f"Ckpt: {ckpt_path}")
    print(f"{'='*60}")

    # Derive a human-readable model name from the run_id
    # Convention: <model_name>_<hash>  →  take everything before the last '_'

    base_path = Path(results_root) / run_id 
    results_csv = base_path / 'results.csv'
    
    map_idx_dataloader = {0: 'ptbxl', 1: 'mimic', 2: 'cpsc'}

    model = RegressionTrainer(
        model=base_model,
        config=config,
        len_train_dataset=1,          # not used during test
        map_idx_dataloader=map_idx_dataloader,
        save_results_path=results_csv,
    )

    trainer = utils.get_trainer(config, 'train-age', wandb=False, version=run_id)
    trainer.test(model=model, dataloaders=test_dataloaders, ckpt_path=ckpt_path)

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args   = parser.parse_args()
    config = parse_config(args.config_file, 'config_defaults/train_age_defaults.yaml')

    base_model = utils.get_base_model(config)
    test_dataloaders   = build_test_dataloaders(config, args.batch_size)

    for run_id in args.experiment_list:
        evaluate_run(
            run_id           = run_id,
            config           = config,
            ckpt_root        = args.ckpt_root,
            results_root     = args.results_root,
            base_model       = base_model,
            test_dataloaders = test_dataloaders 
        )
