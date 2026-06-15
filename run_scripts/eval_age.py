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
from bench_xecg.utils.eval_utils import find_checkpoint

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

# sbatch run_scripts/slurm/slurm_train_l40s.sh run_scripts/eval_age.py configs/age/ecgcpc_lp.yaml --experiment_list g73u9qjj psri40oa spkmfkmd 9z8ppkas 9z8ppkas --results_root logs_lightning/train-age/best_ecgcpc_lp
# sbatch run_scripts/slurm/slurm_train_l40s.sh run_scripts/eval_age.py configs/age/ecgcpc_ft.yaml --experiment_list hjov3u3c bdub4z5r 9xqav20g 8f6z37wi 3zyw8rfw --results_root logs_lightning/train-age/best_ecgcpc_ft

# sbatch run_scripts/slurm/slurm_train_l40s.sh run_scripts/eval_age.py configs/age/jepa_ft.yaml --experiment_list ftjefsce 35b1y7iv bqb7i00k pqz41faw 9faxr5vr --results_root logs_lightning/train-age/best_jepa_ft
# sbatch run_scripts/slurm/slurm_train_l40s.sh run_scripts/eval_age.py configs/age/jepa_lp.yaml --experiment_list 1kjf61j1 59stpj3q 61rrpmcu c7z7dkzk 200fcph4 --results_root logs_lightning/train-age/best_jepa_lp

# sbatch run_scripts/slurm/slurm_train_l40s.sh run_scripts/eval_age.py configs/age/stmem_ft.yaml --experiment_list dbuckbca z4139x28 8v6eptg5 wek14ycs 1vp2svhh --results_root logs_lightning/train-age/best_stmem_ft
# sbatch run_scripts/slurm/slurm_train_l40s.sh run_scripts/eval_age.py configs/age/stmem_lp.yaml --experiment_list pphn4it1 588v7224 gxe6888w fbn70rw0 ju17vfwf --results_root logs_lightning/train-age/best_stmem_lp

# sbatch run_scripts/slurm/slurm_train_l40s.sh run_scripts/eval_age.py configs/age/ecgfm_ft.yaml --experiment_list uro1ka9x 5z9p163l 2wz4yceh 50vcr89u r8gb68f4 --results_root logs_lightning/train-age/best_ecgfm_ft --batch_size 4096
# sbatch run_scripts/slurm/slurm_train_l40s.sh run_scripts/eval_age.py configs/age/ecgfm_lp.yaml --experiment_list snx6tkl1 pr59rc0n zk27ixgm cpusr98n 81c8ye71 --results_root logs_lightning/train-age/best_ecgfm_lp --batch_size 4096

# sbatch run_scripts/slurm/slurm_train_l40s.sh run_scripts/eval_age.py configs/age/transformer_lp.yaml --experiment_list 2nhlmg7t cbd43ystv 43pp0je3 law2qz6u gp3jjdi0 --results_root logs_lightning/train-age/best_trans_lp
# sbatch run_scripts/slurm/slurm_train_l40s.sh run_scripts/eval_age.py configs/age/transformer_ft.yaml --experiment_list 1q88hrzo qj4mtqzb fm90ie10 iq56dbbv orygw9th --results_root logs_lightning/train-age/best_trans_ft

# sbatch run_scripts/slurm/slurm_train_l40s.sh run_scripts/eval_age.py configs/age/xlstm_ft.yaml --experiment_list pxb0m582 pnghetc3 6ubu3eff 0u3msa1q e4vtuxiz --results_root logs_lightning/train-age/best_xlstm_ft --batch_size 8192
# sbatch run_scripts/slurm/slurm_train_l40s.sh run_scripts/eval_age.py configs/age/xlstm_lp.yaml --experiment_list 4a2yepf1 zmvkninr nhurx2vj iaeo9ay8 sjuvoihm --results_root logs_lightning/train-age/best_xlstm_lp --batch_size 8192

# sbatch run_scripts/slurm/slurm_train_l40s.sh run_scripts/eval_age.py configs/age/xlstm_dinoecg_ft.yaml --experiment_list 2rqz16et pd5dngrd wseqwwlc v8blelfh 1vrxssly --results_root logs_lightning/train-age/best_dinoecg_xlstm_ft --batch_size 8192
# sbatch run_scripts/slurm/slurm_train_l40s.sh run_scripts/eval_age.py configs/age/xlstm_dinoecg_lp.yaml --experiment_list lesld1f7 j5vlnotq 1a4bx29t f06h3v3p v0xhk2qb --results_root logs_lightning/train-age/best_dinoecg_xlstm_lp --batch_size 8192

# sbatch run_scripts/slurm/slurm_train_l40s.sh run_scripts/eval_age.py configs/age/xlstm_code15_lp.yaml --experiment_list mhfbcduo i3c01xwg 3y6g8lw8 urhyek0r x6ahg5np --results_root logs_lightning/train-age/best_code15_xlstm_lp --batch_size 8192
# sbatch run_scripts/slurm/slurm_train_l40s.sh run_scripts/eval_age.py configs/age/xlstm_code15_ft.yaml --experiment_list 789d9sk7  7nlj8a11 yrqtz1ra 9z38ud9t 0s81024y --results_root logs_lightning/train-age/best_code15_xlstm_ft --batch_size 8192

# sbatch run_scripts/slurm/slurm_train_l40s.sh run_scripts/eval_age.py configs/age/xlstm_supervised.yaml --experiment_list u4j3hh3i l2exsxvx zv3y7vyd ytl12maz dklblrpl --results_root logs_lightning/train-age/best_xlstm_sup --batch_size 8192
 
def build_test_dataloaders(config, batch_size):
    """Build the three test dataloaders (PTB-XL, MIMIC-IV, CPSC-2018)."""
    transforms = get_transforms(config, split='test')
    collate_fn = make_collate_fn_task(config, key_label='age')
    loader_kw  = dict(batch_size=batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=collate_fn)

    dl_code15 =  DataLoader(code.ECGCODE15AgeDataset(config, split='all', global_augmentations=transforms),  **loader_kw)
    dl_ptbxl = DataLoader(ptbxl.ECGPTBXLAgeDataset(config, split='all', global_augmentations=transforms), **loader_kw)
    dl_mimic = DataLoader(mimic_iv.ECGMIMICDataset(config, split='all', global_augmentations=transforms, downstream_task='age'), **loader_kw)
    dl_cpsc  = DataLoader(cpsc2018.ECGCPSC2018AgeDataset(config, split='all', global_augmentations=transforms), **loader_kw)

    return [dl_ptbxl, dl_mimic, dl_cpsc, dl_code15]

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
    
    map_idx_dataloader = {0: 'ptbxl', 1: 'mimic', 2: 'cpsc', 3:'code15'}

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
