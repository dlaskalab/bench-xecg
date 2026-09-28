import torch
import yaml

from bench_xecg.config import parse_config
from run_scripts.train_ptb_xl import train as train_ptb_xl
from run_scripts.train_cpsc2018 import train as train_cpsc2018
from run_scripts.train_survival import train as train_survival
from run_scripts.train_age import train as train_age
from run_scripts.train_sleep_apnea import train as train_sleep_apnea
from run_scripts.train_lab_mimic import train as train_lab_mimic
from run_scripts.train_mit_bih import train as train_mit_bih
from run_scripts.train_r_peak_intense import train as train_r_peak_intense
from run_scripts.train_ppg_af import train as train_ppg_af

import argparse
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--config_file', type=str, default='configs/task_config_list_lp.yaml', help='Path to the config file')
parser.add_argument('--num_runs', type=int, default=5, help='Number of runs to train for each task')
parser.add_argument('--wandb_log', action='store_true', help='Whether to log to wandb')

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')

    args = parser.parse_args()

    # read all config files task config list
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    # # ----------- (1) train ptb-xl  -----------
    # ptb_xl_config = parse_config(config['ptb_xl'], 'config_defaults/train_ptb_xl_defaults.yaml')
    # ptb_xl_config.deterministic = False

    # for i in range(args.num_runs):
    #     train_ptb_xl(ptb_xl_config, wandb=args.wandb_log)


    # # ----------- (2) train cpsc2018 -----------
    # cpsc2018_config = parse_config(config['cpsc2018'], 'config_defaults/train_cpsc2018_defaults.yaml')
    # cpsc2018_config.deterministic = False

    # for i in range(args.num_runs):
    #     train_cpsc2018(cpsc2018_config, wandb=args.wandb_log)


    # # ----------- (3) train mit bih arrhythmia classification -----------
    # mit_bih_config = parse_config(config['mit_bih'], 'config_defaults/train_mit_bih_defaults.yaml')
    # mit_bih_config.deterministic = False

    # for i in range(args.num_runs):
    #     train_mit_bih(mit_bih_config, wandb=args.wandb_log)


    # # ----------- (4) train mit bih r-peak detection -----------
    # r_peak_config = parse_config(config['r_peaks'], 'config_defaults/train_mit_bih_defaults.yaml')
    # r_peak_config.deterministic = False
    # r_peak_config.r_peaks_detection = True

    # for i in range(args.num_runs):
    #     train_mit_bih(r_peak_config, wandb=args.wandb_log)


    # # ------------ (5) train survival analysis -----------
    # survival_config = parse_config(config['survival'], 'config_defaults/train_survival_defaults.yaml')
    # survival_config.deterministic = False

    # for i in range(args.num_runs):
    #     train_survival(survival_config, wandb=args.wandb_log)


    # # ------------ (6) train age prediction -----------
    # age_config = parse_config(config['age'], 'config_defaults/train_age_defaults.yaml')
    # age_config.deterministic = False

    # for i in range(args.num_runs):
    #     train_age(age_config, wandb=args.wandb_log)


    # # ------------ (7) train sleep apnea detection -----------
    # sleep_apnea_config = parse_config(config['sleep_apnea'], 'config_defaults/train_sleep_apnea_defaults.yaml')
    # sleep_apnea_config.deterministic = False

    # for i in range(args.num_runs):
    #     train_sleep_apnea(sleep_apnea_config, wandb=args.wandb_log)


    # # ------------ (8) train lab mimic prediction -----------
    # lab_mimic_config = parse_config(config['lab_test'], 'config_defaults/train_blood_test_defaults.yaml')
    # lab_mimic_config.deterministic = False

    # for i in range(args.num_runs):
    #     train_lab_mimic(lab_mimic_config, wandb=args.wandb_log)


    # # ------------ (9) exercise r-peak detection -----------
    # exercise_r_peak_config = parse_config(config['exercise'], 'config_defaults/train_high_intensity_defaults.yaml')
    # exercise_r_peak_config.deterministic = False
    # exercise_r_peak_config.r_peaks_detection = True

    # for i in range(args.num_runs):
    #     train_r_peak_intense(exercise_r_peak_config, wandb=args.wandb_log)


    # ------------ (10) train ppg af -----------
    ppg_af_config = parse_config(config['ppg_af'], 'config_defaults/train_ppg_af_defaults.yaml')
    ppg_af_config.deterministic = False

    for i in range(args.num_runs):
        train_ppg_af(ppg_af_config, wandb=args.wandb_log)
