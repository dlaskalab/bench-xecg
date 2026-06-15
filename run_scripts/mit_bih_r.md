# MIT-BIH classification

## ecg founder OK

for i in {1..5}; do
    sbatch --job-name=fm_ft_mit_r  run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_mit_bih.py configs/mit_bih_r/ecgfm_ft.yaml --version $i
done

for i in {1..5}; do
    sbatch --job-name=fm_lp_mit_r  run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_mit_bih.py configs/mit_bih_r/ecgfm_lp.yaml --version $i
done

## jepa OK

for i in {1..5}; do
    sbatch --job-name=jepa_lp_mit_r  run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_mit_bih.py configs/mit_bih_r/jepa_lp.yaml --version $i
done

for i in {1..5}; do
    sbatch --job-name=jepa_ft_mit_r  run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_mit_bih.py configs/mit_bih_r/jepa_ft.yaml --version $i
done

## st-mem OK

for i in {1..5}; do
    sbatch --job-name=stmem_lp_mit_r  run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_mit_bih.py configs/mit_bih_r/stmem_lp.yaml --version $i
done

for i in {1..5}; do
    sbatch --job-name=stmem_ft_mit_r  run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_mit_bih.py configs/mit_bih_r/stmem_ft.yaml --version $i
done

## transformer OK

for i in {1..5}; do
    sbatch --job-name=trans_lp_mit_r  run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_mit_bih.py configs/mit_bih_r/transformer_lp.yaml --version $i
done

for i in {1..5}; do
    sbatch --job-name=trans_ft_mit_r  run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_mit_bih.py configs/mit_bih_r/transformer_ft.yaml --version $i
done

## xlstm OK

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_mit_r  run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_mit_bih.py configs/mit_bih_r/xlstm_lp.yaml --version $i
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_mit_r  run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_mit_bih.py configs/mit_bih_r/xlstm_ft.yaml --version $i
done 

## supervised


for i in {1..5}; do
    sbatch --job-name=xlstm_sup_mit_r  run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_mit_bih.py configs/mit_bih_r/xlstm_supervised.yaml --version $i
done 

## xlstm CODE 15 

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_mit_r  run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_mit_bih.py configs/mit_bih_r/xlstm_code15_lp.yaml --version $i
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_mit_r  run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_mit_bih.py configs/mit_bih_r/xlstm_code15_ft.yaml --version $i
done 

## ECG-CPC

for i in {1..5}; do
    sbatch --job-name=cpc_lp_mit_r  run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_mit_bih.py configs/mit_bih_r/ecgcpc_lp.yaml --version $i
done

for i in {1..5}; do
    sbatch --job-name=cpc_ft_mit_r  run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_mit_bih.py configs/mit_bih_r/ecgcpc_ft.yaml --version $i
done 


## xlstm DINOECG

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_mit_r  run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_mit_bih.py configs/mit_bih_r/xlstm_dinoecg_lp.yaml --version $i
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_mit_r  run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_mit_bih.py configs/mit_bih_r/xlstm_dinoecg_ft.yaml --version $i
done 
