# Age

## ecg founder OK

for i in {1..4}; do
    sbatch --job-name=fm_ft_age run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_age.py configs/age/ecgfm_ft.yaml --version $i 
done

for i in {1..4}; do
    sbatch --job-name=fm_lp_age run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_age.py configs/age/ecgfm_lp.yaml --version $i
done

## jepa OK

for i in {1..5}; do
    sbatch --job-name=jepa_lp_age run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_age.py configs/age/jepa_lp.yaml --version $i 
done

for i in {1..5}; do
    sbatch --job-name=jepa_ft_age run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_age.py configs/age/jepa_ft.yaml --version $i
done

## st-mem OK

for i in {1..5}; do
    sbatch --job-name=stmem_lp_age run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_age.py configs/age/stmem_lp.yaml --version $i
done

for i in {1..5}; do
    sbatch --job-name=stmem_ft_age run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_age.py configs/age/stmem_ft.yaml --version $i
done

## transformer OK

for i in {1..5}; do
    sbatch --job-name=trans_lp_age run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_age.py configs/age/transformer_lp.yaml --version $i
done

for i in {1..5}; do
    sbatch --job-name=trans_ft_age run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_age.py configs/age/transformer_ft.yaml --version $i 
done

## xlstm OK

for i in {1..4}; do
    sbatch --job-name=xlstm_lp_age run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_age.py configs/age/xlstm_lp.yaml --version $i 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_age run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_age.py configs/age/xlstm_ft.yaml --version $i
done 

## supervised

for i in {1..5}; do
    sbatch --job-name=xlstm_sup_age run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_age.py configs/age/xlstm_supervised.yaml --version $i
done 

## xlstm CODE15

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_age run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_age.py configs/age/xlstm_code15_lp.yaml --version $i 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_age run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_age.py configs/age/xlstm_code15_ft.yaml --version $i
done 

## ECG-CPC

for i in {1..5}; do
    sbatch --job-name=cpc_lp_age run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_age.py configs/age/ecgcpc_lp.yaml --version $i 
done

for i in {1..5}; do
    sbatch --job-name=cpc_ft_age run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_age.py configs/age/ecgcpc_ft.yaml --version $i
done 

## xlstm DINOECG

for i in {1..5}; do
    sbatch --job-name=xlstm_lp_age run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_age.py configs/age/xlstm_dinoecg_lp.yaml --version $i 
done

for i in {1..5}; do
    sbatch --job-name=xlstm_ft_age run_scripts/slurm/slurm_train_l40s.sh run_scripts/train_age.py configs/age/xlstm_dinoecg_ft.yaml --version $i
done 
