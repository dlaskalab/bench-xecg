#!/bin/bash -l
#SBATCH --job-name=pretrain
#SBATCH --partition=h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=512M
#SBATCH --time=7-00:00:00
#SBATCH -o ./logs/slurm_output_%j_%x.out # STDOUT

echo "Visible GPUs: "
echo $CUDA_VISIBLE_DEVICES
export TORCH_CUDA_ARCH_LIST="9.0"

ulimit -n
ulimit -n 16384
ulimit -n

SCRIPT=$1
CONFIG=$2
shift 2

srun uv run "$SCRIPT" --config_file "$CONFIG" "$@"