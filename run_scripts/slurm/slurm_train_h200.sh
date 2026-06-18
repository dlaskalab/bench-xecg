#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --partition=h200
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=512M
#SBATCH --time=14-00:00:00
#SBATCH -o ./logs/slurm_output_%j_%x.out

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_LOCALID: $SLURM_LOCALID"

export TORCH_CUDA_ARCH_LIST="9.0a"

export CUDA_HOME=$HOME/.cuda/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$CPATH

ulimit -n
ulimit -n 16384
ulimit -n

# run script from above
SCRIPT=$1
CONFIG=$2
shift 2

srun uv run "$SCRIPT" --config_file "$CONFIG" "$@"