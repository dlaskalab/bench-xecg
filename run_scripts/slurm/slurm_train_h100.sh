#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --partition=h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=512M
#SBATCH --time=14-00:00:00
#SBATCH -o ./logs/slurm_output_%j_%x.out

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_LOCALID: $SLURM_LOCALID"

#source /home/$USER/.bashrc
#conda init
#conda activate xlstm_pretrained
ulimit -n
ulimit -n 16384
ulimit -n

export TORCH_CUDA_ARCH_LIST="9.0"

# export CUDA_HOME=/usr/local/cuda
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
# export LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LIBRARY_PATH

# run script from above

SCRIPT=$1
CONFIG=$2
shift 2

srun uv run "$SCRIPT" --config_file "$CONFIG" "$@"