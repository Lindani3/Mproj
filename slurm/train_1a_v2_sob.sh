#!/bin/bash
#SBATCH --job-name=tr_1as_v2
#SBATCH --output=slurm/logs/train_1a_v2_sob_%j.out
#SBATCH --error=slurm/logs/train_1a_v2_sob_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=bigbatch

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Start  : $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate swapnet

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs data

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

python -u c5_train_v2_sob.py \
    --model       1a \
    --data        data/train_1a_v2_sob.h5 \
    --epochs      200 \
    --batch_size  1024 \
    --lr          1e-3 \
    --hidden_dim  128 \
    --n_layers    2 \
    --lam         0.001 \
    --alpha       1.0 \
    --num_threads 8 \
    --device      cpu \
    --save        data/best_model_1a_v2_sob.pt \
    --seed        42

echo "End: $(date)"
