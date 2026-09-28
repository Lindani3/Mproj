#!/bin/bash
#SBATCH --job-name=r2_sob_v2
#SBATCH --output=slurm/logs/r2_sob_v2_%j.out
#SBATCH --error=slurm/logs/r2_sob_v2_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --partition=bigbatch

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Start  : $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate swapnet

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs data/r2

echo "=== Model 1a v22 Sobolev ==="
python -u b6_r2_profile_v2.py \
    --checkpoint data/best_model_1a_v22_sob.pt \
    --data       data/train_1a_v22_sob.h5 \
    --label      1a_v22_sob \
    --out_dir    data/r2

echo "=== Model 1b v2 Sobolev ==="
python -u b6_r2_profile_v2.py \
    --checkpoint data/best_model_1b_v2_sob.pt \
    --data       data/train_1b_v2_sob.h5 \
    --label      1b_v2_sob \
    --out_dir    data/r2

echo "=== Model 2 v2 Sobolev pathwise ==="
python -u b6_r2_profile_v2.py \
    --checkpoint data/best_model_2_v2_sob_pw.pt \
    --data       data/train_2_v2_sob_pw.h5 \
    --label      2_v2_sob_pw \
    --out_dir    data/r2

echo "=== Model 1b v2 Sobolev pathwise ==="
python -u b6_r2_profile_v2.py \
    --checkpoint data/best_model_1b_v2_sob_pw.pt \
    --data       data/train_1b_v2_sob_pw.h5 \
    --label      1b_v2_sob_pw \
    --out_dir    data/r2

echo "=== All done ==="
echo "End: $(date)"
