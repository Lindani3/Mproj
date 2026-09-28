#!/bin/bash
#SBATCH --job-name=eval_v2_std
#SBATCH --output=slurm/logs/eval_v2_std_%j.out
#SBATCH --error=slurm/logs/eval_v2_std_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --partition=bigbatch

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Start  : $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate swapnet

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs data/r2

# =============================================================================
# R2 profiles — standard v2 models (1k curves, standard MSE)
# =============================================================================

echo "=== R2: Model 1a v2 ==="
python -u b6_r2_profile_v2.py \
    --checkpoint data/best_model_1a_v2.pt \
    --data       data/train_1a_v2.h5 \
    --label      1a_v2 \
    --out_dir    data/r2

echo "=== R2: Model 1b v2 ==="
python -u b6_r2_profile_v2.py \
    --checkpoint data/best_model_1b_v2.pt \
    --data       data/train_1b_v2.h5 \
    --label      1b_v2 \
    --out_dir    data/r2

echo "=== R2: Model 2 v2 ==="
python -u b6_r2_profile_v2.py \
    --checkpoint data/best_model_2_v2.pt \
    --data       data/train_2_v2.h5 \
    --label      2_v2 \
    --out_dir    data/r2

# =============================================================================
# DV01 profiles — standard v2 models
# =============================================================================

echo "=== DV01: Model 1a v2 ==="
python -u b8_dv01_profile.py \
    --checkpoint data/best_model_1a_v2.pt \
    --data       data/train_1a_v2.h5 \
    --label      1a_v2 \
    --out_dir    data/r2

echo "=== DV01: Model 1b v2 ==="
python -u b8_dv01_profile.py \
    --checkpoint data/best_model_1b_v2.pt \
    --data       data/train_1b_v2.h5 \
    --label      1b_v2 \
    --out_dir    data/r2

echo "=== DV01: Model 2 v2 ==="
python -u b8_dv01_profile.py \
    --checkpoint data/best_model_2_v2.pt \
    --data       data/train_2_v2.h5 \
    --label      2_v2 \
    --out_dir    data/r2

echo "=== All done ==="
echo "End: $(date)"
