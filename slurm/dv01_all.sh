#!/bin/bash
#SBATCH --job-name=dv01_all
#SBATCH --output=slurm/logs/dv01_all_%j.out
#SBATCH --error=slurm/logs/dv01_all_%j.err
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

# --- Standard scalar models ---
echo "=== 1a std ==="
python -u b8_dv01_profile.py \
    --checkpoint data/best_model_1a.pt \
    --data       data/train_1a.h5 \
    --label      1a_std \
    --out_dir    data/r2

echo "=== 1b std ==="
python -u b8_dv01_profile.py \
    --checkpoint data/best_model_1b.pt \
    --data       data/train_1b.h5 \
    --label      1b_std \
    --out_dir    data/r2

echo "=== 2 std ==="
python -u b8_dv01_profile.py \
    --checkpoint data/best_model_2.pt \
    --data       data/train_2.h5 \
    --label      2_std \
    --out_dir    data/r2

# --- V22 profile models (best standard MSE variants) ---
echo "=== 1a v22 ==="
python -u b8_dv01_profile.py \
    --checkpoint data/best_model_1a_v22.pt \
    --data       data/train_1a_v22.h5 \
    --label      1a_v22 \
    --out_dir    data/r2

echo "=== 1b v22 ==="
python -u b8_dv01_profile.py \
    --checkpoint data/best_model_1b_v22.pt \
    --data       data/train_1b_v22.h5 \
    --label      1b_v22 \
    --out_dir    data/r2

echo "=== 2 v22 ==="
python -u b8_dv01_profile.py \
    --checkpoint data/best_model_2_v22.pt \
    --data       data/train_2_v22.h5 \
    --label      2_v22 \
    --out_dir    data/r2

echo "=== All done ==="
echo "End: $(date)"
