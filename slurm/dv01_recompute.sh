#!/bin/bash
#SBATCH --job-name=dv01_fix
#SBATCH --output=slurm/logs/dv01_recompute_%j.out
#SBATCH --error=slurm/logs/dv01_recompute_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Start  : $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swapnet
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs data/r2

echo "=== Model 1a Baseline (v2) ==="
python -u b8_dv01_profile.py \
    --checkpoint data/best_model_1a_v2.pt \
    --data       data/train_1a_v2.h5 \
    --label      1a_v2_fixed \
    --out_dir    data/r2

echo "=== Model 1a Scaled (v22) ==="
python -u b8_dv01_profile.py \
    --checkpoint data/best_model_1a_v22.pt \
    --data       data/train_1a_v22.h5 \
    --label      1a_v22_fixed \
    --out_dir    data/r2

echo "=== Model 1b Baseline (v2) ==="
python -u b8_dv01_profile.py \
    --checkpoint data/best_model_1b_v2.pt \
    --data       data/train_1b_v2.h5 \
    --label      1b_v2_fixed \
    --out_dir    data/r2

echo "=== Model 1b Scaled (v22) ==="
python -u b8_dv01_profile.py \
    --checkpoint data/best_model_1b_v22.pt \
    --data       data/train_1b_v22.h5 \
    --label      1b_v22_fixed \
    --out_dir    data/r2

echo "=== Model 2 Baseline (v2) ==="
python -u b8_dv01_profile.py \
    --checkpoint data/best_model_2_v2.pt \
    --data       data/train_2_v2.h5 \
    --label      2_v2_fixed \
    --out_dir    data/r2

echo "=== Model 2 Scaled (v22) ==="
python -u b8_dv01_profile.py \
    --checkpoint data/best_model_2_v22.pt \
    --data       data/train_2_v22.h5 \
    --label      2_v22_fixed \
    --out_dir    data/r2

echo "=== All done ==="
echo "End: $(date)"
