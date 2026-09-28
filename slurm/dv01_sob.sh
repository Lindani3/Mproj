#!/bin/bash
#SBATCH --job-name=dv01_sob
#SBATCH --output=slurm/logs/dv01_sob_%j.out
#SBATCH --error=slurm/logs/dv01_sob_%j.err
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

# Model 1a Sobolev variants (swap-value DV01 reference via b8)
echo "=== 1a v2 sob ==="
python -u b8_dv01_profile.py \
    --checkpoint data/best_model_1a_v2_sob.pt \
    --data       data/train_1a_v2.h5 \
    --label      1a_v2_sob \
    --out_dir    data/r2

echo "=== 1a v22 sob ==="
python -u b8_dv01_profile.py \
    --checkpoint data/best_model_1a_v22_sob.pt \
    --data       data/train_1a_v22.h5 \
    --label      1a_v22_sob \
    --out_dir    data/r2

# Model 1b Sobolev variants (swap-value DV01 reference via b8)
echo "=== 1b v22 sob ==="
python -u b8_dv01_profile.py \
    --checkpoint data/best_model_1b_v22_sob.pt \
    --data       data/train_1b_v22.h5 \
    --label      1b_v22_sob \
    --out_dir    data/r2

echo "=== 1b v2 sob pw ==="
python -u b8_dv01_profile.py \
    --checkpoint data/best_model_1b_v2_sob_pw.pt \
    --data       data/train_1b_v2.h5 \
    --label      1b_v2_sob_pw \
    --out_dir    data/r2

echo "=== All done ==="
echo "End: $(date)"
