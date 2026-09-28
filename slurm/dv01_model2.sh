#!/bin/bash
#SBATCH --job-name=dv01_model2
#SBATCH --output=slurm/logs/dv01_model2_%j.out
#SBATCH --error=slurm/logs/dv01_model2_%j.err
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

# All Model 2 variants — EPE DV01 reference (b9), n_mc=500
echo "=== 2 std ==="
python -u b9_epe_dv01.py \
    --checkpoint data/best_model_2.pt \
    --data       data/train_2.h5 \
    --label      2_std \
    --n_mc       500 \
    --out_dir    data/r2

echo "=== 2 v2 ==="
python -u b9_epe_dv01.py \
    --checkpoint data/best_model_2_v2.pt \
    --data       data/train_2_v2.h5 \
    --label      2_v2 \
    --n_mc       500 \
    --out_dir    data/r2

echo "=== 2 v22 ==="
python -u b9_epe_dv01.py \
    --checkpoint data/best_model_2_v22.pt \
    --data       data/train_2_v22.h5 \
    --label      2_v22 \
    --n_mc       500 \
    --out_dir    data/r2

echo "=== 2 v22 sob ==="
python -u b9_epe_dv01.py \
    --checkpoint data/best_model_2_v22_sob.pt \
    --data       data/train_2_v22.h5 \
    --label      2_v22_sob \
    --n_mc       500 \
    --out_dir    data/r2

echo "=== 2 v2 sob pw ==="
python -u b9_epe_dv01.py \
    --checkpoint data/best_model_2_v2_sob_pw.pt \
    --data       data/train_2_v2.h5 \
    --label      2_v2_sob_pw \
    --n_mc       500 \
    --out_dir    data/r2

echo "=== All done ==="
echo "End: $(date)"
