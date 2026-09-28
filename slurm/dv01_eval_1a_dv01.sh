#!/bin/bash
#SBATCH --job-name=dv01_eval1a
#SBATCH --output=slurm/logs/dv01_eval_1a_dv01_%j.out
#SBATCH --error=slurm/logs/dv01_eval_1a_dv01_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Start  : $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate swapnet
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs data/r2

python -u b8_dv01_profile.py \
    --checkpoint data/best_model_1a_v2_dv01.pt \
    --data       data/train_1a_v2.h5 \
    --label      1a_v2_dv01 \
    --out_dir    data/r2

echo "End: $(date)"
