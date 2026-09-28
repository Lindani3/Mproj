#!/bin/bash
#SBATCH --job-name=dv01_diag_sob
#SBATCH --output=slurm/logs/diagnose_dv01_precision_sob_%j.out
#SBATCH --error=slurm/logs/diagnose_dv01_precision_sob_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Start  : $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate swapnet
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

python -u diagnose_dv01_precision.py \
    --checkpoint data/best_model_1a_v22_sob.pt \
    --data       data/train_1a_v22.h5

echo "End: $(date)"
