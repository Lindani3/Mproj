#!/bin/bash
#SBATCH --job-name=dv01_sanity
#SBATCH --output=slurm/logs/dv01_sanity_%j.out
#SBATCH --error=slurm/logs/dv01_sanity_%j.err
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Start  : $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate swapnet
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

python -u verify_dv01_loss_sanity.py

echo "End: $(date)"
