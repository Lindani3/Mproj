#!/bin/bash
#SBATCH --job-name=cva_1b
#SBATCH --output=slurm/logs/cva_bump_1b_v22_%j.out
#SBATCH --error=slurm/logs/cva_bump_1b_v22_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Start  : $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate swapnet
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

python -u test_cva_bump_recovery.py \
    --checkpoint data/best_model_1b_v22.pt \
    --model_type 1b

echo "End: $(date)"
