#!/bin/bash
#SBATCH --job-name=dg_1av2
#SBATCH --output=slurm/logs/datagen_1a_v2_%j.out
#SBATCH --error=slurm/logs/datagen_1a_v2_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=bigbatch

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Start  : $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate swapnet

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs data

python -u c1_datagen_1a_v2.py \
    --out  data/train_1a_v2.h5 \
    --feds data/feds_3000.csv \
    --seed 42

echo "End: $(date)"
