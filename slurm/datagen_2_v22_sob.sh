#!/bin/bash
#SBATCH --job-name=dg_2s_v22
#SBATCH --output=slurm/logs/datagen_2_v22_sob_%j.out
#SBATCH --error=slurm/logs/datagen_2_v22_sob_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=bigbatch

echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Start  : $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate swapnet

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs data

python -u c3_datagen_2_v2_sob.py \
    --feds    data/feds_3000.csv \
    --out     data/train_2_v22_sob.h5 \
    --m_paths 5000 \
    --seed    42

echo "End: $(date)"
