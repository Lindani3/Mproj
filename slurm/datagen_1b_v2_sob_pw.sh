#!/bin/bash
#SBATCH --job-name=dg_1bs_pw
#SBATCH --output=slurm/logs/datagen_1b_v2_sob_pw_%j.out
#SBATCH --error=slurm/logs/datagen_1b_v2_sob_pw_%j.err
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

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

python -u c1_datagen_1b_v2_sob_pw.py \
    --feds    data/feds_1000.csv \
    --out     data/train_1b_v2_sob_pw.h5 \
    --m_paths 5000 \
    --seed    42

echo "End: $(date)"
