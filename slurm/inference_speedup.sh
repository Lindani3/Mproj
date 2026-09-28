#!/bin/bash
#SBATCH --job-name=inf_speed
#SBATCH --output=slurm/logs/inference_speedup_%j.out
#SBATCH --error=slurm/logs/inference_speedup_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURMD_NODENAME"
echo "Start  : $(date)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate swapnet
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs data
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
python -u b10_inference_speedup.py
echo "End: $(date)"
