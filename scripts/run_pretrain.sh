#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM script: JEPA offline pretraining
# Run first before any policy search experiments.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=jepa_pretrain
#SBATCH --partition=public
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/jepa_pretrain_%j.out
#SBATCH --error=logs/jepa_pretrain_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mroycho1@asu.edu

module purge
module load mamba/latest
source activate mdebnath

cd /scratch/mroycho1/mujoco/jepa_props

mkdir -p logs data checkpoints

echo "=== JEPA Pretraining ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python main.py --config configs/pretrain_jepa.yaml

echo "End: $(date)"
