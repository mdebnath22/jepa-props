#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM script: Run comparison experiments for a single environment
# Usage: sbatch run_compare.sh CartPole-v1 8
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=jepa_compare
#SBATCH --partition=htc
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --output=logs/jepa_compare_%j.out
#SBATCH --error=logs/jepa_compare_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mroycho1@asu.edu

ENV_NAME=${1:-CartPole-v1}
PARAM_DIM=${2:-8}

module purge
module load mamba/latest
conda activate mdebnath

cd /scratch/mroycho1/jepa_props
mkdir -p logs results/compare

echo "=== JEPA vs Baselines: $ENV_NAME ==="
echo "Start: $(date)"

python analysis/compare_methods.py \
    --env "$ENV_NAME" \
    --seeds 0 1 2 3 4 \
    --max_iterations 400 \
    --device cpu

echo "End: $(date)"
