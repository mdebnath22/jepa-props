#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM array job: run pure JEPA across all 15 ProPS environments × 10 seeds
# Matches ProPS's evaluation protocol exactly.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=jepa_bench
#SBATCH --partition=htc
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/jepa_bench_%A_%a.out
#SBATCH --error=logs/jepa_bench_%A_%a.err
#SBATCH --array=0-149   # 15 envs × 10 seeds
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mroycho1@asu.edu

module purge
module load mamba/latest
conda activate mdebnath

cd /scratch/mroycho1/jepa_props
mkdir -p logs results

# ── Environment table (matching ProPS) ───────────────────────────────────────
ENVS=(
    "CartPole-v1:8"
    "MountainCarContinuous-v0:6"
    "InvertedPendulum-v4:16"
    "InvertedDoublePendulum-v4:22"
    "Reacher-v4:14"
    "Swimmer-v4:16"
    "Hopper-v4:33"
    "Walker2d-v4:51"
    "FrozenLake-v1:64"
    "CliffWalking-v0:144"
    "MountainCar-v0:9"
    "HalfCheetah-v4:102"
    "Humanoid-v4:595"
    "Ant-v4:352"
    "Pusher-v4:56"
)

N_ENVS=15
N_SEEDS=10

ENV_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))

ENV_ENTRY="${ENVS[$ENV_IDX]}"
ENV_NAME=$(echo "$ENV_ENTRY" | cut -d: -f1)
PARAM_DIM=$(echo "$ENV_ENTRY" | cut -d: -f2)

echo "=== JEPA Benchmark ==="
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Env: $ENV_NAME, param_dim: $PARAM_DIM, seed: $SEED"
echo "Start: $(date)"

python -c "
import sys
sys.path.insert(0, '.')
from runner.jepa_runner import run_training_loop
import logging
logging.basicConfig(level=logging.INFO)
results = run_training_loop(
    env_name='$ENV_NAME',
    param_dim=$PARAM_DIM,
    max_iterations=400,
    n_eval_episodes=5,
    seed=$SEED,
    d_model=128,
    nhead=4,
    num_enc_layers=3,
    num_pred_layers=3,
    dim_ff=256,
    desired_pct=80.0,
    noise_std=0.05,
    n_candidates=5,
    online_finetune=True,
    online_lr=1e-4,
    online_finetune_every=10,
    online_finetune_steps=5,
    log_dir='results/${ENV_NAME}_jepa',
    run_name='seed$SEED',
    device='cpu',
)
print(f'Best reward: {results[\"best_reward\"]:.4f}')
"

echo "End: $(date)"
