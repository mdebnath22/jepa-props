#!/bin/bash
# run_all_remaining.sh
# Master script for all remaining experiments.
# Run from: /scratch/mroycho1/mujoco/jepa_props
# Usage:
#   bash run_all_remaining.sh group_a        # no Ollama needed
#   bash run_all_remaining.sh group_b        # Ollama must be running
#   bash run_all_remaining.sh group_c        # needs checkpoints first
#   bash run_all_remaining.sh fix_rollout    # fixes swarm_jepa_runner.py bug
#   bash run_all_remaining.sh all            # everything (sequential)

set -e
export PYTHONWARNINGS="ignore::DeprecationWarning"
cd /scratch/mroycho1/mujoco/jepa_props

OLLAMA_HOST="http://:11434"
CKPT_CP="checkpoints/jepa_pretrained/jepa_pretrained.pt"
CKPT_SW="checkpoints/jepa_pretrained_swimmer/jepa_pretrained.pt"
CKPT_HO="results/hopper_ckpt/checkpoints/shuffle_none.pt"
CKPT_MC="results/mountaincar_ckpt/checkpoints/shuffle_none.pt"
CKPT_IP="results/invpendulum_ckpt/checkpoints/shuffle_none.pt"

run_group_a() {
    echo "========================================"
    echo "GROUP A: No-dependency experiments"
    echo "========================================"

    # A1: PSO term ablation on Swimmer (multi-seed) -- HIGHEST PRIORITY
    echo "--- A1: PSO term ablation Swimmer ---"
    for seed in 0 1 2; do
        python run_full_eval.py --mode ablations --ablation pso_terms \
            --tasks Swimmer-v4 --seeds $seed \
            --pretrained_ckpt $CKPT_SW \
            --max_iterations 400 --results_dir results/ablations_swimmer
    done

    # A1b: K sweep on Swimmer
    echo "--- A1b: K sweep Swimmer ---"
    for seed in 0 1 2; do
        python run_full_eval.py --mode ablations --ablation k_sweep \
            --tasks Swimmer-v4 --seeds $seed \
            --pretrained_ckpt $CKPT_SW \
            --max_iterations 400 --results_dir results/ablations_swimmer
    done

    # A1c: Sigma sweep on Swimmer
    echo "--- A1c: Sigma sweep Swimmer ---"
    for seed in 0 1 2; do
        python run_full_eval.py --mode ablations --ablation noise_sigma \
            --tasks Swimmer-v4 --seeds $seed \
            --pretrained_ckpt $CKPT_SW \
            --max_iterations 400 --results_dir results/ablations_swimmer
    done

    # A2: More domain baselines
    echo "--- A2: Baselines on Walker2d, Reacher ---"
    python run_full_eval.py --mode baselines \
        --tasks Walker2d-v4 Reacher-v4 \
        --seeds 0 1 2 3 4 --max_iterations 400 \
        --results_dir results/full_eval

    # A4: Latent CEM on Swimmer seeds 1,2 (seed 0 already done: best=349.16)
    echo "--- A4: Latent CEM Swimmer seeds 1,2 ---"
    python run_full_eval.py --mode ablations --ablation latent_cem \
        --tasks Swimmer-v4 --seeds 1 2 \
        --pretrained_ckpt $CKPT_SW \
        --max_iterations 400 --results_dir results/ablations_swimmer

    echo "Group A complete."
}

fix_rollout_bug() {
    echo "========================================"
    echo "FIXING swarm_jepa_runner.py rollout bug"
    echo "========================================"
    # swarm_jepa_runner.py uses runner.jepa_runner.rollout_policy (no-bias, wrong dims)
    # Fix: use envs.rollout_policy (bias-aware, correct for all tasks)
    python3 - << 'PYEOF'
with open("swarm_jepa_runner.py") as f:
    content = f.read()

old = "from runner.jepa_runner import rollout_policy"
new = "from envs import rollout_policy"

if old in content:
    content = content.replace(old, new, 1)
    with open("swarm_jepa_runner.py", "w") as f:
        f.write(content)
    print("Fixed: swarm_jepa_runner.py now uses envs.rollout_policy")
else:
    # Find whatever import is used
    for line in content.split('\n')[:30]:
        if 'rollout' in line:
            print("Current rollout import:", repr(line))
    print("ERROR: could not find import anchor -- check manually")
    exit(1)
PYEOF
}

run_group_c() {
    echo "========================================"
    echo "GROUP C: Swarm-JEPA on new domains"
    echo "(requires fix_rollout to be run first)"
    echo "========================================"

    # Verify fix is in place
    python3 -c "
import sys; sys.path.insert(0,'.')
from swarm_jepa_runner import run_swarm_jepa
print('Import OK')
"

    # C3: Hopper (checkpoint exists)
    if [ -f "$CKPT_HO" ]; then
        echo "--- C3: Swarm-JEPA on Hopper ---"
        python3 -c "
import sys, numpy as np, json; sys.path.insert(0,'.')
from swarm_jepa_runner import run_swarm_jepa
import logging; logging.basicConfig(level=logging.INFO)
results = {}
for seed in [0,1,2]:
    r = run_swarm_jepa('Hopper-v4', param_dim=36, seed=seed,
        max_iterations=400, device='cuda', K=3, noise_std=0.3,
        n_candidates=20, n_eval_episodes=10,
        pretrained_ckpt='$CKPT_HO',
        migration_every=30, isolation_every=60, isolation_steps=6)
    results[seed] = {'best': r['best_reward'], 'mean': float(np.mean(r['rewards']))}
    print(f'Hopper seed={seed}: best={r[\"best_reward\"]:.2f} mean={np.mean(r[\"rewards\"]):.2f}')
bests = [v['best'] for v in results.values()]
print(f'Hopper Swarm-JEPA: mean_best={np.mean(bests):.2f} std={np.std(bests):.2f}')
with open('results/swarm_jepa_hopper.json','w') as f:
    json.dump(results, f, indent=2)
"
    else
        echo "WARNING: Hopper checkpoint not found at $CKPT_HO -- train it first"
    fi

    # C1: MountainCar
    if [ -f "$CKPT_MC" ]; then
        echo "--- C1: Swarm-JEPA on MountainCar ---"
        python3 -c "
import sys, numpy as np, json; sys.path.insert(0,'.')
from swarm_jepa_runner import run_swarm_jepa
import logging; logging.basicConfig(level=logging.INFO)
results = {}
for seed in [0,1,2]:
    r = run_swarm_jepa('MountainCarContinuous-v0', param_dim=3, seed=seed,
        max_iterations=400, device='cpu', K=3, noise_std=0.3,
        n_candidates=20, n_eval_episodes=10,
        pretrained_ckpt='$CKPT_MC',
        migration_every=30, isolation_every=60, isolation_steps=6)
    results[seed] = {'best': r['best_reward'], 'mean': float(np.mean(r['rewards']))}
    print(f'MountainCar seed={seed}: best={r[\"best_reward\"]:.2f} mean={np.mean(r[\"rewards\"]):.2f}')
bests = [v['best'] for v in results.values()]
print(f'MountainCar Swarm-JEPA: mean_best={np.mean(bests):.2f}')
with open('results/swarm_jepa_mountaincar.json','w') as f:
    json.dump(results, f, indent=2)
"
    else
        echo "Train MountainCar checkpoint first:"
        echo "  python jepa_retraining_experiments.py --experiment reward_shuffle \\"
        echo "      --train_envs MountainCarContinuous-v0 --eval_envs MountainCarContinuous-v0 \\"
        echo "      --n_trajectories 5000 --n_epochs 100 --device cuda \\"
        echo "      --shuffle_types none --out_dir results/mountaincar_ckpt"
    fi

    # C2: InvertedPendulum
    if [ -f "$CKPT_IP" ]; then
        echo "--- C2: Swarm-JEPA on InvertedPendulum ---"
        python3 -c "
import sys, numpy as np, json; sys.path.insert(0,'.')
from swarm_jepa_runner import run_swarm_jepa
import logging; logging.basicConfig(level=logging.INFO)
results = {}
for seed in [0,1,2]:
    r = run_swarm_jepa('InvertedPendulum-v4', param_dim=5, seed=seed,
        max_iterations=400, device='cpu', K=3, noise_std=0.3,
        n_candidates=20, n_eval_episodes=10,
        pretrained_ckpt='$CKPT_IP',
        migration_every=30, isolation_every=60, isolation_steps=6)
    results[seed] = {'best': r['best_reward'], 'mean': float(np.mean(r['rewards']))}
    print(f'InvPendulum seed={seed}: best={r[\"best_reward\"]:.2f} mean={np.mean(r[\"rewards\"]):.2f}')
bests = [v['best'] for v in results.values()]
print(f'InvPendulum Swarm-JEPA: mean_best={np.mean(bests):.2f}')
with open('results/swarm_jepa_invpendulum.json','w') as f:
    json.dump(results, f, indent=2)
"
    else
        echo "Train InvertedPendulum checkpoint first:"
        echo "  python jepa_retraining_experiments.py --experiment reward_shuffle \\"
        echo "      --train_envs InvertedPendulum-v4 --eval_envs InvertedPendulum-v4 \\"
        echo "      --n_trajectories 5000 --n_epochs 100 --device cuda \\"
        echo "      --shuffle_types none --out_dir results/invpendulum_ckpt"
    fi

    echo "Group C complete."
}

run_group_b() {
    echo "========================================"
    echo "GROUP B: LLM experiments (Ollama required)"
    echo "========================================"

    # Verify Ollama is running
    python3 -c "
import requests
try:
    r = requests.get('$OLLAMA_HOST/api/tags', timeout=5)
    models = [m['name'] for m in r.json().get('models', [])]
    print('Ollama OK, models:', models)
except Exception as e:
    print('ERROR: Ollama not reachable:', e)
    exit(1)
"

    # B1: MountainCar ProPS/ProPS+/hints -- most important LLM experiment
    echo "--- B1: MountainCar ProPS/ProPS+/hints ---"
    python3 -c "
import sys, numpy as np, json; sys.path.insert(0,'.')
from envs import get_param_dim
from llm_baseline_plus import compare_prompt_modes
HOST = '$OLLAMA_HOST'
all_results = {}
for seed in [0,1,2]:
    results = compare_prompt_modes('MountainCarContinuous-v0',
        get_param_dim('MountainCarContinuous-v0'),
        seed=seed, max_iterations=400, ollama_host=HOST)
    for mode, r in results.items():
        key = f'seed{seed}_{mode}'
        all_results[key] = {'best': r['best_reward'], 'mean': float(np.mean(r['rewards'])),
                            'parse_fail': r['diagnostics']['parse_failure_rate']}
        print(f'seed={seed} {mode}: best={r[\"best_reward\"]:.2f} mean={np.mean(r[\"rewards\"]):.2f} parse_fail={r[\"diagnostics\"][\"parse_failure_rate\"]:.1%}')
with open('results/llm_mountaincar_modes.json','w') as f:
    json.dump(all_results, f, indent=2)
"

    # B2: Hint ablation on MountainCar
    echo "--- B2: Hint ablation MountainCar ---"
    python3 -c "
import sys, numpy as np, json; sys.path.insert(0,'.')
from envs import get_param_dim
from llm_baseline_plus import sweep_hint_ablations
HOST = '$OLLAMA_HOST'
all_results = {}
for seed in [0,1,2]:
    r = sweep_hint_ablations('MountainCarContinuous-v0',
        get_param_dim('MountainCarContinuous-v0'),
        seed=seed, max_iterations=400, ollama_host=HOST)
    for abl, res in r.items():
        key = f'seed{seed}_{abl}'
        all_results[key] = {'best': res['best_reward'], 'mean': float(np.mean(res['rewards']))}
        print(f'seed={seed} {abl}: best={res[\"best_reward\"]:.2f} mean={np.mean(res[\"rewards\"]):.2f}')
with open('results/llm_mountaincar_hint_ablation.json','w') as f:
    json.dump(all_results, f, indent=2)
"

    # B3: Prompt robustness (Tables 11,12) on MountainCar
    echo "--- B3: Prompt robustness MountainCar ---"
    python3 -c "
import sys, numpy as np, json; sys.path.insert(0,'.')
from envs import get_param_dim
from llm_baseline_plus import sweep_prompt_variations
HOST = '$OLLAMA_HOST'
all_results = {}
for mode in ['props', 'props_plus']:
    r = sweep_prompt_variations('MountainCarContinuous-v0',
        get_param_dim('MountainCarContinuous-v0'),
        seed=0, max_iterations=400, mode=mode, ollama_host=HOST)
    for var, res in r.items():
        key = f'{mode}_{var}'
        all_results[key] = {'best': res['best_reward'], 'mean': float(np.mean(res['rewards']))}
        print(f'{mode}/{var}: best={res[\"best_reward\"]:.2f} mean={np.mean(res[\"rewards\"]):.2f}')
with open('results/llm_prompt_robustness.json','w') as f:
    json.dump(all_results, f, indent=2)
"

    # B4: History length sweep (LLM side) on MountainCar
    echo "--- B4: History length sweep MountainCar ---"
    python3 -c "
import sys, numpy as np, json; sys.path.insert(0,'.')
from envs import get_param_dim
from llm_baseline_plus import sweep_history_length
HOST = '$OLLAMA_HOST'
r = sweep_history_length('MountainCarContinuous-v0',
    get_param_dim('MountainCarContinuous-v0'),
    seed=0, max_iterations=400, history_lengths=[1, 20, 40, 80, None],
    ollama_host=HOST)
for k, v in r.items():
    print(f'max_history={k}: best={v[\"best_reward\"]:.2f} mean={float(__import__(\"numpy\").mean(v[\"rewards\"])):.2f}')
with open('results/llm_history_length_mountaincar.json','w') as f:
    import json; json.dump({k: {'best': v['best_reward'], 'mean': float(__import__('numpy').mean(v['rewards']))} for k,v in r.items()}, f, indent=2)
"

    # B5: CartPole ProPS seeds 1,2 for completeness (seed 0 done already)
    echo "--- B5: CartPole ProPS seeds 1,2 ---"
    python3 -c "
import sys, numpy as np, json; sys.path.insert(0,'.')
from envs import get_param_dim
from llm_baseline_plus import compare_prompt_modes
HOST = '$OLLAMA_HOST'
# seed=0 already done: all modes best=500, mean=495
# seeds 1,2 already done too from earlier session
print('CartPole seeds 0,1,2 already complete -- skipping')
print('Results: seed=0 all modes best=500; seed=1 props=23 pp=30 pph=29; seed=2 props=36 pp=54 pph=69')
"

    # B6: LLM baseline on Swimmer (R15)
    echo "--- B6: LLM baseline on Swimmer ---"
    python3 -c "
import sys, numpy as np, json; sys.path.insert(0,'.')
from envs import get_param_dim
from llm_baseline import run_llm_iterative_optimizer
HOST = '$OLLAMA_HOST'
all_results = {}
for seed in [0,1,2]:
    r = run_llm_iterative_optimizer('Swimmer-v4',
        get_param_dim('Swimmer-v4'), seed=seed, max_iterations=400,
        ollama_model='qwen2.5:7b', ollama_host=HOST, n_episodes=10)
    all_results[seed] = {'best': r['best_reward'], 'mean': float(np.mean(r['rewards'])),
                         'parse_fail': r['diagnostics']['parse_failure_rate']}
    print(f'Swimmer LLM seed={seed}: best={r[\"best_reward\"]:.2f} mean={np.mean(r[\"rewards\"]):.2f}')
with open('results/llm_swimmer_baseline.json','w') as f:
    json.dump(all_results, f, indent=2)
"

    echo "Group B complete."
}

train_missing_checkpoints() {
    echo "========================================"
    echo "TRAINING missing task-specific checkpoints"
    echo "(needed before Group C)"
    echo "========================================"

    echo "--- Training MountainCar checkpoint ---"
    python jepa_retraining_experiments.py --experiment reward_shuffle \
        --train_envs MountainCarContinuous-v0 \
        --eval_envs MountainCarContinuous-v0 \
        --n_trajectories 5000 --n_epochs 100 --device cuda \
        --shuffle_types none --out_dir results/mountaincar_ckpt

    echo "--- Training InvertedPendulum checkpoint ---"
    python jepa_retraining_experiments.py --experiment reward_shuffle \
        --train_envs InvertedPendulum-v4 \
        --eval_envs InvertedPendulum-v4 \
        --n_trajectories 5000 --n_epochs 100 --device cuda \
        --shuffle_types none --out_dir results/invpendulum_ckpt

    echo "Checkpoint training complete."
    echo "Now run: bash run_all_remaining.sh fix_rollout && bash run_all_remaining.sh group_c"
}

# ── Main dispatch ──────────────────────────────────────────────────────────
case "${1:-help}" in
    group_a)     run_group_a ;;
    group_b)     run_group_b ;;
    group_c)     run_group_c ;;
    fix_rollout) fix_rollout_bug ;;
    train_ckpts) train_missing_checkpoints ;;
    all)
        run_group_a
        fix_rollout_bug
        train_missing_checkpoints
        run_group_c
        echo ""
        echo "======================================"
        echo "Group B (LLM) requires Ollama running."
        echo "Start Ollama then run:"
        echo "  bash run_all_remaining.sh group_b"
        echo "======================================"
        ;;
    help|*)
        echo "Usage: bash run_all_remaining.sh <command>"
        echo "Commands:"
        echo "  group_a      -- ablations + more domains (no Ollama needed)"
        echo "  group_b      -- all LLM experiments (Ollama must be running)"
        echo "  group_c      -- Swarm-JEPA on new domains (run fix_rollout first)"
        echo "  fix_rollout  -- patch swarm_jepa_runner.py rollout bug"
        echo "  train_ckpts  -- train MountainCar + InvertedPendulum checkpoints"
        echo "  all          -- A + fix + train + C (then manually run B with Ollama)"
        ;;
esac
