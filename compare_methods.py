"""
Comparative Evaluation: ProPS vs JEPA vs CEM vs Random
=======================================================
Reads hyperparams directly from the YAML configs so the
comparison uses exactly the same settings as main.py.

Usage:
    python analysis/compare_methods.py --env CartPole-v1 --seeds 0 1 2
    python analysis/compare_methods.py --env Swimmer-v4 --seeds 0 1 2 --skip_props
    python analysis/compare_methods.py --env CartPole-v1 --seeds 0 1 2 \
        --props_model qwen2.5:7b --ollama_host http://localhost:11434
"""

import os, argparse, yaml, logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# Config → env hyperparams mapping
# ─────────────────────────────────────────────────────────────────────────────

# Maps env_name → which config file to load JEPA hyperparams from
ENV_CONFIGS = {
    "CartPole-v1":              "configs/cartpole_jepa.yaml",
    "MountainCarContinuous-v0": "configs/cartpole_jepa.yaml",  # same arch
    "Swimmer-v4":               "configs/swimmer_jepa.yaml",
    "Hopper-v4":                "configs/swimmer_jepa.yaml",
    "InvertedPendulum-v4":      "configs/cartpole_jepa.yaml",
}

ENV_PARAM_DIM = {
    "CartPole-v1":              8,
    "MountainCarContinuous-v0": 6,
    "Swimmer-v4":               16,
    "Hopper-v4":                33,
    "InvertedPendulum-v4":      16,
}

# ProPS task prompts (from paper Appendix)
PROPS_PROMPTS = {
    "CartPole-v1": (
        "Design a quadruped robot for forward locomotion on flat terrain. "
        "The robot should have a main torso body, four legs each with at least two "
        "segments (thigh and shin), and actuated hip and knee joints. Include a floor "
        "plane. The robot should be stable when standing."
    ),
    "Swimmer-v4": (
        "Design a fish-like swimming robot for aquatic propulsion. "
        "Set fluid density to 1000 and viscosity to 0.8 in the <option> tag. "
        "Set gravity to 0. The robot should have a main body segment and 3-6 tail "
        "segments connected by hinge joints actuated for undulatory motion."
    ),
}


def load_jepa_config(env_name: str) -> dict:
    """Load the JEPA config for this environment, with env-specific overrides."""
    cfg_path = ENV_CONFIGS.get(env_name, "configs/cartpole_jepa.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # Override env-specific fields
    cfg["env_name"] = env_name
    cfg["param_dim"] = ENV_PARAM_DIM.get(env_name, cfg.get("param_dim", 8))
    # Cast any string numerics (YAML 1.1 scientific notation issue)
    for key in ("lr", "online_lr", "ema_decay", "noise_std", "desired_pct"):
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = float(cfg[key])
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Method runners
# ─────────────────────────────────────────────────────────────────────────────

def run_jepa(env_name: str, seed: int, max_iterations: int, device: str) -> dict:
    """Run pure JEPA using exact hyperparams from the config file."""
    from runner.jepa_runner import run_training_loop
    cfg = load_jepa_config(env_name)
    return run_training_loop(
        env_name=env_name,
        param_dim=cfg["param_dim"],
        d_model=cfg.get("d_model", 128),
        nhead=cfg.get("nhead", 4),
        num_enc_layers=cfg.get("num_enc_layers", 3),
        num_pred_layers=cfg.get("num_pred_layers", 3),
        dim_ff=cfg.get("dim_ff", 256),
        ema_decay=float(cfg.get("ema_decay", 0.996)),
        desired_pct=float(cfg.get("desired_pct", 80.0)),
        noise_std=float(cfg.get("noise_std", 0.5)),
        n_candidates=cfg.get("n_candidates", 20),
        max_iterations=max_iterations,
        n_eval_episodes=cfg.get("n_eval_episodes", 5),
        seed=seed,
        pretrained_ckpt=cfg.get("pretrained_ckpt", ""),
        online_finetune=cfg.get("online_finetune", True),
        online_lr=float(cfg.get("online_lr", 1e-4)),
        online_finetune_every=cfg.get("online_finetune_every", 5),
        online_finetune_steps=cfg.get("online_finetune_steps", 10),
        log_dir=f"results/compare/{env_name}",
        run_name=f"jepa_seed{seed}",
        device=device,
    )


def run_jepa_hybrid(env_name: str, seed: int, max_iterations: int, device: str) -> dict:
    """Run JEPA+Semantic hybrid using exact hyperparams from the config file."""
    from runner.jepa_semantics_runner import run_training_loop
    cfg = load_jepa_config(env_name)
    hybrid_cfg_path = ENV_CONFIGS.get(env_name, "configs/cartpole_jepa.yaml").replace(
        "_jepa.yaml", "_jepa_hybrid.yaml"
    )
    if os.path.exists(hybrid_cfg_path):
        with open(hybrid_cfg_path) as f:
            hybrid_cfg = yaml.safe_load(f)
        cfg.update(hybrid_cfg)
    return run_training_loop(
        env_name=env_name,
        param_dim=cfg["param_dim"],
        d_model=cfg.get("d_model", 128),
        nhead=cfg.get("nhead", 4),
        num_enc_layers=cfg.get("num_enc_layers", 3),
        num_pred_layers=cfg.get("num_pred_layers", 3),
        dim_ff=cfg.get("dim_ff", 256),
        ema_decay=float(cfg.get("ema_decay", 0.996)),
        desired_pct=float(cfg.get("desired_pct", 80.0)),
        noise_std=float(cfg.get("noise_std", 0.5)),
        n_candidates=cfg.get("n_candidates", 20),
        semantic_dim=cfg.get("semantic_dim", 384),
        max_iterations=max_iterations,
        n_eval_episodes=cfg.get("n_eval_episodes", 5),
        seed=seed,
        pretrained_ckpt=cfg.get("pretrained_ckpt", ""),
        online_finetune=cfg.get("online_finetune", True),
        online_lr=float(cfg.get("online_lr", 1e-4)),
        online_finetune_every=cfg.get("online_finetune_every", 5),
        online_finetune_steps=cfg.get("online_finetune_steps", 10),
        log_dir=f"results/compare/{env_name}",
        run_name=f"jepa_hybrid_seed{seed}",
        device=device,
    )


def run_props_baseline(
    env_name: str,
    seed: int,
    max_iterations: int,
    ollama_model: str = "qwen2.5:7b",
    ollama_host: str = "http://localhost:11434",
) -> dict:
    """
    Run the original ProPS baseline using Ollama.
    This is the key comparison: ProPS (LLM every iter) vs JEPA (zero LLM).
    """
    import ollama as _ollama
    from runner.jepa_runner import rollout_policy
    import re, time

    np.random.seed(seed)
    cfg = load_jepa_config(env_name)
    param_dim = cfg["param_dim"]
    n_eval = cfg.get("n_eval_episodes", 5)

    # ProPS system prompt
    system_prompt = (
        "You are an expert numerical optimizer. You will be given a history of "
        "parameter vectors and their rewards. Propose the next parameter vector "
        "to maximize reward. Output ONLY a JSON array of floats, no explanation."
    )

    def props_propose(history_params, history_rewards):
        """One ProPS LLM call: history → next θ."""
        if not history_params:
            return np.random.randn(param_dim)

        # Normalize rewards to [0,1] for the prompt
        r_arr = np.array(history_rewards)
        r_min, r_max = r_arr.min(), r_arr.max()
        r_norm = (r_arr - r_min) / (r_max - r_min + 1e-8)

        # Build in-context history (last 20 to stay within context)
        history_str = ""
        for i, (p, r) in enumerate(zip(history_params[-20:], r_norm[-20:])):
            history_str += f"params: {[round(x,4) for x in p]}, reward: {r:.4f}\n"

        user_prompt = (
            f"Environment: {env_name}, param_dim={param_dim}\n"
            f"History (normalized rewards):\n{history_str}"
            f"Propose the next {param_dim} parameters to maximize reward. "
            f"Output ONLY a JSON array of {param_dim} floats."
        )

        try:
            resp = _ollama.chat(
                model=ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 1.0},
            )
            text = resp["message"]["content"]
            # Extract JSON array
            m = re.search(r"\[([^\]]+)\]", text)
            if m:
                vals = [float(x.strip()) for x in m.group(1).split(",")]
                if len(vals) == param_dim:
                    return np.array(vals)
        except Exception as e:
            log.warning(f"ProPS LLM call failed: {e}")
        return np.random.randn(param_dim)

    # Run ProPS loop
    history_params, history_rewards = [], []
    all_rewards = []
    best = -np.inf
    t_start = time.time()

    # Warm-start
    n_warmup = max(10, param_dim * 2)
    for i in range(n_warmup):
        theta = np.random.randn(param_dim)
        r = rollout_policy(env_name, theta, n_eval_episodes=n_eval, seed=seed + i)
        history_params.append(theta)
        history_rewards.append(r)
        all_rewards.append(r)
        best = max(best, r)

    for t in range(max_iterations):
        theta = props_propose(history_params, history_rewards)
        r = rollout_policy(env_name, theta, n_eval_episodes=n_eval, seed=seed + 100 + t)
        history_params.append(theta)
        history_rewards.append(r)
        all_rewards.append(r)
        best = max(best, r)
        if (t + 1) % 50 == 0:
            elapsed = time.time() - t_start
            log.info(f"  ProPS [{t+1}/{max_iterations}] best={best:.2f} elapsed={elapsed:.0f}s")

    return {"rewards": all_rewards, "best_reward": best}


def run_cem_baseline(env_name: str, seed: int, max_iterations: int) -> dict:
    from runner.jepa_runner import rollout_policy
    cfg = load_jepa_config(env_name)
    param_dim = cfg["param_dim"]
    n_eval = cfg.get("n_eval_episodes", 5)
    np.random.seed(seed)

    pop_size = 20
    n_elite = max(2, pop_size // 5)
    mean = np.zeros(param_dim)
    std = np.ones(param_dim)
    rewards, best = [], -np.inf
    done = 0

    while done < max_iterations:
        population = mean + std * np.random.randn(pop_size, param_dim)
        pop_r = []
        for theta in population:
            if done >= max_iterations:
                break
            r = rollout_policy(env_name, theta, n_eval_episodes=n_eval, seed=seed + done)
            pop_r.append(r)
            rewards.append(r)
            best = max(best, r)
            done += 1
        if len(pop_r) >= n_elite:
            elites = population[np.argsort(pop_r)[-n_elite:]]
            mean = elites.mean(axis=0)
            std = elites.std(axis=0) + 1e-5

    return {"rewards": rewards, "best_reward": best}


def run_random_baseline(env_name: str, seed: int, max_iterations: int) -> dict:
    from runner.jepa_runner import rollout_policy
    cfg = load_jepa_config(env_name)
    param_dim = cfg["param_dim"]
    n_eval = cfg.get("n_eval_episodes", 5)
    np.random.seed(seed)
    rewards, best = [], -np.inf
    for i in range(max_iterations):
        theta = np.random.randn(param_dim)
        r = rollout_policy(env_name, theta, n_eval_episodes=n_eval, seed=seed + i)
        rewards.append(r)
        best = max(best, r)
    return {"rewards": rewards, "best_reward": best}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting and reporting
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "JEPA (ours)":    "#2196F3",
    "JEPA+Semantic":  "#4CAF50",
    "ProPS (qwen2.5:7b)":   "#9C27B0",
    "ProPS (deepseek-16b)": "#E91E63",
    "CEM":            "#FF9800",
    "Random":         "#9E9E9E",
}


def smooth(x, w=20):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_comparison(results: dict, env_name: str, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for method, seed_rewards in results.items():
        arr = np.stack([np.array(r) for r in seed_rewards])
        mean = arr.mean(0)
        sem  = arr.std(0) / np.sqrt(arr.shape[0])
        color = COLORS.get(method)
        x = np.arange(1, len(mean) + 1)
        sm = smooth(mean)
        sx = x[:len(sm)]
        axes[0].plot(x, mean, alpha=0.25, color=color)
        axes[0].fill_between(x, mean-sem, mean+sem, alpha=0.1, color=color)
        axes[1].plot(sx, sm, label=method, color=color, linewidth=2)
        axes[1].fill_between(sx,
            smooth(mean-sem)[:len(sx)], smooth(mean+sem)[:len(sx)],
            alpha=0.2, color=color)
    for ax, title in zip(axes, ["Raw Rewards", "Smoothed (window=20)"]):
        ax.set_xlabel("Iteration"); ax.set_ylabel("Reward")
        ax.set_title(f"{env_name} — {title}")
        ax.legend(); ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Plot saved: {save_path}")


def print_table(results: dict, env_name: str):
    print(f"\n{'='*65}")
    print(f"Results for {env_name}")
    print(f"{'='*65}")
    print(f"{'Method':<25} {'Final mean±std':>20} {'Best':>10}")
    print(f"{'-'*65}")
    for method, seed_rewards in results.items():
        finals = [np.mean(r[int(len(r)*0.8):]) for r in seed_rewards]
        bests  = [max(r) for r in seed_rewards]
        print(f"{method:<25} {np.mean(finals):>8.2f}±{np.std(finals):<10.2f}"
              f"{np.mean(bests):>10.2f}")
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",            default="CartPole-v1")
    parser.add_argument("--seeds",          type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max_iterations", type=int, default=400)
    parser.add_argument("--device",         default="cpu")
    parser.add_argument("--skip_hybrid",    action="store_true")
    parser.add_argument("--skip_props",     action="store_true",
                        help="Skip ProPS baseline (no Ollama available)")
    parser.add_argument("--props_model",    default="qwen2.5:7b",
                        help="Ollama model for ProPS baseline")
    parser.add_argument("--ollama_host",    default="http://localhost:11434")
    args = parser.parse_args()

    env = args.env
    results = {}

    log.info(f"\n{'='*50}\nRunning Pure JEPA on {env}\n{'='*50}")
    results["JEPA (ours)"] = [
        run_jepa(env, s, args.max_iterations, args.device)["rewards"]
        for s in args.seeds
    ]

    if not args.skip_hybrid:
        log.info(f"\n{'='*50}\nRunning JEPA+Semantic on {env}\n{'='*50}")
        results["JEPA+Semantic"] = [
            run_jepa_hybrid(env, s, args.max_iterations, args.device)["rewards"]
            for s in args.seeds
        ]

    if not args.skip_props:
        label = f"ProPS ({args.props_model})"
        log.info(f"\n{'='*50}\nRunning {label} on {env}\n{'='*50}")
        results[label] = [
            run_props_baseline(env, s, args.max_iterations,
                               args.props_model, args.ollama_host)["rewards"]
            for s in args.seeds
        ]

    log.info(f"\n{'='*50}\nRunning CEM on {env}\n{'='*50}")
    results["CEM"] = [
        run_cem_baseline(env, s, args.max_iterations)["rewards"]
        for s in args.seeds
    ]

    log.info(f"\n{'='*50}\nRunning Random on {env}\n{'='*50}")
    results["Random"] = [
        run_random_baseline(env, s, args.max_iterations)["rewards"]
        for s in args.seeds
    ]

    print_table(results, env)
    plot_comparison(results, env, f"results/compare/{env}/learning_curves.png")


if __name__ == "__main__":
    main()
