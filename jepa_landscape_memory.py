"""
jepa_landscape_memory.py — Three interconnected experiments probing the
relationship between optimization landscape structure and JEPA's memory
requirements. Together they support the central claim:

  "The optimization history required by a learned optimizer is
   environment-dependent. On smooth landscapes, JEPA exploits the full
   trajectory; on rugged landscapes, the current best dominates."

Experiments:

E_ORD. Reward Ordinality Test
  Does JEPA care about absolute reward values or only their relative ranking?
  Constructs four versions of the same history:
    (1) original rewards
    (2) rank-transformed rewards (1, 2, 3, ... N)
    (3) monotonic transformation (log or square)
    (4) randomly permuted rewards (destroys ordering)
  If (1)≈(2)≈(3) >> (4): JEPA learned ordering structure, not magnitudes.
  If (1)≈(2)≈(3)≈(4): JEPA ignores reward signal entirely (concerning).
  If (1) >> (2)≈(3) >> (4): JEPA uses both magnitude and ordering.

E_RUG. Landscape Ruggedness Characterization
  For each environment, compute:
    - Autocorrelation length (how far apart must samples be to be uncorrelated?)
    - Fitness-distance correlation (FDC): do nearby params give similar rewards?
    - Local smoothness: average |r(theta+eps) - r(theta)| / ||eps||
    - Basin size estimate: size of the region around the best-known theta
      that maintains >90% of best reward
  These are standard landscape analysis metrics from the EAs literature.

E_MEM. History Dependence vs Landscape Ruggedness
  Runs the E8 forgetting experiment (full/best_only/recent/random/top_rewards)
  across multiple environments, computes:
    History Dependence = (R_full - R_best_only) / R_full
  Then correlates History Dependence with landscape ruggedness metrics.
  The predicted result: smooth landscapes → high HD (trajectory matters);
  rugged landscapes → low HD (best point dominates).

USAGE:
    python jepa_landscape_memory.py \
        --ckpt_cartpole checkpoints/jepa_pretrained/jepa_pretrained.pt \
        --ckpt_swimmer  checkpoints/jepa_pretrained_swimmer/jepa_pretrained.pt \
        --out_dir results/landscape_memory \
        --experiments all
"""

import os, sys, json, argparse, logging, time
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
sys.path.insert(0, os.getcwd())


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

try:
    from runner.jepa_runner import rollout_policy as _runner_rollout
except Exception:
    _runner_rollout = None
from envs import rollout_policy as _envs_rollout


def rollout_policy(env_name, theta, n_episodes, seed):
    """Dispatch: use native 8/16-dim runner for pretrained ckpts,
    envs rollout for task-specific ckpts with bias-augmented dims."""
    if len(theta) in (8, 16) and _runner_rollout is not None:
        try:
            return _runner_rollout(env_name, theta, n_episodes, seed)
        except Exception:
            pass
    return _envs_rollout(env_name, theta, n_episodes, seed)


def load_agent(ckpt_path, param_dim, device="cpu"):
    import torch
    from agent.jepa_agent import JEPAAgent
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model", ckpt)
    # Infer actual param_dim from checkpoint decoder bias shape
    decoder_bias = [v for k, v in sd.items()
                    if "decoder" in k and k.endswith(".bias")
                    and "decoder.0" not in k and "decoder.2" not in k]
    if decoder_bias:
        inferred = decoder_bias[-1].shape[0]
        if inferred != param_dim:
            import logging
            logging.getLogger(__name__).info(
                f"load_agent: param_dim={inferred} inferred from checkpoint "
                f"(arg was {param_dim})")
            param_dim = inferred
    agent = JEPAAgent(param_dim=param_dim, device=device)
    agent.load_state_dict(sd, strict=False)
    agent.eval()
    return agent


def collect_random_samples(env_name, param_dim, n_samples=500,
                            n_episodes=5, seed=0):
    """Collect (theta, reward) pairs via pure random sampling.
    Used for landscape characterization."""

    rng = np.random.RandomState(seed)
    thetas, rewards = [], []
    for i in range(n_samples):
        theta = rng.randn(param_dim)
        r = rollout_policy(env_name, theta, n_episodes, seed + i)
        thetas.append(theta)
        rewards.append(r)
    return np.array(thetas), np.array(rewards)


# ─────────────────────────────────────────────────────────────────────────────
# E_ORD: Reward Ordinality Test
# ─────────────────────────────────────────────────────────────────────────────

def run_reward_ordinality(ckpt_path, env_name, param_dim,
                           n_seeds=5, max_iterations=200,
                           n_eval_episodes=5, device="cpu"):
    """
    Does JEPA use absolute reward magnitudes, relative ordering, or neither?

    Four reward transformations applied to the warm-up history:
      original:   rewards as collected
      ranked:     replace rewards with their ranks (1..N)
      monotonic:  sign-preserving monotonic transform (log1p of shifted rewards)
      permuted:   randomly permute rewards (destroys all ordering information)

    After warm-up with transformed history, JEPA proposes normally.
    The transformation only affects what JEPA 'sees' in context, not
    what rewards are actually collected during the main optimization loop.
    """
    import torch

    transformations = {
        "original": lambda r: r,
        "ranked": lambda r: np.argsort(np.argsort(r)).astype(float) + 1,
        "monotonic_log": lambda r: np.sign(r) * np.log1p(np.abs(r)),
        "monotonic_sqrt": lambda r: np.sign(r) * np.sqrt(np.abs(r)),
        "permuted": lambda r: np.random.RandomState(42).permutation(r),
    }

    results = {}

    for transform_name, transform_fn in transformations.items():
        seed_bests, seed_means = [], []

        for seed in range(n_seeds):
            agent = load_agent(ckpt_path, param_dim, device)
            rng = np.random.RandomState(seed)

            # Warm-up: collect real (theta, reward) pairs
            n_warmup = max(16, param_dim * 2)
            warmup_thetas, warmup_rewards = [], []
            for i in range(n_warmup):
                theta = rng.randn(param_dim)
                r = rollout_policy(env_name, theta, n_eval_episodes, seed + i)
                warmup_thetas.append(theta)
                warmup_rewards.append(r)

            # Apply transformation to rewards before adding to JEPA's history
            transformed_rewards = transform_fn(np.array(warmup_rewards))
            for theta, r_transformed in zip(warmup_thetas, transformed_rewards):
                agent.add_to_history(theta, float(r_transformed))

            # Main optimization: JEPA proposes, evaluate with REAL rewards
            rewards = list(warmup_rewards)  # track REAL rewards
            for t in range(max_iterations - n_warmup):
                theta = agent.propose()
                r_real = rollout_policy(env_name, theta, n_eval_episodes,
                                        seed + n_warmup + t)

                # Add to history with the SAME transformation applied to new rewards
                # (so JEPA sees a consistent transformed signal throughout)
                all_real_so_far = np.array(rewards + [r_real])
                r_transformed_new = transform_fn(all_real_so_far)[-1]
                agent.add_to_history(theta, float(r_transformed_new))
                rewards.append(r_real)

            seed_bests.append(float(max(rewards)))
            seed_means.append(float(np.mean(rewards)))

        results[transform_name] = {
            "mean_best": float(np.mean(seed_bests)),
            "std_best": float(np.std(seed_bests)),
            "mean_reward": float(np.mean(seed_means)),
            "std_reward": float(np.std(seed_means)),
        }
        log.info(f"E_ORD {transform_name}: "
                 f"mean_best={np.mean(seed_bests):.2f}±{np.std(seed_bests):.2f}")

    # Compute ordinality score: how much does permutation hurt vs monotonic?
    original = results["original"]["mean_best"]
    permuted = results["permuted"]["mean_best"]
    ranked = results["ranked"]["mean_best"]
    monotonic = results["monotonic_log"]["mean_best"]

    # Normalized ordinality score: 0 = JEPA ignores ordering, 1 = needs exact magnitudes
    ordering_sensitivity = float(
        (original - permuted) / (abs(original) + 1e-6)
    )
    magnitude_sensitivity = float(
        (original - ranked) / (abs(original) + 1e-6)
    )

    results["analysis"] = {
        "ordering_sensitivity": ordering_sensitivity,
        "magnitude_sensitivity": magnitude_sensitivity,
        "interpretation": (
            f"ordering_sensitivity={ordering_sensitivity:.3f}: "
            f"{'JEPA needs ordering info (permutation hurts)' if ordering_sensitivity > 0.05 else 'JEPA ignores ordering'}. "
            f"magnitude_sensitivity={magnitude_sensitivity:.3f}: "
            f"{'JEPA uses absolute magnitudes (rank transform hurts)' if magnitude_sensitivity > 0.05 else 'JEPA uses only relative ordering (rank-invariant)'}. "
            f"If ordering_sensitivity>0 but magnitude_sensitivity≈0: "
            f"JEPA learned ordering structure, not magnitudes -- the cleanest result."
        )
    }
    log.info(f"E_ORD analysis: ordering_sens={ordering_sensitivity:.3f} "
             f"magnitude_sens={magnitude_sensitivity:.3f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# E_RUG: Landscape Ruggedness Characterization
# ─────────────────────────────────────────────────────────────────────────────

def measure_landscape_ruggedness(env_name, param_dim,
                                  n_samples=300, n_episodes=5,
                                  seed=0, eps=0.1):
    """
    Computes four landscape analysis metrics for a given environment:

    1. Autocorrelation length (ACL): correlation between r(theta) and
       r(theta + delta) as a function of ||delta||. The decay rate
       characterizes how quickly the landscape 'forgets' local structure.

    2. Fitness-Distance Correlation (FDC): correlation between
       r(theta) and -||theta - theta_best|| (negative distance to best).
       High FDC = funnel landscape (easy). Low/negative FDC = rugged (hard).

    3. Local smoothness (LS): mean |r(theta + eps_i) - r(theta)| / ||eps_i||
       over random local perturbations. Low = smooth. High = rugged.

    4. Basin size (BS): fraction of random starting points from which
       greedy local search reaches within 90% of the best-known reward.
    """

    rng = np.random.RandomState(seed)
    log.info(f"E_RUG: collecting {n_samples} samples for {env_name}...")
    thetas, rewards = collect_random_samples(env_name, param_dim, n_samples,
                                              n_episodes, seed)

    # ── 1. Autocorrelation length ────────────────────────────────────────────
    # Compute r(theta_i) for random walk: theta_{t+1} = theta_t + eps * z
    walk_len = min(100, n_samples)
    walk_thetas = [rng.randn(param_dim)]
    for _ in range(walk_len - 1):
        walk_thetas.append(walk_thetas[-1] + rng.randn(param_dim) * eps)

    walk_rewards = [rollout_policy(env_name, th, n_episodes, seed + 10000 + i)
                    for i, th in enumerate(walk_thetas)]
    walk_rewards = np.array(walk_rewards)

    # Autocorrelation at lag k
    acf = []
    mean_r = walk_rewards.mean()
    var_r = walk_rewards.var() + 1e-8
    for lag in range(1, min(20, walk_len // 2)):
        corr = float(np.mean((walk_rewards[:-lag] - mean_r) *
                              (walk_rewards[lag:] - mean_r)) / var_r)
        acf.append(corr)

    # ACL = lag at which ACF first drops below 1/e
    acl = next((k + 1 for k, c in enumerate(acf) if c < 1 / np.e),
               len(acf))

    # ── 2. Fitness-Distance Correlation ─────────────────────────────────────
    best_idx = int(np.argmax(rewards))
    best_theta = thetas[best_idx]
    distances = np.linalg.norm(thetas - best_theta, axis=1)
    fdc = float(np.corrcoef(rewards, -distances)[0, 1])

    # ── 3. Local smoothness ──────────────────────────────────────────────────
    n_smooth = min(50, n_samples)
    smoothness_vals = []
    for i in range(n_smooth):
        theta = thetas[i]
        r0 = rewards[i]
        for _ in range(3):  # 3 random perturbations per point
            delta = rng.randn(param_dim) * eps
            r_perturbed = rollout_policy(env_name, theta + delta,
                                          n_episodes, seed + 20000 + i)
            grad_approx = abs(r_perturbed - r0) / (np.linalg.norm(delta) + 1e-8)
            smoothness_vals.append(grad_approx)
    local_smoothness = float(np.mean(smoothness_vals))

    # ── 4. Basin size ────────────────────────────────────────────────────────
    r_best = float(rewards.max())
    r_threshold = 0.9 * r_best if r_best > 0 else 1.1 * r_best

    # Run greedy local search from n_basin random starts
    n_basin = min(30, n_samples // 10)
    basin_successes = 0
    for b in range(n_basin):
        theta = rng.randn(param_dim)
        r = rollout_policy(env_name, theta, n_episodes, seed + 30000 + b)
        # Greedy: accept if better, up to 20 steps
        for step in range(20):
            candidate = theta + rng.randn(param_dim) * 0.3
            r_cand = rollout_policy(env_name, candidate, n_episodes,
                                     seed + 30000 + b * 100 + step)
            if r_cand > r:
                theta, r = candidate, r_cand
            if r >= r_threshold:
                basin_successes += 1
                break
    basin_size = float(basin_successes / n_basin)

    result = {
        "autocorrelation_length": acl,
        "acf_values": acf[:10],
        "fitness_distance_correlation": fdc,
        "local_smoothness": local_smoothness,
        "basin_size": basin_size,
        "best_reward_found": float(r_best),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "ruggedness_summary": {
            "acl": acl,
            "fdc": fdc,
            "smoothness": local_smoothness,
            "basin": basin_size,
        }
    }
    log.info(f"E_RUG {env_name}: ACL={acl} FDC={fdc:.3f} "
             f"smoothness={local_smoothness:.3f} basin={basin_size:.3f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# E_MEM: History Dependence across environments
# ─────────────────────────────────────────────────────────────────────────────

def run_history_dependence(ckpt_path, env_name, param_dim,
                            n_seeds=5, max_iterations=200,
                            n_eval_episodes=5, device="cpu"):
    """
    Extended E8 with more conditions, designed to measure history dependence
    as a single scalar that can be correlated with landscape ruggedness.

    Conditions:
      full:        all history (control)
      best_only:   keep only the single best-seen (theta, reward) pair
      recent_k:    keep only the most recent k=8 steps
      top_k:       keep only the top-k highest-reward steps
      oldest_k:    keep only the oldest k=8 steps (early exploration)
      random_k:    keep a random subset of k=8 steps

    History Dependence metric:
      HD = (R_full - R_best_only) / |R_full|

    HD > 0: trajectory provides information beyond the best point
    HD ≈ 0: best point alone is sufficient (rugged landscape)
    HD < 0: best-only actually outperforms full history (overcrowded context)
    """
    import torch

    K = 8  # subset size for recent/top/oldest/random conditions
    conditions = ["full", "best_only", f"recent_{K}", f"top_{K}",
                  f"oldest_{K}", f"random_{K}"]
    results = {}

    for condition in conditions:
        seed_bests = []
        for seed in range(n_seeds):
            agent = load_agent(ckpt_path, param_dim, device)
            rng = np.random.RandomState(seed)
            thetas, rewards = [], []

            n_warmup = max(16, param_dim * 2)
            for i in range(n_warmup):
                theta = rng.randn(param_dim)
                r = rollout_policy(env_name, theta, n_eval_episodes, seed + i)
                agent.add_to_history(theta, r)
                thetas.append(theta)
                rewards.append(r)

            for t in range(max_iterations - n_warmup):
                theta = agent.propose()
                r = rollout_policy(env_name, theta, n_eval_episodes,
                                    seed + n_warmup + t)
                thetas.append(theta)
                rewards.append(r)

                # Apply history modification AFTER standard add
                agent.add_to_history(theta, r)

                # Now cull the history according to the condition
                if condition == "best_only":
                    best_idx = int(np.argmax(agent.history.rewards))
                    agent.history.params = [agent.history.params[best_idx]]
                    agent.history.rewards = [agent.history.rewards[best_idx]]

                elif condition == f"recent_{K}" and len(agent.history) > K:
                    agent.history.params = agent.history.params[-K:]
                    agent.history.rewards = agent.history.rewards[-K:]

                elif condition == f"top_{K}" and len(agent.history) > K:
                    top_idx = np.argsort(agent.history.rewards)[-K:]
                    agent.history.params = [agent.history.params[i] for i in top_idx]
                    agent.history.rewards = [agent.history.rewards[i] for i in top_idx]

                elif condition == f"oldest_{K}" and len(agent.history) > K:
                    agent.history.params = agent.history.params[:K]
                    agent.history.rewards = agent.history.rewards[:K]

                elif condition == f"random_{K}" and len(agent.history) > K:
                    idx = rng.choice(len(agent.history), K, replace=False)
                    agent.history.params = [agent.history.params[i] for i in idx]
                    agent.history.rewards = [agent.history.rewards[i] for i in idx]

            seed_bests.append(float(max(rewards)))

        results[condition] = {
            "mean_best": float(np.mean(seed_bests)),
            "std_best": float(np.std(seed_bests)),
            "all_bests": seed_bests,
        }
        log.info(f"E_MEM {env_name} {condition}: "
                 f"mean_best={np.mean(seed_bests):.2f}±{np.std(seed_bests):.2f}")

    # Compute History Dependence scalar
    r_full = results["full"]["mean_best"]
    r_best_only = results["best_only"]["mean_best"]
    hd = float((r_full - r_best_only) / (abs(r_full) + 1e-6))

    # Secondary metrics
    r_recent = results[f"recent_{K}"]["mean_best"]
    r_top = results[f"top_{K}"]["mean_best"]
    r_oldest = results[f"oldest_{K}"]["mean_best"]

    results["history_dependence"] = hd
    results["recency_bias"] = float(
        (r_recent - r_oldest) / (abs(r_full) + 1e-6))
    results["quality_bias"] = float(
        (r_top - r_oldest) / (abs(r_full) + 1e-6))
    results["interpretation"] = {
        "history_dependence": (
            f"HD={hd:.3f}: "
            f"{'full trajectory matters' if hd > 0.05 else 'best point dominates' if hd < -0.05 else 'trajectory and best point equivalent'}"
        ),
        "recency_bias": (
            f"{'Recent history more useful than oldest' if results['recency_bias'] > 0.05 else 'Old history as useful as recent'}"
        ),
        "quality_bias": (
            f"{'Top-K rewards more useful than oldest (quality > age)' if results['quality_bias'] > 0.05 else 'Age matters more than quality'}"
        ),
    }
    log.info(f"E_MEM {env_name}: HD={hd:.3f} recency={results['recency_bias']:.3f} "
             f"quality={results['quality_bias']:.3f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Cross-environment synthesis: correlate HD with ruggedness
# ─────────────────────────────────────────────────────────────────────────────

def synthesize_hd_ruggedness(hd_results, ruggedness_results):
    """
    Given per-environment HD and ruggedness metrics, compute Pearson
    correlations between HD and each ruggedness metric.

    This is the core analysis for the paper claim:
    "History dependence varies systematically with landscape ruggedness."

    hd_results: {env_name: {"history_dependence": float, ...}}
    ruggedness_results: {env_name: {"fitness_distance_correlation": float,
                                     "autocorrelation_length": int,
                                     "local_smoothness": float,
                                     "basin_size": float}}
    """
    envs = [e for e in hd_results if e in ruggedness_results]
    if len(envs) < 3:
        return {"error": "Need at least 3 environments for correlation analysis",
                "n_envs": len(envs)}

    hd_vals = np.array([hd_results[e]["history_dependence"] for e in envs])
    fdc_vals = np.array([ruggedness_results[e]["fitness_distance_correlation"]
                          for e in envs])
    acl_vals = np.array([ruggedness_results[e]["autocorrelation_length"]
                          for e in envs])
    smooth_vals = np.array([ruggedness_results[e]["local_smoothness"]
                             for e in envs])
    basin_vals = np.array([ruggedness_results[e]["basin_size"] for e in envs])

    def safe_corr(x, y):
        if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    result = {
        "environments": envs,
        "history_dependence_values": hd_vals.tolist(),
        "correlations": {
            "HD_vs_FDC": safe_corr(hd_vals, fdc_vals),
            "HD_vs_ACL": safe_corr(hd_vals, acl_vals),
            "HD_vs_smoothness": safe_corr(hd_vals, -smooth_vals),  # neg: smoother = lower smoothness metric
            "HD_vs_basin_size": safe_corr(hd_vals, basin_vals),
        },
        "per_env": {
            e: {
                "history_dependence": hd_results[e]["history_dependence"],
                "fdc": float(fdc_vals[i]),
                "acl": float(acl_vals[i]),
                "smoothness": float(smooth_vals[i]),
                "basin_size": float(basin_vals[i]),
            }
            for i, e in enumerate(envs)
        },
        "interpretation": (
            "HD_vs_FDC > 0: smooth funnel landscapes → history matters. "
            "HD_vs_ACL > 0: long autocorrelation → sequential history exploited. "
            "HD_vs_basin_size > 0: large basins → trajectory useful for navigation. "
            "All three positive would strongly support the landscape-memory claim."
        )
    }

    log.info(f"\nCross-environment synthesis ({len(envs)} envs):")
    for k, v in result["correlations"].items():
        log.info(f"  {k} = {v:.3f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

# Default environment suite spanning different landscape characteristics
ENV_SUITE = [
    # (env_name, param_dim, ckpt_key)
    ("CartPole-v1",             8,  "cartpole"),
    ("Swimmer-v4",              16, "swimmer"),
    ("MountainCarContinuous-v0", 3, "mountaincar"),
    ("InvertedPendulum-v4",     5,  "cartpole"),   # use cartpole ckpt via projection
    ("FrozenLake-v1",           16, "cartpole"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_cartpole",
                   default="checkpoints/jepa_pretrained/jepa_pretrained.pt")
    p.add_argument("--ckpt_swimmer",
                   default="checkpoints/jepa_pretrained_swimmer/jepa_pretrained.pt")
    p.add_argument("--ckpt_mountaincar",
                   default="results/mountaincar_ckpt/checkpoints/shuffle_none.pt")
    p.add_argument("--out_dir", default="results/landscape_memory")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--max_iterations", type=int, default=200)
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--envs", nargs="+",
                   default=["CartPole-v1", "Swimmer-v4",
                             "MountainCarContinuous-v0"])
    p.add_argument("--experiments", nargs="+", default=["all"],
                   choices=["ordinality", "ruggedness", "memory", "synthesis", "all"])
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run_all = "all" in args.experiments

    ckpt_map = {
        "CartPole-v1":              args.ckpt_cartpole,
        "Swimmer-v4":               args.ckpt_swimmer,
        "MountainCarContinuous-v0": args.ckpt_mountaincar,
        "InvertedPendulum-v4":      args.ckpt_cartpole,
        "FrozenLake-v1":            args.ckpt_cartpole,
    }
    pdim_map = {
        "CartPole-v1": 8, "Swimmer-v4": 16,
        "MountainCarContinuous-v0": 3, "InvertedPendulum-v4": 5,
        "FrozenLake-v1": 16, "Hopper-v4": 36,
    }

    hd_all, rug_all = {}, {}

    for env in args.envs:
        ckpt = ckpt_map.get(env, args.ckpt_cartpole)
        pdim = pdim_map.get(env, 8)

        if not os.path.exists(ckpt):
            log.warning(f"Checkpoint not found for {env}: {ckpt} -- skipping")
            continue

        common = dict(ckpt_path=ckpt, env_name=env, param_dim=pdim,
                      n_seeds=args.n_seeds, max_iterations=args.max_iterations,
                      n_eval_episodes=args.n_episodes, device=args.device)

        # E_ORD: Reward Ordinality
        if run_all or "ordinality" in args.experiments:
            log.info(f"\n{'='*60}\nE_ORD: Reward Ordinality — {env}\n{'='*60}")
            r = run_reward_ordinality(**common)
            with open(f"{args.out_dir}/ordinality_{env.replace('/','-')}.json","w") as f:
                json.dump(r, f, indent=2)
            log.info(f"Saved ordinality results for {env}")

        # E_RUG: Landscape Ruggedness
        if run_all or "ruggedness" in args.experiments:
            log.info(f"\n{'='*60}\nE_RUG: Landscape Ruggedness — {env}\n{'='*60}")
            r = measure_landscape_ruggedness(env, pdim,
                                              n_samples=200,
                                              n_episodes=args.n_episodes,
                                              seed=0)
            rug_all[env] = r
            with open(f"{args.out_dir}/ruggedness_{env.replace('/','-')}.json","w") as f:
                json.dump(r, f, indent=2)

        # E_MEM: History Dependence
        if run_all or "memory" in args.experiments:
            log.info(f"\n{'='*60}\nE_MEM: History Dependence — {env}\n{'='*60}")
            r = run_history_dependence(**common)
            hd_all[env] = r
            with open(f"{args.out_dir}/memory_{env.replace('/','-')}.json","w") as f:
                json.dump(r, f, indent=2)

    # Cross-environment synthesis
    if (run_all or "synthesis" in args.experiments) and hd_all and rug_all:
        log.info(f"\n{'='*60}\nSynthesis: HD vs Ruggedness\n{'='*60}")
        r = synthesize_hd_ruggedness(hd_all, rug_all)
        with open(f"{args.out_dir}/synthesis.json", "w") as f:
            json.dump(r, f, indent=2)
        log.info(f"Synthesis complete. Key correlations:")
        for k, v in r.get("correlations", {}).items():
            log.info(f"  {k} = {v:.3f}")

    log.info(f"\nAll results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
