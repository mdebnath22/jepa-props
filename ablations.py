"""
ablations.py — JEPA-specific sensitivity analyses and design-choice ablations.

These all require the existing SwarmJEPAAgent / single-trajectory JEPA agent
already deployed on the cluster (agent/jepa_agent.py, swarm_jepa_agent.py,
swarm_jepa_runner.py). This module does NOT duplicate that code — it
imports and parametrizes it.

Addresses reviewer requests:
  - "Ablate inertia/cognitive/social terms individually and vary K"
  - "How sensitive are results to ... migration frequency, and number of
    sub-populations?"
  - "What is the effect of optional online fine-tuning?"
  - "why is candidate selection based on cosine similarity ... ablate this"
  - "comparison ... to latent-space CEM/PF"

Each function returns the same standardized dict as baselines.py:
    {"rewards": [...], "best_reward": float, "wallclock": {...}}
so results can be pooled into one results table across all experiment types.
"""

import time
import logging
import numpy as np

log = logging.getLogger(__name__)

from ablations_patch import (
    propose_best_of_n_fixed,
    propose_random_decode_fixed,
    run_latent_space_cem_fixed,
)


def _import_swarm():
    """Deferred import so this module can be inspected/tested without the
    cluster-specific JEPA code present (e.g. in this sandbox)."""
    import sys, os
    sys.path.insert(0, os.getcwd())
    from swarm_jepa_runner import run_swarm_jepa
    from runner.jepa_runner import rollout_policy as jepa_rollout_policy
    return run_swarm_jepa, jepa_rollout_policy


# ─────────────────────────────────────────────────────────────────────────────
# K sweep — number of sub-populations
# ─────────────────────────────────────────────────────────────────────────────

def sweep_population_size(env_name, param_dim, seed, max_iterations,
                           k_values=(1, 2, 3, 5, 8), pretrained_ckpt="",
                           noise_std=0.5, n_candidates=20, n_eval_episodes=10,
                           device="cpu"):
    """
    Vary K (number of parallel sub-populations) holding everything else
    fixed. K=1 reduces to single-trajectory search (no migration target
    exists), directly bridging to the main single-trajectory results.
    """
    run_swarm_jepa, _ = _import_swarm()
    results = {}
    for K in k_values:
        t0 = time.perf_counter()
        r = run_swarm_jepa(
            env_name=env_name, param_dim=param_dim, seed=seed,
            max_iterations=max_iterations, device=device, K=K,
            noise_std=noise_std, n_candidates=n_candidates,
            n_eval_episodes=n_eval_episodes, pretrained_ckpt=pretrained_ckpt,
            migration_every=30, isolation_every=60, isolation_steps=6,
        )
        r["wallclock"] = {"total_s": time.perf_counter() - t0}
        results[f"K={K}"] = r
        log.info(f"K={K}: best={r['best_reward']:.1f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Migration frequency sweep
# ─────────────────────────────────────────────────────────────────────────────

def sweep_migration_frequency(env_name, param_dim, seed, max_iterations,
                               freqs=(10, 30, 60, 100, None), pretrained_ckpt="",
                               noise_std=0.5, n_candidates=20, K=3,
                               n_eval_episodes=10, device="cpu"):
    """
    freqs: list of migration_every values. None means "never migrate"
    (implemented as migration_every > max_iterations so it never fires),
    isolating migration's specific contribution.
    """
    run_swarm_jepa, _ = _import_swarm()
    results = {}
    for freq in freqs:
        effective_freq = freq if freq is not None else max_iterations + 1
        label = f"migrate_every={freq if freq is not None else 'never'}"
        t0 = time.perf_counter()
        r = run_swarm_jepa(
            env_name=env_name, param_dim=param_dim, seed=seed,
            max_iterations=max_iterations, device=device, K=K,
            noise_std=noise_std, n_candidates=n_candidates,
            n_eval_episodes=n_eval_episodes, pretrained_ckpt=pretrained_ckpt,
            migration_every=effective_freq, isolation_every=60, isolation_steps=6,
        )
        r["wallclock"] = {"total_s": time.perf_counter() - t0}
        results[label] = r
        log.info(f"{label}: best={r['best_reward']:.1f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Exploration sigma sweep
# ─────────────────────────────────────────────────────────────────────────────

def sweep_noise_std(env_name, param_dim, seed, max_iterations,
                     sigma_values=(0.1, 0.3, 0.5, 0.8, 1.2), pretrained_ckpt="",
                     n_candidates=20, K=3, n_eval_episodes=10, device="cpu"):
    run_swarm_jepa, _ = _import_swarm()
    results = {}
    for sigma in sigma_values:
        label = f"sigma={sigma}"
        t0 = time.perf_counter()
        r = run_swarm_jepa(
            env_name=env_name, param_dim=param_dim, seed=seed,
            max_iterations=max_iterations, device=device, K=K,
            noise_std=sigma, n_candidates=n_candidates,
            n_eval_episodes=n_eval_episodes, pretrained_ckpt=pretrained_ckpt,
            migration_every=30, isolation_every=60, isolation_steps=6,
        )
        r["wallclock"] = {"total_s": time.perf_counter() - t0}
        results[label] = r
        log.info(f"{label}: best={r['best_reward']:.1f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Weight-schedule ablation: fixed weights vs. the phased schedule
# ─────────────────────────────────────────────────────────────────────────────

def compare_weight_schedules(env_name, param_dim, seed, max_iterations,
                              pretrained_ckpt="", noise_std=0.5,
                              n_candidates=20, K=3, n_eval_episodes=10,
                              device="cpu"):
    """
    Compares the phased (0.5,0.3,0.2)->(0.3,0.3,0.4)->(0.2,0.2,0.6) schedule
    (already implemented in swarm_jepa_agent.pso_weights) against a fixed
    equal-weight schedule (1/3, 1/3, 1/3) throughout, via monkeypatching.
    """
    import swarm_jepa_agent as sja

    original_pso_weights = sja.pso_weights

    def fixed_weights(t, total_T):
        return (1/3, 1/3, 1/3)

    run_swarm_jepa, _ = _import_swarm()
    results = {}

    t0 = time.perf_counter()
    r_phased = run_swarm_jepa(
        env_name=env_name, param_dim=param_dim, seed=seed,
        max_iterations=max_iterations, device=device, K=K,
        noise_std=noise_std, n_candidates=n_candidates,
        n_eval_episodes=n_eval_episodes, pretrained_ckpt=pretrained_ckpt,
        migration_every=30, isolation_every=60, isolation_steps=6,
    )
    r_phased["wallclock"] = {"total_s": time.perf_counter() - t0}
    results["phased_schedule"] = r_phased

    sja.pso_weights = fixed_weights
    try:
        t0 = time.perf_counter()
        r_fixed = run_swarm_jepa(
            env_name=env_name, param_dim=param_dim, seed=seed,
            max_iterations=max_iterations, device=device, K=K,
            noise_std=noise_std, n_candidates=n_candidates,
            n_eval_episodes=n_eval_episodes, pretrained_ckpt=pretrained_ckpt,
            migration_every=30, isolation_every=60, isolation_steps=6,
        )
        r_fixed["wallclock"] = {"total_s": time.perf_counter() - t0}
        results["fixed_equal_weights"] = r_fixed
    finally:
        sja.pso_weights = original_pso_weights  # always restore

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Online fine-tuning ablation
# ─────────────────────────────────────────────────────────────────────────────

def compare_online_finetuning(env_name, param_dim, seed, max_iterations,
                               pretrained_ckpt="", noise_std=0.5,
                               n_candidates=20, K=3, n_eval_episodes=10,
                               device="cpu"):
    run_swarm_jepa, _ = _import_swarm()
    results = {}
    for finetune in (True, False):
        label = f"online_finetune={finetune}"
        t0 = time.perf_counter()
        r = run_swarm_jepa(
            env_name=env_name, param_dim=param_dim, seed=seed,
            max_iterations=max_iterations, device=device, K=K,
            noise_std=noise_std, n_candidates=n_candidates,
            n_eval_episodes=n_eval_episodes, pretrained_ckpt=pretrained_ckpt,
            migration_every=30, isolation_every=60, isolation_steps=6,
            online_finetune=finetune,
        )
        r["wallclock"] = {"total_s": time.perf_counter() - t0}
        results[label] = r
        log.info(f"{label}: best={r['best_reward']:.1f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Candidate-selection-rule ablation
# ─────────────────────────────────────────────────────────────────────────────

def compare_selection_rules(env_name, param_dim, seed, max_iterations,
                             pretrained_ckpt="", noise_std=0.5,
                             n_candidates=20, n_eval_episodes=10, device="cpu"):
    """
    Reviewer asks: why cosine-similarity-to-predicted-continuation rather
    than a simpler scheme? This ablates THREE candidate-selection rules in
    single-trajectory JEPA search (agent.jepa_agent.JEPAAgent.propose):

      1. "cosine"      — current default: pick candidate whose predicted
                          latent has max cosine similarity to context's own
                          predicted continuation.
      2. "best_of_n"    — decode n candidates, score each by predicted-
                          latent norm as a simple confidence proxy (the
                          predictor has no explicit reward head), pick max.
      3. "random"       — decode n candidates, pick uniformly at random
                          (sanity floor — if this matches "cosine", the
                          selection rule is doing no work at all).
    """
    import sys, os
    sys.path.insert(0, os.getcwd())
    from agent.jepa_agent import JEPAAgent
    from runner.jepa_runner import rollout_policy
    import torch

    results = {}

    for rule in ("cosine", "best_of_n", "random"):
        np.random.seed(seed)
        torch.manual_seed(seed)

        agent = JEPAAgent(param_dim=param_dim, device=device)
        if pretrained_ckpt:
            try:
                ckpt = torch.load(pretrained_ckpt, map_location=device,
                                  weights_only=False)
                sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
                agent.load_state_dict(sd, strict=False)
            except Exception as e:
                log.warning(f"checkpoint load failed: {e}")

        rng = np.random.RandomState(seed)
        rewards = []
        best = -np.inf
        t0 = time.perf_counter()

        n_warmup = max(10, param_dim * 2)
        for i in range(n_warmup):
            theta = rng.randn(param_dim)
            r = rollout_policy(env_name, theta, n_eval_episodes, seed + i)
            agent.add_to_history(theta, r)
            rewards.append(r)
            best = max(best, r)

        for t in range(max_iterations - n_warmup):
            if rule == "cosine":
                theta = agent.propose()  # default behavior, unmodified
            elif rule == "best_of_n":
                theta = propose_best_of_n_fixed(agent, n_candidates, rng)
            else:  # "random"
                theta = propose_random_decode_fixed(agent, n_candidates, rng)

            r = rollout_policy(env_name, theta, n_eval_episodes,
                               seed + n_warmup + t + 100)
            agent.add_to_history(theta, r)
            rewards.append(r)
            best = max(best, r)

        results[rule] = {
            "rewards": rewards[:max_iterations],
            "best_reward": float(best),
            "wallclock": {"total_s": time.perf_counter() - t0},
        }
        log.info(f"selection_rule={rule}: best={best:.1f}")

    return results


def _propose_best_of_n(agent, n_candidates, rng):
    """Decode n candidates, pick by predicted-latent-norm proxy (higher norm
    interpreted as more confident/extreme prediction -- a simple, defensible
    alternative to cosine-similarity selection without requiring a reward head)."""
    import torch
    s_x, mean, std = agent._encode_history()
    desired_r = (agent.history_rewards[-1] if agent.history_rewards else 0.0) * 1.05
    desired_r_t = torch.tensor([[desired_r]], dtype=torch.float32, device=agent.device)

    best_theta, best_score = None, -np.inf
    for _ in range(n_candidates):
        noise = torch.randn_like(s_x) * agent.noise_std
        s_y_hat = agent.predictor(s_x + noise, desired_r_t)
        theta_norm = agent.decoder(s_y_hat).squeeze(0).detach().cpu().numpy()
        theta = theta_norm * std + mean
        score = float(torch.norm(s_y_hat).item())
        if score > best_score:
            best_score, best_theta = score, theta
    return best_theta if best_theta is not None else rng.randn(agent.param_dim)


def _propose_random_decode(agent, n_candidates, rng):
    """Decode n candidates, pick uniformly at random — sanity floor."""
    import torch
    s_x, mean, std = agent._encode_history()
    desired_r = (agent.history_rewards[-1] if agent.history_rewards else 0.0) * 1.05
    desired_r_t = torch.tensor([[desired_r]], dtype=torch.float32, device=agent.device)

    candidates = []
    for _ in range(n_candidates):
        noise = torch.randn_like(s_x) * agent.noise_std
        s_y_hat = agent.predictor(s_x + noise, desired_r_t)
        theta_norm = agent.decoder(s_y_hat).squeeze(0).detach().cpu().numpy()
        candidates.append(theta_norm * std + mean)
    idx = rng.randint(0, len(candidates))
    return candidates[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Latent-space CEM (operates on JEPA latents, not raw theta) — reviewer's
# request to compare against world-model latent-planning methods (e.g. CEM
# inside a learned latent space, as in PlaNet/Dreamer-style planners)
# ─────────────────────────────────────────────────────────────────────────────

def run_latent_space_cem(env_name, param_dim, seed, max_iterations,
                          pretrained_ckpt="", n_eval_episodes=10,
                          popsize=20, elite_frac=0.2, device="cpu"):
    """Delegates to the corrected implementation in ablations_patch.py."""
    return run_latent_space_cem_fixed(
        env_name, param_dim, seed, max_iterations,
        pretrained_ckpt=pretrained_ckpt, n_eval_episodes=n_eval_episodes,
        popsize=popsize, elite_frac=elite_frac, device=device,
    )


def run_with_repeated_evaluation(run_fn, env_name, param_dim, seed,
                                  max_iterations, n_repeats=5, **kwargs):
    """
    Wraps any run_X function from baselines.py or this module so that EVERY
    proposed theta is evaluated with n_repeats independent rollouts and
    averaged, reducing per-candidate return noise. This directly addresses
    the reviewer's concern about "random best > CEM best" arising from
    evaluation stochasticity rather than a real result.

    Implementation note: this monkeypatches envs.rollout_policy for the
    duration of the call, since every baseline/ablation function calls
    through that single function. Thread-unsafe; intended for sequential
    experiment scripts only.
    """
    import envs as envs_module

    original_rollout = envs_module.rollout_policy

    def repeated_rollout(env_name_inner, theta, n_episodes, seed_inner):
        from envs import rollout_policy_repeated
        res = rollout_policy_repeated(env_name_inner, theta, n_episodes,
                                       n_repeats=n_repeats, base_seed=seed_inner)
        return res["mean"]

    envs_module.rollout_policy = repeated_rollout
    try:
        result = run_fn(env_name, param_dim, seed, max_iterations, **kwargs)
    finally:
        envs_module.rollout_policy = original_rollout  # always restore

    return result
