"""
Swarm-JEPA Runner
=================
Implements the full K-island Swarm-JEPA policy search loop.

Designed to be called from compare_methods.py exactly like run_jepa(),
returning the same {"rewards", "best_reward"} dict — just swap the function.

The key loop structure:
  - Round-robin across K islands: each island proposes one theta per round
  - After each round: update global best, check migration / isolation triggers
  - Online fine-tune each island's JEPA agent using only its own history

This directly mirrors SMS Algorithm 1 from the MorphoEvo paper, but in
continuous parameter space with JEPA as the mutation oracle instead of an LLM.
"""

import os
import logging
import numpy as np
import yaml
import torch

log = logging.getLogger(__name__)


def run_swarm_jepa(
    env_name: str,
    param_dim: int,
    seed: int,
    max_iterations: int,
    device: str = "cpu",
    K: int = 3,              # number of islands
    noise_std: float = 0.5,
    n_candidates: int = 20,
    n_eval_episodes: int = 10,
    desired_pct: float = 90.0,
    pretrained_ckpt: str = "",
    online_finetune: bool = True,
    online_lr: float = 1e-4,
    online_finetune_every: int = 5,
    online_finetune_steps: int = 10,
    migration_every: int = 30,
    isolation_every: int = 60,
    isolation_steps: int = 6,
    log_dir: str = "results/swarm_jepa",
    run_name: str = "swarm_jepa",
    # JEPA architecture (must match the pretrained checkpoint)
    d_model: int = 128,
    nhead: int = 4,
    num_enc_layers: int = 3,
    num_pred_layers: int = 3,
    dim_ff: int = 256,
    ema_decay: float = 0.996,
    max_reward: float = 500.0,
) -> dict:
    """
    Run K-island Swarm-JEPA policy search.

    Returns: {"rewards": [...], "best_reward": float}
    The rewards list is length max_iterations (same as single-island runner).
    Each entry is the reward of the best proposal across ALL islands that round.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from agent.jepa_agent import JEPAAgent
    from envs import rollout_policy
    from swarm_jepa_agent import SwarmJEPAAgent
    from karma_jepa_v2 import patch_swarm_jepa_agent as _patch_kj
    _patch_kj(SwarmJEPAAgent, enable_k2=True, enable_m=False)  # K2 only

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Architecture kwargs forwarded to each island's JEPAAgent
    agent_kwargs = dict(
        param_dim=param_dim,
        d_model=d_model,
        nhead=nhead,
        num_enc_layers=num_enc_layers,
        num_pred_layers=num_pred_layers,
        dim_ff=dim_ff,
        ema_decay=ema_decay,
        desired_pct=desired_pct,
        noise_std=noise_std,
        # device passed separately to SwarmJEPAAgent, not duplicated here
    )

    swarm = SwarmJEPAAgent(
        jepa_agent_cls=JEPAAgent,
        agent_kwargs=agent_kwargs,
        pretrained_ckpt=pretrained_ckpt,
        device=device,
        K=K,
        param_dim=param_dim,
        noise_std=noise_std,
        n_candidates=n_candidates,
        migration_every=migration_every,
        isolation_every=isolation_every,
        isolation_steps=isolation_steps,
        desired_pct=desired_pct,
        max_reward=max_reward,
    )

    all_rewards = []
    global_best = -np.inf
    t = 0  # global iteration counter

    # ── Warm-start: n_warmup rounds of random rollouts across all islands ─────
    n_warmup_per_island = max(5, param_dim)
    log.info(f"Swarm warm-start: {n_warmup_per_island} random rollouts per island "
             f"({K} islands)")

    for i in range(n_warmup_per_island):
        for k in range(K):
            theta = np.random.randn(param_dim)
            r = rollout_policy(env_name, theta, n_eval_episodes, seed + k * 1000 + i)
            swarm.add_to_island(k, theta, r)

    # ── Main loop ─────────────────────────────────────────────────────────────
    log.info(f"Starting Swarm-JEPA main loop: {max_iterations} iterations, "
             f"K={K} islands, env={env_name}")

    finetune_counters = [0] * K

    while t < max_iterations:
        # Each "round" = one proposal from each island
        # But we report one reward per global iteration to match compare_methods
        round_rewards = []

        for k in range(K):
            if t >= max_iterations:
                break

            # Propose and evaluate
            theta = swarm.propose_for_island(k, t, max_iterations)
            r = rollout_policy(env_name, theta, n_eval_episodes,
                               seed + k * 1000 + t + 100)
            swarm.add_to_island(k, theta, r)

            round_rewards.append(r)

            # Online fine-tune this island's agent
            finetune_counters[k] += 1
            if (online_finetune and
                    finetune_counters[k] % online_finetune_every == 0):
                swarm.online_finetune_island(k, lr=online_lr,
                                              steps=online_finetune_steps)

            # Record best reward across this round
            _, gb_reward = swarm.global_best()
            all_rewards.append(max(round_rewards))
            global_best = gb_reward

            t += 1

        # Migration check (every migration_every global iters)
        swarm.maybe_migrate(t)

        # Isolation check
        swarm.maybe_isolate(t)

        if t % 50 == 0 and t > 0:
            log.info(f"[Swarm t={t}/{max_iterations}] "
                     f"global_best={global_best:.1f} | "
                     f"{swarm.island_summary()}")

    # Trim/pad to exactly max_iterations
    all_rewards = all_rewards[:max_iterations]
    while len(all_rewards) < max_iterations:
        all_rewards.append(all_rewards[-1] if all_rewards else 0.0)

    _, final_best = swarm.global_best()
    log.info(f"Swarm-JEPA done. best={final_best:.1f}")

    os.makedirs(log_dir, exist_ok=True)
    import csv
    csv_path = os.path.join(log_dir, f"{run_name}_seed{seed}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "reward"])
        for i, r in enumerate(all_rewards):
            writer.writerow([i, r])

    return {"rewards": all_rewards, "best_reward": float(final_best)}
