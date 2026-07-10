"""
Hybrid LLM + JEPA Runner
==========================
Analog of props-llm/runner/llm_num_optim_semantics_runner.py (ProPS+)

Architecture:
    LLM  →  task embedding (once per task)  →  semantic conditioning
    JEPA →  numerical policy proposal (every iteration)

The LLM's role is REDUCED to:
  1. Encoding the task description into a dense vector (one forward pass)
  2. Optionally, providing constraint feedback at the end (optional)

This avoids the expensive LLM call at EVERY iteration that ProPS requires,
while keeping semantic grounding.

Requires:
    - A sentence-transformer or similar for task embeddings (or a frozen LLM encoder)
    - The JEPAAgent with semantic_dim > 0
"""

import os
import time
import logging
import numpy as np
import torch
from typing import Optional

from agent.jepa_agent import JEPAAgent
from runner.jepa_runner import rollout_policy, _online_finetune

log = logging.getLogger(__name__)

# sentence_transformers imported lazily inside get_task_embedding() only —
# avoids pulling in datasets/pyarrow at module load time (pyarrow compat issue).
HAS_SENTENCE_TRANSFORMERS = None  # checked on first use


# ─────────────────────────────────────────────────────────────────────────────
# Task description templates (mirrors ProPS+ prompts but distilled to strings)
# ─────────────────────────────────────────────────────────────────────────────

TASK_DESCRIPTIONS = {
    "CartPole-v1": (
        "CartPole: balance a pole on a cart. "
        "State: cart position (-4.8 to 4.8), cart velocity, pole angle (-0.418 rad), pole angular velocity. "
        "Actions: push left or right. Reward: +1 per timestep pole is upright. "
        "Goal: maximize time balanced. Large angular velocity changes destabilize the pole."
    ),
    "MountainCarContinuous-v0": (
        "MountainCar Continuous: drive an underpowered car up a mountain. "
        "State: position (-1.2 to 0.6), velocity (-0.07 to 0.07). "
        "Action: continuous force (-1 to 1). Reward: +100 at goal, -0.1 * action^2 penalty. "
        "Strategy: build momentum by oscillating back and forth before ascending."
    ),
    "Swimmer-v4": (
        "Swimmer: control a 2-link snake robot in a viscous fluid. "
        "State: angles and angular velocities of joints. "
        "Action: torques on joints. Reward: forward velocity minus control cost. "
        "Goal: maximize forward swimming speed."
    ),
    "HalfCheetah-v4": (
        "HalfCheetah: control a 2D cheetah robot. "
        "Reward: forward velocity minus control cost. "
        "Goal: run as fast as possible forward."
    ),
    "Hopper-v4": (
        "Hopper: control a one-legged robot. "
        "Reward: forward velocity, alive bonus, minus control cost. "
        "Goal: hop forward without falling."
    ),
    "InvertedPendulum-v4": (
        "InvertedPendulum: balance a pole on a cart (MuJoCo). "
        "Reward: +1 per timestep. Goal: keep pole upright."
    ),
    "Walker2d-v4": (
        "Walker2d: bipedal walker robot. "
        "Reward: forward speed plus alive bonus minus control cost. "
        "Goal: walk forward without falling."
    ),
    "FrozenLake-v1": (
        "FrozenLake: navigate frozen lake avoiding holes. "
        "State: discrete grid position. Actions: left, down, right, up. "
        "Reward: +1 at goal. Slippery surface causes stochastic transitions."
    ),
    "Reacher-v4": (
        "Reacher: move robotic arm to target position. "
        "State: joint angles, velocities, target position. "
        "Reward: negative distance to target. Goal: minimize distance."
    ),
}


def get_task_embedding(
    task_description: str,
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
) -> torch.Tensor:
    """
    Encode a task description string into a dense embedding vector.
    Returns: (1, embed_dim) tensor
    Imports sentence_transformers lazily to avoid pyarrow version conflicts.
    """
    global HAS_SENTENCE_TRANSFORMERS
    if HAS_SENTENCE_TRANSFORMERS is None:
        try:
            from sentence_transformers import SentenceTransformer as _ST
            HAS_SENTENCE_TRANSFORMERS = True
        except Exception:
            HAS_SENTENCE_TRANSFORMERS = False
            log.warning("sentence-transformers unavailable. Using zero embedding.")

    if not HAS_SENTENCE_TRANSFORMERS:
        return torch.zeros(1, 384)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode([task_description], convert_to_tensor=True)
    return emb.unsqueeze(0) if emb.dim() == 1 else emb


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_training_loop(
    # Required
    env_name: str,
    param_dim: int,
    # Semantic
    task_description: str = None,      # custom description, else uses TASK_DESCRIPTIONS
    semantic_encoder: str = "all-MiniLM-L6-v2",
    semantic_dim: int = 384,           # must match sentence-transformer output
    # JEPA architecture
    d_model: int = 128,
    nhead: int = 4,
    num_enc_layers: int = 3,
    num_pred_layers: int = 3,
    dim_ff: int = 256,
    ema_decay: float = 0.996,
    desired_pct: float = 80.0,
    noise_std: float = 0.05,
    # Training
    max_iterations: int = 400,
    n_eval_episodes: int = 5,
    n_candidates: int = 5,
    seed: int = 0,
    # Pretrained weights
    pretrained_ckpt: str = None,
    online_finetune: bool = True,
    online_lr: float = 1e-4,
    online_finetune_every: int = 10,
    online_finetune_steps: int = 5,
    # Logging
    log_dir: str = "results",
    run_name: str = None,
    device: str = "cpu",
    task: str = "cont_state_jepa_semantics",
    **kwargs,
):
    """
    Run the LLM+JEPA hybrid policy search.

    The LLM (sentence-transformer) runs ONCE at the start to encode the task.
    JEPA runs at every iteration.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(log_dir, exist_ok=True)
    run_name = run_name or f"jepa_hybrid_{env_name}_{seed}"
    log_path = os.path.join(log_dir, f"{run_name}.csv")

    # ── Get task embedding (LLM call, once) ────────────────────────────────
    description = task_description or TASK_DESCRIPTIONS.get(env_name, env_name)
    log.info(f"Encoding task: {description[:80]}...")
    semantic_emb = get_task_embedding(description, model_name=semantic_encoder, device=device)
    semantic_emb = semantic_emb.to(device)
    log.info(f"Semantic embedding shape: {semantic_emb.shape}")

    # ── Build agent with semantic conditioning ─────────────────────────────
    agent = JEPAAgent(
        param_dim=param_dim,
        d_model=d_model,
        nhead=nhead,
        num_enc_layers=num_enc_layers,
        num_pred_layers=num_pred_layers,
        dim_ff=dim_ff,
        ema_decay=ema_decay,
        desired_pct=desired_pct,
        noise_std=noise_std,
        semantic_dim=semantic_dim,
        device=device,
    )

    if pretrained_ckpt and os.path.exists(pretrained_ckpt):
        log.info(f"Loading pretrained JEPA from {pretrained_ckpt}")
        agent.load_state_dict(
            torch.load(pretrained_ckpt, map_location=device), strict=False
        )

    online_params = [p for n, p in agent.named_parameters()
                     if "ema_head" not in n and p.requires_grad]
    online_optimizer = torch.optim.AdamW(online_params, lr=online_lr)

    # ── Warm-start ──────────────────────────────────────────────────────────
    for i in range(5):
        theta = np.random.randn(param_dim)
        r = rollout_policy(env_name, theta, n_eval_episodes, seed + i)
        agent.add_to_history(theta, r)

    # ── Policy search loop ─────────────────────────────────────────────────
    all_rewards, all_params = [], []
    best_reward, best_params = -np.inf, None
    t_start = time.time()

    with open(log_path, "w") as f:
        f.write("iteration,reward,best_reward,elapsed\n")

    for t in range(max_iterations):
        # JEPA proposes θ conditioned on history + semantic embedding
        theta = agent.propose(semantic_emb=semantic_emb, n_candidates=n_candidates)
        reward = rollout_policy(env_name, theta, n_eval_episodes, seed + 100 + t)

        agent.add_to_history(theta, reward)
        all_rewards.append(reward)
        all_params.append(theta)

        if reward > best_reward:
            best_reward = reward
            best_params = theta.copy()

        elapsed = time.time() - t_start
        log.info(f"[Hybrid {t+1}/{max_iterations}] r={reward:.2f} best={best_reward:.2f}")

        with open(log_path, "a") as f:
            f.write(f"{t+1},{reward:.4f},{best_reward:.4f},{elapsed:.2f}\n")

        if online_finetune and len(agent.history) >= 10 and (t + 1) % online_finetune_every == 0:
            _online_finetune(agent, online_optimizer, online_finetune_steps, device)

    return {
        "rewards": all_rewards,
        "params": all_params,
        "best_reward": best_reward,
        "best_params": best_params,
        "log_path": log_path,
    }
