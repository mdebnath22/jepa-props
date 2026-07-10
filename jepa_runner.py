"""
JEPA Numerical Policy Search Runner
=====================================
Direct analog of props-llm/runner/llm_num_optim_runner.py

ProPS loop:
    for t in range(max_iter):
        theta_t = LLM(history)
        r_t = env.rollout(theta_t)
        history.append((theta_t, r_t))

JEPA loop (identical interface):
    for t in range(max_iter):
        theta_t = JEPAAgent.propose(history)   ← replaces LLM call
        r_t = env.rollout(theta_t)
        history.append((theta_t, r_t))
"""

import os
import time
import logging
import numpy as np
import torch
import gymnasium as gym
from typing import List, Dict, Optional

from agent.jepa_agent import JEPAAgent

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Policy types (matching ProPS)
# ─────────────────────────────────────────────────────────────────────────────

def linear_policy_act(params: np.ndarray, obs: np.ndarray, action_space) -> np.ndarray:
    """Linear policy: action = argmax(W @ obs)  or  W @ obs (continuous)."""
    if hasattr(action_space, "n"):
        act_dim = action_space.n
        obs_dim = len(obs)
        W = params.reshape(act_dim, obs_dim)
        logits = W @ obs
        return int(np.argmax(logits))
    else:
        act_dim = action_space.shape[0]
        obs_dim = len(obs)
        W = params.reshape(act_dim, obs_dim)
        raw = W @ obs
        return np.clip(raw, action_space.low, action_space.high)


def rollout_policy(
    env_name: str,
    params: np.ndarray,
    n_eval_episodes: int = 5,
    seed: int = 0,
    max_steps: int = 1000,
    render: bool = False,
) -> float:
    """Evaluate policy parameters in the environment and return mean reward."""
    env = gym.make(env_name, render_mode="human" if render else None)
    total_reward = 0.0
    for ep in range(n_eval_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        steps = 0
        while not done and steps < max_steps:
            action = linear_policy_act(params, obs, env.action_space)
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += r
            steps += 1
        total_reward += ep_reward
    env.close()
    return total_reward / n_eval_episodes


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop (ProPS-compatible interface)
# ─────────────────────────────────────────────────────────────────────────────

def run_training_loop(
    # Required
    env_name: str,
    param_dim: int,
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
    max_iterations: int = 400,        # same budget as ProPS
    n_eval_episodes: int = 5,
    n_candidates: int = 5,
    seed: int = 0,
    # Pretrained weights
    pretrained_ckpt: str = None,      # path to jepa_pretrained.pt
    online_finetune: bool = True,     # finetune online with policy search data
    online_lr: float = 1e-4,
    online_finetune_every: int = 10,
    online_finetune_steps: int = 5,
    # Logging
    log_dir: str = "results",
    run_name: str = None,
    device: str = "cpu",
    # Misc
    task: str = "cont_space_jepa",    # for config compatibility
    **kwargs,
):
    """
    Run JEPA-based policy search on a Gymnasium environment.

    Returns: dict with keys 'rewards', 'params', 'best_reward', 'best_params'
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(log_dir, exist_ok=True)
    run_name = run_name or f"jepa_{env_name}_{seed}"
    log_path = os.path.join(log_dir, f"{run_name}.csv")

    log.info(f"[JEPA-ProPS] env={env_name} param_dim={param_dim} max_iter={max_iterations}")

    # ── Build agent ────────────────────────────────────────────────────────
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
        device=device,
    )

    if pretrained_ckpt and os.path.exists(pretrained_ckpt):
        log.info(f"Loading pretrained JEPA from {pretrained_ckpt}")
        state = torch.load(pretrained_ckpt, map_location=device)
        # Strict=False allows partial loading (e.g. param_dim mismatch)
        agent.load_state_dict(state, strict=False)

    # Online optimizer for fine-tuning during policy search
    online_params = [p for n, p in agent.named_parameters()
                     if "ema_head" not in n and p.requires_grad]
    online_optimizer = torch.optim.AdamW(online_params, lr=online_lr)

    # ── Warm-start: random rollouts ────────────────────────────────────────
    log.info("Warm-start: 5 random rollouts")
    for i in range(5):
        theta = np.random.randn(param_dim)
        r = rollout_policy(env_name, theta, n_eval_episodes, seed + i)
        agent.add_to_history(theta, r)
        log.info(f"  Warm-start {i}: reward={r:.2f}")

    # ── Policy search loop ────────────────────────────────────────────────
    all_rewards = []
    all_params = []
    best_reward = -np.inf
    best_params = None
    t_start = time.time()

    with open(log_path, "w") as f:
        f.write("iteration,reward,best_reward,elapsed\n")

    for t in range(max_iterations):
        t_iter_start = time.time()

        # ── Propose next parameters (replaces LLM call) ─────────────────
        theta = agent.propose(n_candidates=n_candidates)

        # ── Evaluate in environment ──────────────────────────────────────
        reward = rollout_policy(env_name, theta, n_eval_episodes, seed + 100 + t)

        # ── Update history ───────────────────────────────────────────────
        agent.add_to_history(theta, reward)
        all_rewards.append(reward)
        all_params.append(theta)

        if reward > best_reward:
            best_reward = reward
            best_params = theta.copy()

        elapsed = time.time() - t_start
        log.info(
            f"[{t+1}/{max_iterations}] reward={reward:.2f} "
            f"best={best_reward:.2f} elapsed={elapsed:.1f}s"
        )

        with open(log_path, "a") as f:
            f.write(f"{t+1},{reward:.4f},{best_reward:.4f},{elapsed:.2f}\n")

        # ── Online fine-tuning step ──────────────────────────────────────
        if online_finetune and len(agent.history) >= 10 and (t + 1) % online_finetune_every == 0:
            _online_finetune(agent, online_optimizer, online_finetune_steps, device)

    results = {
        "rewards": all_rewards,
        "params": all_params,
        "best_reward": best_reward,
        "best_params": best_params,
        "log_path": log_path,
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Online fine-tuning (adapts JEPA to the current task distribution)
# ─────────────────────────────────────────────────────────────────────────────

def _online_finetune(agent: JEPAAgent, optimizer, n_steps: int, device: str):
    """
    Fine-tune JEPA on the current history buffer.
    Creates (context, next) pairs from the history by rolling a window.
    """
    agent.train()
    device = torch.device(device)

    history = agent.history
    if len(history) < 5:
        return

    params_all = np.stack(history.params, axis=0)    # (T, param_dim)
    rewards_all = np.array(history.rewards)          # (T,)

    # Normalize
    mean = params_all.mean(axis=0)
    std = params_all.std(axis=0) + 1e-8
    params_norm = (params_all - mean) / std

    for _ in range(n_steps):
        # Sample a random window
        T = len(history)
        ctx_end = np.random.randint(3, T)
        ctx_start = max(0, ctx_end - 40)

        params_ctx_np = params_norm[ctx_start:ctx_end]
        rewards_ctx_np = rewards_all[ctx_start:ctx_end, None]
        params_target_np = params_norm[ctx_end] if ctx_end < T - 1 else params_norm[-1]

        params_ctx = torch.tensor(params_ctx_np[None], dtype=torch.float32, device=device)
        rewards_ctx = torch.tensor(rewards_ctx_np[None], dtype=torch.float32, device=device)
        params_target = torch.tensor(params_target_np[None], dtype=torch.float32, device=device)
        desired_r = torch.tensor(
            [[float(np.percentile(rewards_all[:ctx_end], 80))]],
            dtype=torch.float32, device=device
        )

        loss = agent.jepa_loss(params_ctx, rewards_ctx, params_target, desired_r)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in agent.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        agent.target_encoder.update_ema()

    agent.eval()
