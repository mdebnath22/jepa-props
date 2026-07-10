"""
JEPA Pretraining Pipeline
==========================
Generates offline rollout datasets and pretrain the JEPA modules.

Phase 1 — Data Generation:
    Run random search / CEM / ES across all registered environments.
    Store trajectories as (history, next_theta) pairs.

Phase 2 — JEPA Training:
    Train on the offline dataset using the JEPA latent prediction loss.
    Target encoder is updated via EMA after each step.
    Decoder is jointly trained with an auxiliary reconstruction loss.

This mirrors how I-JEPA / V-JEPA are pretrained on large unlabeled datasets,
except our "data" is rollout trajectories in parameter space.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import gymnasium as gym
import logging
from pathlib import Path

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Rollout Data Generation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_linear_policy(
    env_name: str,
    params: np.ndarray,
    n_eval_episodes: int = 3,
    seed: int = 0,
    max_steps: int = 1000,
) -> float:
    """Evaluate a linear policy in a Gymnasium environment."""
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = (
        env.action_space.n
        if hasattr(env.action_space, "n")
        else env.action_space.shape[0]
    )

    total_reward = 0.0
    for ep in range(n_eval_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        steps = 0
        while not done and steps < max_steps:
            # Linear policy: action = argmax(W @ obs)  or  W @ obs (continuous)
            logits = params.reshape(act_dim, obs_dim) @ obs
            if hasattr(env.action_space, "n"):
                action = int(np.argmax(logits))
            else:
                action = np.clip(logits, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            steps += 1
        total_reward += ep_reward
    env.close()
    return total_reward / n_eval_episodes


def generate_trajectory(
    env_name: str,
    param_dim: int,
    n_iterations: int = 200,
    history_window: int = 50,
    obs_dim: int = None,
    seed: int = 42,
) -> List[Dict]:
    """
    Generate one trajectory of (history, next_theta) pairs via random search.

    Each sample in the trajectory:
        {
          'params_ctx'  : (T, param_dim) — history of parameters
          'rewards_ctx' : (T, 1)         — history of rewards
          'params_target': (param_dim,)  — the NEXT parameter tried
          'desired_reward': float        — 80th pct reward in context
        }
    """
    rng = np.random.RandomState(seed)
    params_history = []
    rewards_history = []
    samples = []

    for i in range(n_iterations):
        # Random perturbation around the best seen so far
        if i == 0 or not params_history:
            theta = rng.randn(param_dim)
        else:
            best_idx = int(np.argmax(rewards_history))
            best_theta = params_history[best_idx]
            theta = best_theta + rng.randn(param_dim) * 0.5

        try:
            reward = evaluate_linear_policy(
                env_name, theta, n_eval_episodes=2, seed=seed + i
            )
        except Exception:
            reward = -1e6

        # Store sample BEFORE adding current step (context = past only)
        if len(params_history) >= 2:
            window_start = max(0, len(params_history) - history_window)
            params_ctx = np.stack(params_history[window_start:], axis=0)
            rewards_ctx = np.array(rewards_history[window_start:])[:, None]
            desired_r = float(np.percentile(rewards_history, 80))
            samples.append(
                {
                    "params_ctx": params_ctx.astype(np.float32),
                    "rewards_ctx": rewards_ctx.astype(np.float32),
                    "params_target": theta.astype(np.float32),
                    "desired_reward": np.float32(desired_r),
                }
            )

        params_history.append(theta)
        rewards_history.append(reward)

    return samples


def build_pretraining_dataset(
    env_configs: List[Dict],   # [{'name': 'CartPole-v1', 'param_dim': 8}, ...]
    n_trajectories_per_env: int = 20,
    n_iterations: int = 200,
    history_window: int = 50,
    save_path: str = "data/jepa_pretrain.pkl",
    seed_base: int = 0,
):
    """Generate pretraining data across all environments and save to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    all_samples = []
    for env_cfg in env_configs:
        env_name = env_cfg["name"]
        param_dim = env_cfg["param_dim"]
        log.info(f"Generating data for {env_name} (param_dim={param_dim})")
        for traj_i in range(n_trajectories_per_env):
            seed = seed_base + traj_i * 1000
            try:
                samples = generate_trajectory(
                    env_name, param_dim, n_iterations, history_window, seed=seed
                )
                # Tag samples with env name and param_dim
                for s in samples:
                    s["env_name"] = env_name
                    s["param_dim"] = param_dim
                all_samples.extend(samples)
                log.info(f"  [{env_name}] traj {traj_i}: {len(samples)} samples")
            except Exception as e:
                log.warning(f"  [{env_name}] traj {traj_i} failed: {e}")

    with open(save_path, "wb") as f:
        pickle.dump(all_samples, f)
    log.info(f"Saved {len(all_samples)} pretraining samples to {save_path}")
    return all_samples


# ─────────────────────────────────────────────────────────────────────────────
# 2. PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class JEPAPretrainDataset(Dataset):
    """
    Dataset of (history, next_theta) pairs for JEPA pretraining.
    Handles variable-length histories via padding.
    """

    def __init__(self, samples: List[Dict], max_ctx_len: int = 50):
        self.samples = samples
        self.max_ctx_len = max_ctx_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        T = s["params_ctx"].shape[0]
        param_dim = s["params_ctx"].shape[1]
        pad_len = self.max_ctx_len - T

        # Pad or truncate context to max_ctx_len
        if T > self.max_ctx_len:
            params_ctx = s["params_ctx"][-self.max_ctx_len:]
            rewards_ctx = s["rewards_ctx"][-self.max_ctx_len:]
            mask = torch.zeros(self.max_ctx_len, dtype=torch.bool)
        else:
            params_ctx = np.concatenate(
                [np.zeros((pad_len, param_dim), dtype=np.float32), s["params_ctx"]], axis=0
            )
            rewards_ctx = np.concatenate(
                [np.zeros((pad_len, 1), dtype=np.float32), s["rewards_ctx"]], axis=0
            )
            # True = padding position
            mask = torch.tensor(
                [True] * pad_len + [False] * T, dtype=torch.bool
            )

        return {
            "params_ctx": torch.tensor(params_ctx, dtype=torch.float32),
            "rewards_ctx": torch.tensor(rewards_ctx, dtype=torch.float32),
            "params_target": torch.tensor(s["params_target"], dtype=torch.float32),
            "desired_reward": torch.tensor([[s["desired_reward"]]], dtype=torch.float32),
            "key_padding_mask": mask,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. JEPA Pretraining Loop
# ─────────────────────────────────────────────────────────────────────────────

def pretrain_jepa(
    agent,                         # JEPAAgent instance
    dataset: JEPAPretrainDataset,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    decoder_loss_weight: float = 0.1,
    save_dir: str = "checkpoints",
    log_every: int = 100,
    device: str = "cpu",
):
    """
    Train JEPA on the pretraining dataset.

    Loss = JEPA_latent_loss + decoder_loss_weight * decoder_reconstruction_loss

    The decoder reconstruction loss trains the decoder to invert the target encoder
    (θ → latent → θ̂ ≈ θ), giving it a useful initialization for policy proposal.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device)
    agent = agent.to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Optimizer: all params EXCEPT ema_head (frozen)
    params_to_opt = [
        p for n, p in agent.named_parameters()
        if "ema_head" not in n and p.requires_grad
    ]
    optimizer = torch.optim.AdamW(params_to_opt, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs * len(dataloader)
    )

    step = 0
    for epoch in range(n_epochs):
        agent.train()
        epoch_loss = 0.0
        for batch in dataloader:
            params_ctx = batch["params_ctx"].to(device)
            rewards_ctx = batch["rewards_ctx"].to(device)
            params_target = batch["params_target"].to(device)
            desired_reward = batch["desired_reward"].squeeze(-1).to(device)  # (B,1)
            mask = batch["key_padding_mask"].to(device)

            # ── JEPA latent loss ──
            jepa_loss = agent.jepa_loss(
                params_ctx, rewards_ctx, params_target, desired_reward,
                key_padding_mask=mask,
            )

            # ── Decoder reconstruction loss (auxiliary) ──
            with torch.no_grad():
                s_y = agent.target_encoder.forward_online(params_target)
            theta_hat = agent.decoder(s_y)
            decoder_loss = torch.nn.functional.smooth_l1_loss(theta_hat, params_target)

            loss = jepa_loss + decoder_loss_weight * decoder_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_opt, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # EMA update of target encoder
            agent.target_encoder.update_ema()

            epoch_loss += loss.item()
            step += 1

            if step % log_every == 0:
                log.info(
                    f"[Pretrain] epoch={epoch+1} step={step} "
                    f"loss={loss.item():.4f} "
                    f"jepa={jepa_loss.item():.4f} "
                    f"dec={decoder_loss.item():.4f}"
                )

        avg = epoch_loss / len(dataloader)
        log.info(f"[Pretrain] Epoch {epoch+1}/{n_epochs} avg_loss={avg:.4f}")

        if (epoch + 1) % 10 == 0:
            ckpt = os.path.join(save_dir, f"jepa_epoch_{epoch+1}.pt")
            torch.save(agent.state_dict(), ckpt)
            log.info(f"Saved checkpoint: {ckpt}")

    final_ckpt = os.path.join(save_dir, "jepa_pretrained.pt")
    torch.save(agent.state_dict(), final_ckpt)
    log.info(f"Pretraining complete. Final checkpoint: {final_ckpt}")
    return agent
