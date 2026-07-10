"""
JEPAAgent — replaces the LLM in ProPS
======================================
This class is the direct analog of the LLM agent in props-llm.
It maintains the history buffer and proposes the next θ.

Two modes:
  Mode A — Pure JEPA  : JEPA proposes θ with NO LLM involvement.
  Mode B — Hybrid     : JEPA proposes candidate latents; LLM (or a frozen
                        sentence encoder) provides task-semantic conditioning.

The interface is identical to ProPS's LLM agent so the same runner code works:
    agent.propose(history_params, history_rewards, desired_reward) → θ_next

Pre-training:
    Call agent.pretrain_step(batch) to run the JEPA loss on offline data.

Online (policy search) loop:
    At each iteration t:
        θ_t = agent.propose(history[:t], rewards[:t])
        r_t  = env.evaluate(θ_t)
        history.append((θ_t, r_t))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

from agent.jepa_context_encoder import JEPAContextEncoder
from agent.jepa_predictor import JEPAPredictor, JEPATargetEncoder, JEPADecoder


class HistoryBuffer:
    """
    Stores the rolling history of (theta, reward) pairs.
    Mirrors ProPS's in-context history window.
    """

    def __init__(self, param_dim: int, max_len: int = 400):
        self.param_dim = param_dim
        self.max_len = max_len
        self.params: List[np.ndarray] = []
        self.rewards: List[float] = []

    def add(self, theta: np.ndarray, reward: float):
        self.params.append(theta.copy())
        self.rewards.append(float(reward))
        if len(self.params) > self.max_len:
            self.params.pop(0)
            self.rewards.pop(0)

    def __len__(self):
        return len(self.params)

    def to_tensors(
        self, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (params_tensor, rewards_tensor) unsqueezed for batch=1."""
        params_arr = np.stack(self.params, axis=0)        # (T, param_dim)
        rewards_arr = np.array(self.rewards)[:, None]     # (T, 1)
        params_t = torch.tensor(params_arr, dtype=torch.float32, device=device).unsqueeze(0)
        rewards_t = torch.tensor(rewards_arr, dtype=torch.float32, device=device).unsqueeze(0)
        return params_t, rewards_t

    def get_stats(self):
        """Return mean/std of params for normalization (ProPS-style)."""
        arr = np.stack(self.params, axis=0) if self.params else np.zeros((1, self.param_dim))
        return arr.mean(axis=0), arr.std(axis=0) + 1e-8

    def get_desired_reward(self, percentile: float = 80.0) -> float:
        """Return the P-th percentile of seen rewards as the desired reward."""
        if not self.rewards:
            return 1.0
        return float(np.percentile(self.rewards, percentile))


class JEPAAgent(nn.Module):
    """
    Full JEPA Policy Search Agent.

    Args:
        param_dim          : number of policy parameters
        d_model            : JEPA latent dimension
        nhead              : transformer attention heads
        num_enc_layers     : context encoder depth
        num_pred_layers    : predictor MLP depth
        dim_ff             : transformer feedforward dim
        max_history        : max history buffer length
        ema_decay          : EMA decay for target encoder
        desired_pct        : percentile of rewards to use as desired_reward
        noise_std          : exploration noise added to decoded θ
        semantic_dim       : if > 0, task embedding is projected and concatenated
                             (for the LLM+JEPA hybrid variant)
        device             : torch device
    """

    def __init__(
        self,
        param_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_enc_layers: int = 3,
        num_pred_layers: int = 3,
        dim_ff: int = 256,
        max_history: int = 400,
        ema_decay: float = 0.996,
        desired_pct: float = 80.0,
        noise_std: float = 0.05,
        semantic_dim: int = 0,
        device: str = "cpu",
    ):
        super().__init__()
        self.param_dim = param_dim
        self.desired_pct = desired_pct
        self.noise_std = noise_std
        # Auto-fallback: if "cuda" requested but unavailable, use cpu silently
        if device == "cuda" and not torch.cuda.is_available():
            import logging
            logging.getLogger(__name__).warning(
                "device='cuda' requested but no CUDA GPU found — falling back to 'cpu'."
            )
            device = "cpu"
        self.device = torch.device(device)

        # ── JEPA modules ──────────────────────────────────────────────────
        self.context_encoder = JEPAContextEncoder(
            param_dim=param_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_enc_layers,
            dim_ff=dim_ff,
            max_history=max_history,
        )

        # Predictor input dim accounts for optional semantic embedding
        pred_in = d_model + (semantic_dim if semantic_dim > 0 else 0)

        self.predictor = JEPAPredictor(
            d_model=pred_in,   # will concat semantic before passing
            hidden_dim=dim_ff * 2,
            num_layers=num_pred_layers,
        )

        self.target_encoder = JEPATargetEncoder(
            param_dim=param_dim,
            d_model=d_model,
            ema_decay=ema_decay,
        )

        self.decoder = JEPADecoder(
            d_model=d_model,
            param_dim=param_dim,
            hidden_dim=dim_ff * 2,
        )

        # Optional semantic projection (for LLM+JEPA hybrid)
        self.semantic_proj = None
        if semantic_dim > 0:
            self.semantic_proj = nn.Linear(semantic_dim, d_model)

        # History buffer (not an nn.Module)
        self.history = HistoryBuffer(param_dim=param_dim, max_len=max_history)

        self.to(self.device)

    # ─────────────────────────────────────────────────────────────────────
    # JEPA pretraining loss
    # ─────────────────────────────────────────────────────────────────────

    def jepa_loss(
        self,
        params_ctx: torch.Tensor,    # (B, T_ctx, param_dim)  — context history
        rewards_ctx: torch.Tensor,   # (B, T_ctx, 1)
        params_target: torch.Tensor, # (B, param_dim)          — next good θ
        desired_reward: torch.Tensor,# (B, 1)
        semantic_emb: torch.Tensor = None,  # (B, semantic_dim) optional
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        JEPA objective: minimise || ŝ_y − stop_grad(s_y) ||²

        No reconstruction in pixel/param space — only latent prediction.
        """
        # Encode context → s_x
        s_x = self.context_encoder(params_ctx, rewards_ctx, key_padding_mask)

        # Optionally fuse semantic embedding
        if semantic_emb is not None and self.semantic_proj is not None:
            sem = self.semantic_proj(semantic_emb)       # (B, d_model)
            # Concatenate then re-project via predictor (predictor sees d_model + sem)
            s_x = torch.cat([s_x, sem], dim=-1)         # (B, d_model+semantic_dim)
            # NOTE: predictor d_model must equal d_model+semantic_dim in this case
        
        # Predict target latent
        s_y_hat = self.predictor(s_x, desired_reward)   # (B, d_model)

        # Target latent (EMA, stop gradient)
        s_y = self.target_encoder.forward_ema(params_target)  # (B, d_model)

        # Smooth-L1 loss in latent space (more stable than MSE)
        loss = F.smooth_l1_loss(s_y_hat, s_y)
        return loss

    # ─────────────────────────────────────────────────────────────────────
    # Online policy proposal (replaces LLM call in ProPS)
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def propose(
        self,
        semantic_emb: torch.Tensor = None,   # (1, semantic_dim) — task embedding
        n_candidates: int = 5,               # sample multiple and pick best by decodingg
    ) -> np.ndarray:
        """
        Propose a new policy parameter vector θ_{t+1}.

        Equivalent to one LLM API call in ProPS, but purely neural.

        Returns: θ_proposed as numpy array (param_dim,)
        """
        self.eval()
        if len(self.history) == 0:
            # Cold start: sample from unit Gaussian
            return np.random.randn(self.param_dim)

        params_t, rewards_t = self.history.to_tensors(self.device)

        # Normalise params by running stats (ProPS also normalises)
        mean, std = self.history.get_stats()
        params_norm = (params_t - torch.tensor(mean, device=self.device, dtype=torch.float32)) / \
                       torch.tensor(std, device=self.device, dtype=torch.float32)

        # Encode context
        s_x = self.context_encoder(params_norm, rewards_t)  # (1, d_model)

        # Fuse semantic if provided
        if semantic_emb is not None and self.semantic_proj is not None:
            sem = self.semantic_proj(semantic_emb.to(self.device))
            s_x = torch.cat([s_x, sem], dim=-1)

        # Desired reward
        r_des = self.history.get_desired_reward(self.desired_pct)
        desired_r_t = torch.tensor([[r_des]], dtype=torch.float32, device=self.device)

        # Generate n_candidates by adding small noise to s_x
        best_theta = None
        best_noise_scale = self.noise_std
        candidates = []
        for i in range(n_candidates):
            noise = torch.randn_like(s_x) * (best_noise_scale * (1 + i * 0.5))
            s_x_noisy = s_x + noise
            s_y_hat = self.predictor(s_x_noisy, desired_r_t)      # (1, d_model)
            theta_norm = self.decoder(s_y_hat).squeeze(0).cpu().numpy()
            # Denormalize
            theta = theta_norm * std + mean
            candidates.append(theta)

        # Return the median candidate (robust against outliers)
        best_theta = np.median(np.stack(candidates, axis=0), axis=0)
        return best_theta

    # ─────────────────────────────────────────────────────────────────────
    # History management
    # ─────────────────────────────────────────────────────────────────────

    def add_to_history(self, theta: np.ndarray, reward: float):
        self.history.add(theta, reward)

    def reset_history(self):
        self.history = HistoryBuffer(param_dim=self.param_dim, max_len=self.history.max_len)
