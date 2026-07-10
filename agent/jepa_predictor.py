"""
JEPA Predictor + Target Encoder
================================
predictor g_φ : s_x  →  ŝ_y   (predicted latent of next good θ)
target encoder h_ψ  : θ_target → s_y  (EMA of context encoder, frozen during pred)

The predictor is REWARD-CONDITIONED:
    ŝ_y = g_φ(s_x, r_desired)
where r_desired is set to a high desired reward (e.g. 95th pct of seen rewards).

This makes the system a GOAL-CONDITIONED JEPA:
    "given history s_x and desired reward r*, predict latent of the policy
     that would achieve r*."

Loss (JEPA style, latent only — no reconstruction):
    L = || ŝ_y − stop_grad(s_y) ||²
"""

import torch
import torch.nn as nn
import copy


class JEPAPredictor(nn.Module):
    """
    Maps (context_latent s_x, desired_reward r*) → predicted target latent ŝ_y.

    Args:
        d_model      : latent dimension (must match context encoder output)
        hidden_dim   : hidden size of the MLP
        num_layers   : depth of the MLP
        dropout      : dropout rate
    """

    def __init__(
        self,
        d_model: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        # +1 for the scalar desired reward
        in_dim = d_model + 1

        layers = []
        prev = in_dim
        for i in range(num_layers - 1):
            layers += [nn.Linear(prev, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            prev = hidden_dim
        layers.append(nn.Linear(prev, d_model))

        self.mlp = nn.Sequential(*layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        s_x: torch.Tensor,          # (batch, d_model)
        desired_reward: torch.Tensor,  # (batch, 1)
    ) -> torch.Tensor:
        inp = torch.cat([s_x, desired_reward], dim=-1)  # (batch, d_model+1)
        return self.out_norm(self.mlp(inp))             # (batch, d_model)


class JEPATargetEncoder(nn.Module):
    """
    Encodes a single policy parameter vector θ into target latent s_y.
    Weights are an EMA copy of the context encoder's input_proj + one extra MLP.

    In practice: we use a SEPARATE small MLP here (not the full transformer)
    because the target encoder only sees a single θ, not a sequence.

    EMA update rule:
        h_ψ ← τ * h_ψ + (1-τ) * f_θ_head

    where f_θ_head is the learnable target head (updated by gradient).
    h_ψ is frozen (no gradient), updated by copy_weights() after each step.
    """

    def __init__(
        self,
        param_dim: int,
        d_model: int = 128,
        hidden_dim: int = 256,
        ema_decay: float = 0.996,
    ):
        super().__init__()
        self.ema_decay = ema_decay

        # Online (gradient) head
        self.online_head = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # EMA (no gradient) head — copy of online at init
        self.ema_head = copy.deepcopy(self.online_head)
        for p in self.ema_head.parameters():
            p.requires_grad_(False)

    def forward_online(self, theta: torch.Tensor) -> torch.Tensor:
        """Used during training loss computation (gradients flow)."""
        return self.online_head(theta)

    def forward_ema(self, theta: torch.Tensor) -> torch.Tensor:
        """Used to produce stop-gradient target (no gradients)."""
        with torch.no_grad():
            return self.ema_head(theta)

    @torch.no_grad()
    def update_ema(self):
        """Call after each optimizer step to update EMA weights."""
        tau = self.ema_decay
        for online_p, ema_p in zip(
            self.online_head.parameters(), self.ema_head.parameters()
        ):
            ema_p.data.mul_(tau).add_((1.0 - tau) * online_p.data)


class JEPADecoder(nn.Module):
    """
    Maps predicted latent ŝ_y → θ_proposed (actual policy parameter vector).

    This is a learned inverse: latent → param space.
    Used only at INFERENCE time (not part of the JEPA pretraining loss).

    Args:
        d_model   : latent dimension
        param_dim : dimensionality of θ
        hidden_dim: hidden size
    """

    def __init__(self, d_model: int = 128, param_dim: int = 10, hidden_dim: int = 256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, param_dim),
        )

    def forward(self, s_y_hat: torch.Tensor) -> torch.Tensor:
        """Returns θ_proposed in normalized space. Caller denormalizes."""
        return self.decoder(s_y_hat)
