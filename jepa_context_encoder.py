"""
JEPA Context Encoder
====================
Encodes a variable-length history buffer of (policy_params, reward) tuples
into a fixed-size latent vector s_x.

Architecture: Causal Transformer over the history sequence.
Each token = [theta_i || r_i] = (param_dim + 1) floats.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class JEPAContextEncoder(nn.Module):
    """
    Encodes the history buffer { (θ_1,r_1), ..., (θ_T, r_T) }
    into a single latent vector s_x ∈ R^d_model.

    Input token at step t: [θ_t (param_dim,), r_t (1,)]  → (param_dim+1,)
    Output: mean-pooled hidden state over the full sequence.

    Args:
        param_dim   : dimensionality of the policy parameter vector θ
        d_model     : transformer hidden size
        nhead       : number of attention heads
        num_layers  : number of transformer encoder layers
        dim_ff      : feedforward hidden dim inside transformer
        max_history : maximum allowed history length
        dropout     : dropout rate
    """

    def __init__(
        self,
        param_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_ff: int = 256,
        max_history: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.param_dim = param_dim
        self.d_model = d_model
        self.input_dim = param_dim + 1  # θ concatenated with r

        # Project input tokens to d_model
        self.input_proj = nn.Linear(self.input_dim, d_model)

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=max_history, dropout=dropout)

        # Causal transformer encoder (full attention; history is small)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Layer norm on output
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        params_history: torch.Tensor,   # (batch, T, param_dim)
        rewards_history: torch.Tensor,  # (batch, T, 1)
        key_padding_mask: torch.Tensor = None,  # (batch, T) bool, True = padding
    ) -> torch.Tensor:
        """
        Returns s_x: (batch, d_model)
        """
        # Build token sequence: (batch, T, param_dim+1)
        tokens = torch.cat([params_history, rewards_history], dim=-1)

        # Project to d_model
        x = self.input_proj(tokens)  # (batch, T, d_model)

        # Add positional encoding
        x = self.pos_enc(x)

        # Transformer encoding (with optional mask for variable-length histories)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # Mean-pool over non-padded positions
        if key_padding_mask is not None:
            # mask: True = padded → zero contribution
            valid = (~key_padding_mask).float().unsqueeze(-1)  # (batch, T, 1)
            s_x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            s_x = x.mean(dim=1)  # (batch, d_model)

        return self.out_norm(s_x)
