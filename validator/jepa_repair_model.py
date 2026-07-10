"""
JEPA Symbol Encoder for XML Repair
====================================
Replaces edit-distance in SP2 with learned latent-space similarity.

The key idea: instead of string edit-distance to match broken references,
we embed both the broken reference AND each candidate declared name into
a shared latent space, then pick the nearest neighbor.

More powerfully, the JEPA *predictor* can take the full XML context
surrounding the broken reference and predict what the correct symbol
*should* be — this is predictive repair, not just similarity matching.

Three components:
  1. SymbolEncoder      : embeds a symbol string + local XML context → latent
  2. JEPARepairPredictor: given broken-ref context → predicts correct latent
  3. JEPARepairMatcher  : nearest-neighbor in latent space over candidates

Training signal (self-supervised, no labels needed):
  - Corrupt a valid XML by randomly perturbing joint names
  - Train JEPA to predict the original (correct) symbol latent
    from the corrupted context
  - Loss: smooth-L1 in latent space (standard JEPA objective)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from typing import List, Tuple, Optional, Dict
import copy


# ─────────────────────────────────────────────────────────────────────────────
# Character-level tokenizer for XML symbols
# ─────────────────────────────────────────────────────────────────────────────

VOCAB = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-. <>=/\"")
VOCAB_SIZE = len(VOCAB) + 2  # +2 for PAD and UNK
PAD_ID = 0
UNK_ID = len(VOCAB) + 1
CHAR2ID = {c: i + 1 for i, c in enumerate(VOCAB)}


def tokenize_symbol(s: str, max_len: int = 64) -> torch.Tensor:
    """Convert a symbol string to character-level token IDs."""
    ids = [CHAR2ID.get(c, UNK_ID) for c in s[:max_len]]
    ids += [PAD_ID] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def tokenize_context(xml_snippet: str, max_len: int = 256) -> torch.Tensor:
    """Convert an XML context snippet to character-level token IDs."""
    return tokenize_symbol(xml_snippet, max_len=max_len)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Symbol Encoder
# ─────────────────────────────────────────────────────────────────────────────

class SymbolEncoder(nn.Module):
    """
    Encodes a symbol string (e.g. joint name) into a latent vector.

    Input:  character sequence of the symbol
    Output: latent vector z ∈ R^d_model

    Architecture: character embedding → 1D conv → mean pool → MLP
    (lightweight; runs on every candidate at repair time)
    """

    def __init__(
        self,
        d_model: int = 64,
        char_embed_dim: int = 16,
        n_filters: int = 64,
        kernel_sizes: List[int] = [2, 3, 4],
        max_symbol_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_symbol_len = max_symbol_len

        self.char_embed = nn.Embedding(VOCAB_SIZE, char_embed_dim, padding_idx=PAD_ID)

        # Multi-scale convolutions (char-CNN, like in NLP character models)
        self.convs = nn.ModuleList([
            nn.Conv1d(char_embed_dim, n_filters, k, padding=k // 2)
            for k in kernel_sizes
        ])

        conv_out = n_filters * len(kernel_sizes)
        self.proj = nn.Sequential(
            nn.Linear(conv_out, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (batch, seq_len)
        returns:   (batch, d_model)
        """
        x = self.char_embed(token_ids)          # (B, L, char_embed_dim)
        x = x.transpose(1, 2)                   # (B, char_embed_dim, L)

        pooled = []
        for conv in self.convs:
            h = F.gelu(conv(x))                  # (B, n_filters, L')
            h = h.max(dim=-1).values             # (B, n_filters) — max pool
            pooled.append(h)

        h = torch.cat(pooled, dim=-1)            # (B, n_filters * len(kernel_sizes))
        return self.proj(h)                      # (B, d_model)

    def encode_string(self, s: str, device="cpu") -> torch.Tensor:
        """Convenience: encode a raw string directly."""
        ids = tokenize_symbol(s, self.max_symbol_len).unsqueeze(0).to(device)
        return self.forward(ids)                 # (1, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Context Encoder
# ─────────────────────────────────────────────────────────────────────────────

class XMLContextEncoder(nn.Module):
    """
    Encodes the XML snippet surrounding a broken reference.

    This gives the repair predictor structural context:
    "this broken ref appears inside <motor joint=...> which is
     in the <actuator> block, the worldbody declared these joints..."

    Input:  character sequence of the XML context window
    Output: context latent c ∈ R^d_model
    """

    def __init__(
        self,
        d_model: int = 64,
        char_embed_dim: int = 16,
        nhead: int = 4,
        num_layers: int = 2,
        max_context_len: int = 256,
    ):
        super().__init__()
        self.max_context_len = max_context_len
        self.d_model = d_model

        self.char_embed = nn.Embedding(VOCAB_SIZE, char_embed_dim, padding_idx=PAD_ID)
        self.input_proj = nn.Linear(char_embed_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (batch, seq_len)
        returns:   (batch, d_model)
        """
        pad_mask = (token_ids == PAD_ID)
        x = self.char_embed(token_ids)     # (B, L, char_embed_dim)
        x = self.input_proj(x)             # (B, L, d_model)
        x = self.transformer(x, src_key_padding_mask=pad_mask)

        # Mean-pool over non-padded positions
        valid = (~pad_mask).float().unsqueeze(-1)
        c = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.out_norm(c)            # (B, d_model)

    def encode_string(self, s: str, device="cpu") -> torch.Tensor:
        ids = tokenize_context(s, self.max_context_len).unsqueeze(0).to(device)
        return self.forward(ids)


# ─────────────────────────────────────────────────────────────────────────────
# 3. JEPA Repair Predictor
# ─────────────────────────────────────────────────────────────────────────────

class JEPARepairPredictor(nn.Module):
    """
    Given:
      - z_broken : embedding of the broken/corrupted reference
      - c_context: embedding of the surrounding XML context
    Predicts:
      - z_correct : embedding of what the correct reference should be

    This is the JEPA prediction in symbol space:
      ẑ_correct = g_φ(z_broken, c_context)

    Training:
      s_y = stop_grad( SymbolEncoder(correct_name) )
      Loss = smooth_L1( ẑ_correct, s_y )

    The key difference from edit-distance SP2:
      Edit-distance: picks candidate nearest to broken string (surface form)
      JEPA-SP2:      predicts correct embedding from context, THEN matches
                     → can exploit structural context that edit-distance ignores
    """

    def __init__(
        self,
        d_model: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = d_model * 2  # z_broken + c_context concatenated

        layers = []
        prev = in_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(prev, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            prev = hidden_dim
        layers.append(nn.Linear(prev, d_model))

        self.mlp = nn.Sequential(*layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        z_broken: torch.Tensor,     # (B, d_model)
        c_context: torch.Tensor,    # (B, d_model)
    ) -> torch.Tensor:
        inp = torch.cat([z_broken, c_context], dim=-1)   # (B, 2*d_model)
        return self.out_norm(self.mlp(inp))               # (B, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# 4. JEPA Repair Matcher — replaces edit-distance in SP2
# ─────────────────────────────────────────────────────────────────────────────

class JEPARepairMatcher(nn.Module):
    """
    Full JEPA-based SP2 replacement.

    Given a broken reference and the set of declared names, returns
    the best match using two strategies:
      Mode A (JEPA-SP2-match):  embed all candidates, pick nearest to z_broken
      Mode B (JEPA-SP2-pred):   predict z_correct from context, pick nearest to z_correct

    Also provides a confidence score analogous to the δ threshold in edit-distance SP2.
    """

    def __init__(
        self,
        d_model: int = 64,
        char_embed_dim: int = 16,
        n_filters: int = 64,
        nhead: int = 4,
        num_context_layers: int = 2,
        num_pred_layers: int = 3,
        ema_decay: float = 0.996,
    ):
        super().__init__()
        self.d_model = d_model

        # Online symbol encoder (trained via gradient)
        self.symbol_encoder = SymbolEncoder(
            d_model=d_model,
            char_embed_dim=char_embed_dim,
            n_filters=n_filters,
        )

        # EMA target symbol encoder (stop-gradient target)
        self.target_symbol_encoder = copy.deepcopy(self.symbol_encoder)
        for p in self.target_symbol_encoder.parameters():
            p.requires_grad_(False)

        # Context encoder
        self.context_encoder = XMLContextEncoder(
            d_model=d_model,
            char_embed_dim=char_embed_dim,
            nhead=nhead,
            num_layers=num_context_layers,
        )

        # Repair predictor (JEPA core)
        self.predictor = JEPARepairPredictor(
            d_model=d_model,
            hidden_dim=d_model * 2,
            num_layers=num_pred_layers,
        )

        self.ema_decay = ema_decay

    # ── JEPA pretraining loss ─────────────────────────────────────────────

    def jepa_repair_loss(
        self,
        broken_ids: torch.Tensor,   # (B, symbol_len) — corrupted name
        context_ids: torch.Tensor,  # (B, context_len) — surrounding XML
        correct_ids: torch.Tensor,  # (B, symbol_len) — ground truth name
    ) -> torch.Tensor:
        """
        JEPA loss for repair:
          z_broken  = SymbolEncoder(broken_name)
          c_context = ContextEncoder(xml_context)
          ẑ_correct = Predictor(z_broken, c_context)
          z_correct = stop_grad( TargetEncoder(correct_name) )
          L = smooth_L1(ẑ_correct, z_correct)
        """
        z_broken = self.symbol_encoder(broken_ids)      # (B, d_model)
        c_context = self.context_encoder(context_ids)   # (B, d_model)
        z_hat = self.predictor(z_broken, c_context)     # (B, d_model)

        with torch.no_grad():
            z_correct = self.target_symbol_encoder(correct_ids)  # (B, d_model)

        return F.smooth_l1_loss(z_hat, z_correct)

    @torch.no_grad()
    def update_ema(self):
        tau = self.ema_decay
        for online_p, ema_p in zip(
            self.symbol_encoder.parameters(),
            self.target_symbol_encoder.parameters(),
        ):
            ema_p.data.mul_(tau).add_((1.0 - tau) * online_p.data)

    # ── Inference: replace SP2 edit-distance ─────────────────────────────

    @torch.no_grad()
    def match(
        self,
        broken_ref: str,
        declared_names: List[str],
        xml_context: str = "",
        mode: str = "predict",     # "match" | "predict"
        confidence_threshold: float = 0.5,
        device: str = "cpu",
    ) -> Tuple[Optional[str], float]:
        """
        Find the best declared name for a broken reference.

        Args:
            broken_ref:           the unresolved reference string
            declared_names:       list of valid declared joint names
            xml_context:          surrounding XML text (for mode='predict')
            mode:                 'match'   → embed broken, find nearest candidate
                                  'predict' → predict correct embedding from context,
                                              find nearest candidate to prediction
            confidence_threshold: minimum cosine similarity to accept a repair
                                  (analog of δ in edit-distance SP2)

        Returns:
            (best_name, confidence) or (None, 0.0) if below threshold
        """
        if not declared_names:
            return None, 0.0

        self.eval()

        # Embed all candidates
        cand_ids = torch.stack(
            [tokenize_symbol(n) for n in declared_names]
        ).to(device)
        z_candidates = self.symbol_encoder(cand_ids)   # (n_cands, d_model)
        z_candidates = F.normalize(z_candidates, dim=-1)

        if mode == "match":
            # Mode A: embed broken ref, cosine-nearest candidate
            broken_ids = tokenize_symbol(broken_ref).unsqueeze(0).to(device)
            z_query = self.symbol_encoder(broken_ids)  # (1, d_model)
            z_query = F.normalize(z_query, dim=-1)

        elif mode == "predict":
            # Mode B: predict correct latent from context, then match
            broken_ids = tokenize_symbol(broken_ref).unsqueeze(0).to(device)
            z_broken = self.symbol_encoder(broken_ids)  # (1, d_model)

            ctx_ids = tokenize_context(xml_context).unsqueeze(0).to(device)
            c_ctx = self.context_encoder(ctx_ids)       # (1, d_model)

            z_query = self.predictor(z_broken, c_ctx)   # (1, d_model) — predicted
            z_query = F.normalize(z_query, dim=-1)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Cosine similarities
        sims = (z_query @ z_candidates.T).squeeze(0)   # (n_cands,)
        best_idx = sims.argmax().item()
        best_sim = sims[best_idx].item()

        # Normalize similarity to [0, 1] range and apply threshold
        # cosine sim is in [-1, 1]; map to [0, 1]
        confidence = (best_sim + 1.0) / 2.0

        if confidence < confidence_threshold:
            return None, confidence

        return declared_names[best_idx], confidence
