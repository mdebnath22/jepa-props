"""
ProPS with Redundant Dual-Encoding (error-correcting-code inspired)
=====================================================================
Standard ProPS asks the LLM for a single JSON array of floats per iteration.
When the array fails to parse (quote marks, wrong length, truncated output,
etc.), the entire proposal is discarded and replaced with PURE RANDOM NOISE.
This is destructive: one malformed token wastes an entire LLM call and
actively HURTS optimization by injecting noise into the search trajectory.

This module borrows from error-correcting codes / redundant channel coding:
the LLM is asked to encode each parameter value TWICE, independently, in
two different unit systems:

  Encoding A (raw)        : float in [-2, 2]      (typical CartPole weight scale)
  Encoding B (percentile) : percentile 0-100 of an implied N(0,1) variable,
                            inverted via the inverse CDF (probit function)

Reconciliation:
  - Both parse, agree (within tolerance)  -> average them (denoised estimate)
  - Both parse, disagree                  -> trust encoding A, flag low confidence
  - Only ONE parses                       -> use that one (partial recovery!)
  - NEITHER parses                        -> exploitation fallback: jitter the
                                              best-known theta, NOT random noise

This converts "malformed LLM output" from a destructive failure into a
soft, partially-recoverable signal -- analogous to how parity bits let a
noisy channel correct single-bit errors instead of discarding the message.

As a side effect, the parse-failure rate becomes a measurable signal of
LLM "confidence" / numerical reliability at each point in the optimization
trajectory -- itself a publishable byproduct of this ablation.
"""

import re
import json
import logging
import numpy as np
from typing import Optional, List, Tuple, Dict
from scipy.stats import norm as _norm

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

DUAL_ENCODING_SYSTEM_PROMPT = """You are a numerical optimizer. Given a history of
parameter vectors and their rewards, propose the next parameter vector to
maximize reward.

For redundancy and error-checking, output your proposal in TWO independent
encodings:

1. "raw": each parameter as a plain float, typically in range [-2, 2]
2. "percentile": each parameter re-expressed as a percentile (0-100) of where
   you believe that value falls in a standard normal distribution N(0,1).
   (e.g. a raw value of 0.0 corresponds to percentile 50; a raw value of
   1.0 corresponds to roughly percentile 84)

Output ONLY a JSON object with this exact structure, no explanation, no markdown:
{"raw": [v1, v2, ...], "percentile": [p1, p2, ...]}

Both arrays must have exactly the same length, matching the parameter
dimension given in the task description."""


def build_dual_encoding_prompt(env_name: str, param_dim: int,
                                history_params: List[np.ndarray],
                                history_rewards: List[float],
                                max_history: int = 20) -> str:
    """Build the user-turn prompt with in-context history."""
    r_arr = np.array(history_rewards)
    r_norm = (r_arr - r_arr.min()) / (np.ptp(r_arr) + 1e-8)

    hist_lines = []
    for p, r in zip(history_params[-max_history:], r_norm[-max_history:]):
        hist_lines.append(f"params:{[round(float(x), 3) for x in p]}, reward:{r:.3f}")
    hist_str = "\n".join(hist_lines)

    return (
        f"Env:{env_name} dim:{param_dim}\n{hist_str}\n"
        f"Propose the next {param_dim}-dimensional parameter vector in BOTH "
        f"encodings as specified."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parsing: extract both encodings independently, tolerate partial failure
# ─────────────────────────────────────────────────────────────────────────────

def _clean_numeric_token(tok: str) -> Optional[float]:
    """Strip quotes/whitespace and attempt float conversion. Returns None on failure."""
    tok = tok.strip().strip('"').strip("'").strip()
    try:
        return float(tok)
    except ValueError:
        return None


def _extract_array(text: str, key: str) -> Optional[List[Optional[float]]]:
    """
    Extract the array following `"key": [...]` from raw LLM text, tolerating
    minor formatting issues (stray quotes, trailing commas, etc).

    Returns a list of floats/None per-element (None = that element failed to
    parse, but the array structure itself was found) or None if the key/array
    wasn't found at all.
    """
    # First try strict JSON parsing of the whole response
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], list):
            out = []
            for v in obj[key]:
                if isinstance(v, (int, float)):
                    out.append(float(v))
                elif isinstance(v, str):
                    out.append(_clean_numeric_token(v))
                else:
                    out.append(None)
            return out
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: regex search for "key": [ ... ]
    pattern = rf'"{key}"\s*:\s*\[([^\]]*)\]'
    m = re.search(pattern, text)
    if not m:
        return None

    raw_items = m.group(1).split(",")
    return [_clean_numeric_token(item) for item in raw_items]


def parse_dual_encoding(
    text: str,
    param_dim: int,
) -> Dict:
    """
    Parse a dual-encoding LLM response.

    Returns a dict:
      {
        'raw':        np.ndarray or None   (per-element None replaced by NaN)
        'percentile': np.ndarray or None   (already inverted to z-scores)
        'raw_valid_mask':        bool array, True where element parsed
        'percentile_valid_mask': bool array, True where element parsed
      }
    """
    result = {
        "raw": None, "percentile": None,
        "raw_valid_mask": None, "percentile_valid_mask": None,
    }

    raw_list = _extract_array(text, "raw")
    pct_list = _extract_array(text, "percentile")

    if raw_list is not None and len(raw_list) == param_dim:
        mask = np.array([v is not None for v in raw_list])
        vals = np.array([v if v is not None else np.nan for v in raw_list])
        result["raw"] = vals
        result["raw_valid_mask"] = mask

    if pct_list is not None and len(pct_list) == param_dim:
        mask = np.array([v is not None for v in pct_list])
        # Invert percentile -> z-score via probit (inverse normal CDF)
        z_vals = np.full(param_dim, np.nan)
        for i, v in enumerate(pct_list):
            if v is not None:
                p = np.clip(v, 0.01, 99.99) / 100.0
                z_vals[i] = _norm.ppf(p)
        result["percentile"] = z_vals
        result["percentile_valid_mask"] = mask

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Reconciliation: combine the two encodings element-wise
# ─────────────────────────────────────────────────────────────────────────────

def reconcile_dual_encoding(
    parsed: Dict,
    param_dim: int,
    disagreement_threshold: float = 1.0,
) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Combine raw + percentile encodings element-wise.

    Returns (theta, info) where theta is None only if BOTH encodings failed
    on EVERY element (complete failure -> caller should use fallback).

    info contains per-call diagnostics:
      - n_recovered_by_redundancy: elements where one encoding failed but the
        other succeeded (this is the direct win vs single-encoding parsing)
      - n_disagreements: elements where both parsed but disagreed beyond threshold
      - n_total_failures: elements where BOTH encodings failed
      - confidence: fraction of elements that parsed in at least one encoding
    """
    raw = parsed["raw"]
    pct = parsed["percentile"]

    info = {
        "n_recovered_by_redundancy": 0,
        "n_disagreements": 0,
        "n_total_failures": 0,
        "confidence": 0.0,
    }

    if raw is None and pct is None:
        info["n_total_failures"] = param_dim
        return None, info

    if raw is None:
        raw = np.full(param_dim, np.nan)
        raw_mask = np.zeros(param_dim, dtype=bool)
    else:
        raw_mask = parsed["raw_valid_mask"]

    if pct is None:
        pct = np.full(param_dim, np.nan)
        pct_mask = np.zeros(param_dim, dtype=bool)
    else:
        pct_mask = parsed["percentile_valid_mask"]

    theta = np.full(param_dim, np.nan)
    for i in range(param_dim):
        a_ok, b_ok = raw_mask[i], pct_mask[i]
        if a_ok and b_ok:
            diff = abs(raw[i] - pct[i])
            if diff <= disagreement_threshold:
                theta[i] = 0.5 * (raw[i] + pct[i])   # denoised average
            else:
                theta[i] = raw[i]                     # trust raw, flag disagreement
                info["n_disagreements"] += 1
        elif a_ok and not b_ok:
            theta[i] = raw[i]
            info["n_recovered_by_redundancy"] += 1
        elif b_ok and not a_ok:
            theta[i] = pct[i]
            info["n_recovered_by_redundancy"] += 1
        else:
            info["n_total_failures"] += 1

    info["confidence"] = 1.0 - (info["n_total_failures"] / param_dim)

    if info["n_total_failures"] == param_dim:
        return None, info  # complete failure on every dimension

    # Fill any remaining NaN (shouldn't happen given logic above, but safe)
    if np.isnan(theta).any():
        theta = np.nan_to_num(theta, nan=0.0)

    return theta, info


# ─────────────────────────────────────────────────────────────────────────────
# Exploitation fallback: jitter best-known theta instead of pure noise
# ─────────────────────────────────────────────────────────────────────────────

def exploitation_fallback(
    best_theta: Optional[np.ndarray],
    param_dim: int,
    jitter_std: float = 0.15,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    When BOTH encodings fail completely, fall back to a small perturbation
    of the best-known theta rather than fresh random noise. This preserves
    the optimization trajectory's progress instead of destroying it.

    If no best_theta is known yet (e.g. very first call), falls back to
    standard Gaussian noise (there's nothing to exploit yet).
    """
    rng = rng or np.random
    if best_theta is None:
        return rng.randn(param_dim)
    return best_theta + rng.randn(param_dim) * jitter_std
