"""
run_props_baseline_redundant — drop-in ablation for compare_methods.py
=========================================================================
Same signature and return type as run_props_baseline(), so it can be
swapped in directly. Tracks additional diagnostics (recovery rate,
disagreement rate, confidence trace) that the plain baseline cannot.
"""

import time
import logging
import numpy as np

from dual_encoding import (
    DUAL_ENCODING_SYSTEM_PROMPT,
    build_dual_encoding_prompt,
    parse_dual_encoding,
    reconcile_dual_encoding,
    exploitation_fallback,
)

log = logging.getLogger(__name__)


def run_props_baseline_redundant(
    env_name, param_dim, seed, max_iterations,
    ollama_model="qwen2.5:7b",
    ollama_host="http://localhost:11434",
    rollout_fn=None,           # injected to avoid importing runner.jepa_runner here
    n_eval_episodes=10,
    timeout=240,
    disagreement_threshold=1.0,
    jitter_std=0.15,
):
    """
    ProPS with dual-encoding redundancy + exploitation fallback.

    rollout_fn(env_name, theta, n_eval_episodes, seed) -> float
        Pass in runner.jepa_runner.rollout_policy from the caller to avoid
        a hard import dependency here.

    Returns the same shape as run_props_baseline:
        {"rewards": [...], "best_reward": float, "diagnostics": {...}}
    """
    import requests as _requests

    if rollout_fn is None:
        raise ValueError("rollout_fn must be provided (e.g. runner.jepa_runner.rollout_policy)")

    rng = np.random.RandomState(seed)

    def _ollama_chat(model, messages, options, timeout):
        resp = _requests.post(
            f"{ollama_host}/api/chat",
            json={"model": model, "messages": messages, "options": options, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

    h_params, h_rewards = [], []
    all_rewards = []
    best, best_theta = -np.inf, None

    # Diagnostics for the "LLM numerical reliability" side-finding
    diag = {
        "n_calls": 0,
        "n_complete_failures": 0,      # both encodings failed -> exploitation fallback used
        "n_partial_recoveries": 0,     # at least one element recovered via redundancy
        "n_disagreements_total": 0,
        "confidence_trace": [],
    }

    def propose(h_params, h_rewards):
        if not h_params:
            return rng.randn(param_dim)

        prompt = build_dual_encoding_prompt(env_name, param_dim, h_params, h_rewards)

        diag["n_calls"] += 1
        try:
            resp = _ollama_chat(
                model=ollama_model,
                messages=[
                    {"role": "system", "content": DUAL_ENCODING_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 1.0},
                timeout=timeout,
            )
            text = resp["message"]["content"]
        except (_requests.exceptions.ConnectionError,
                _requests.exceptions.Timeout) as e:
            raise RuntimeError(
                f"Ollama call failed (connection or timeout) at {ollama_host}. "
                f"Original error: {e}"
            )
        except Exception as e:
            log.warning(f"Unexpected error calling Ollama (non-network): {e}")
            diag["n_complete_failures"] += 1
            diag["confidence_trace"].append(0.0)
            return exploitation_fallback(best_theta, param_dim, jitter_std, rng)

        parsed = parse_dual_encoding(text, param_dim)
        theta, info = reconcile_dual_encoding(parsed, param_dim, disagreement_threshold)

        diag["confidence_trace"].append(info["confidence"])
        diag["n_disagreements_total"] += info["n_disagreements"]
        if info["n_recovered_by_redundancy"] > 0:
            diag["n_partial_recoveries"] += 1

        if theta is None:
            diag["n_complete_failures"] += 1
            log.debug("Complete dual-encoding failure -> exploitation fallback")
            return exploitation_fallback(best_theta, param_dim, jitter_std, rng)

        return theta

    # Warm-start (identical to plain ProPS, for fair comparison)
    n_warmup = max(10, param_dim * 2)
    for i in range(n_warmup):
        theta = rng.randn(param_dim)
        r = rollout_fn(env_name, theta, n_eval_episodes, seed + i)
        h_params.append(theta); h_rewards.append(r)
        all_rewards.append(r)
        if r > best:
            best, best_theta = r, theta.copy()

    t0 = time.time()
    for t in range(max_iterations):
        theta = propose(h_params, h_rewards)
        r = rollout_fn(env_name, theta, n_eval_episodes, seed + 100 + t)
        h_params.append(theta); h_rewards.append(r)
        all_rewards.append(r)
        if r > best:
            best, best_theta = r, theta.copy()
        if (t + 1) % 50 == 0:
            mean_conf = np.mean(diag["confidence_trace"][-50:]) if diag["confidence_trace"] else 1.0
            log.info(
                f"  ProPS-redundant [{t+1}/{max_iterations}] best={best:.1f} "
                f"elapsed={time.time()-t0:.0f}s recent_confidence={mean_conf:.2f}"
            )

    diag["overall_confidence"] = float(np.mean(diag["confidence_trace"])) if diag["confidence_trace"] else 1.0
    diag["complete_failure_rate"] = diag["n_complete_failures"] / max(1, diag["n_calls"])

    return {"rewards": all_rewards, "best_reward": best, "diagnostics": diag}
