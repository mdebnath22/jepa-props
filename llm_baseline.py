"""
llm_baseline.py — Generalized LLM-iterative-optimizer baseline, matching
the core ProPS method (Zhou et al. 2025, arXiv:2511.21928): an LLM is
queried every iteration with the running (theta, reward) history and asked
to propose the next parameter vector, with no external optimizer involved.

Closes two gaps from the original coverage audit:
  - The reviewer's request for "LM baseline on the 16D task" (and others)
  - ProPS's own baseline method, generalized across all envs.py tasks
    rather than being hardcoded to CartPole-via-Ollama as in the earlier
    analysis/compare_methods.py implementation.

Reuses the Ollama infrastructure already deployed on the cluster (assumes
`ollama serve` is running and reachable at --ollama_host).

Returns the same standardized dict as baselines.py:
    {"rewards": [...], "best_reward": float, "wallclock": {...},
     "diagnostics": {parse failures, etc.}}
"""

import time
import re
import json
import logging
import numpy as np

log = logging.getLogger(__name__)

from envs import rollout_policy, TASKS


SYSTEM_PROMPT_TEMPLATE = """You are a good global optimizer, helping me find the global maximum of a function f(params).
I will give you the function evaluation and the current iteration number at each step.
Your goal is to propose input values that efficiently lead us to the global maximum within a limited number of iterations ({max_iterations}).

Regarding the parameters params:
- params is an array of {param_dim} float numbers.
- params values should typically be in the range [-3.0, 3.0].

Here's how we'll interact:
1. I will provide the iteration number and history of previous params and their rewards.
2. You will respond with ONLY a JSON array of {param_dim} floats, e.g. [0.1, -0.5, 2.0, ...]
   Do not include any explanation, markdown formatting, or additional text.

Remember:
- Do not propose previously seen params exactly.
- Explore both positive and negative values.
- Use the reward trend to guide your next proposal toward higher rewards.
"""


def _build_user_prompt(history_params, history_rewards, iteration, max_iterations,
                        max_history=20):
    lines = [f"Iteration {iteration}/{max_iterations}"]
    for p, r in list(zip(history_params, history_rewards))[-max_history:]:
        p_str = ", ".join(f"{x:.3f}" for x in p)
        lines.append(f"params=[{p_str}] reward={r:.3f}")
    lines.append("Propose the next params array (JSON array only):")
    return "\n".join(lines)


def _parse_llm_response(text, param_dim):
    """
    Robust parse: try strict JSON first, fall back to bracket-extraction
    with quote-stripping (mirrors the dual-encoding/quote-bug fixes from
    earlier in this session).
    """
    text = text.strip()
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
        if isinstance(obj, list) and len(obj) == param_dim:
            return np.array([float(x) for x in obj])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: regex-extract a bracketed array, strip stray quotes per element
    m = re.search(r"\[([^\]]+)\]", text)
    if m:
        raw_items = m.group(1).split(",")
        try:
            vals = [float(x.strip().strip('"').strip("'")) for x in raw_items]
            if len(vals) == param_dim:
                return np.array(vals)
        except ValueError:
            pass

    return None


def run_llm_iterative_optimizer(
    env_name, param_dim, seed, max_iterations,
    ollama_model="qwen2.5:7b",
    ollama_host="http://localhost:11434",
    n_episodes=5,
    timeout=240,
    update_every=1,
):
    """
    ProPS-style LLM-iterative-optimizer baseline.

    update_every: query the LLM once per `update_every` iterations (ProPS's
    paper uses update_every=20 episodes -> 400 LLM calls over 8000 episodes;
    here, since our "iteration" already corresponds to one rollout
    evaluation, update_every=1 means one LLM call per rollout, matching the
    simpler single-rollout-per-call convention used elsewhere in this
    pipeline. Set higher to reduce LLM call count if needed for cost.)

    On any LLM parse failure, falls back to a small perturbation around the
    best-known theta (exploitation fallback) rather than silently injecting
    pure random noise, consistent with the dual-encoding design used
    earlier in this project.
    """
    import requests

    timer = {"rollout_s": 0.0, "optimizer_s": 0.0, "llm_call_s": 0.0}
    t_start = time.perf_counter()

    rng = np.random.RandomState(seed)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        max_iterations=max_iterations, param_dim=param_dim
    )

    history_params, history_rewards = [], []
    rewards = []
    best = -np.inf
    best_theta = None

    diagnostics = {"n_llm_calls": 0, "n_parse_failures": 0}

    # Warm-start with a few random rollouts so the LLM has context
    n_warmup = min(5, max_iterations)
    for i in range(n_warmup):
        t0 = time.perf_counter()
        theta = rng.randn(param_dim)
        timer["optimizer_s"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        r = rollout_policy(env_name, theta, n_episodes, seed + i)
        timer["rollout_s"] += time.perf_counter() - t0
        history_params.append(theta)
        history_rewards.append(r)
        rewards.append(r)
        if r > best:
            best, best_theta = r, theta.copy()

    iteration = n_warmup
    while iteration < max_iterations:
        if (iteration - n_warmup) % update_every == 0:
            user_prompt = _build_user_prompt(
                history_params, history_rewards, iteration, max_iterations
            )
            diagnostics["n_llm_calls"] += 1

            t0 = time.perf_counter()
            try:
                resp = requests.post(
                    f"{ollama_host}/api/chat",
                    json={
                        "model": ollama_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "options": {"temperature": 1.0},
                        "stream": False,
                    },
                    timeout=timeout,
                )
                resp.raise_for_status()
                text = resp.json()["message"]["content"]
                timer["llm_call_s"] += time.perf_counter() - t0
            except (requests.exceptions.ConnectionError,
                     requests.exceptions.Timeout) as e:
                raise RuntimeError(
                    f"Ollama call failed (connection/timeout) at {ollama_host}: {e}"
                )
            except Exception as e:
                log.warning(f"LLM call failed (non-network): {e}")
                text = ""

            theta = _parse_llm_response(text, param_dim)
            if theta is None:
                diagnostics["n_parse_failures"] += 1
                # Exploitation fallback (not pure random noise)
                if best_theta is not None:
                    theta = best_theta + rng.randn(param_dim) * 0.15
                else:
                    theta = rng.randn(param_dim)
        else:
            # Reuse last proposed theta with small jitter between LLM calls
            theta = history_params[-1] + rng.randn(param_dim) * 0.05

        t0 = time.perf_counter()
        r = rollout_policy(env_name, theta, n_episodes, seed + iteration + 100)
        timer["rollout_s"] += time.perf_counter() - t0
        history_params.append(theta)
        history_rewards.append(r)
        rewards.append(r)
        if r > best:
            best, best_theta = r, theta.copy()
        iteration += 1

    diagnostics["parse_failure_rate"] = (
        diagnostics["n_parse_failures"] / max(1, diagnostics["n_llm_calls"])
    )

    timer["total_s"] = time.perf_counter() - t_start
    return {
        "rewards": rewards[:max_iterations],
        "best_reward": float(best),
        "wallclock": timer,
        "diagnostics": diagnostics,
    }
