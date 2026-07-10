"""
llm_baseline_plus.py — Extended LLM-iterative-optimizer variants matching
ProPS's ProPS+ method (semantic task descriptions + optional expert hints)
plus ablation experiments from ProPS Figure 8:

  - ProPS+: prompts augmented with environment descriptions and policy
    structure (Table 2 vs Table 1 is ProPS vs ProPS+ in the paper)
  - ProPS+ w/ hints: additional human-provided behavioral hints injected
    per environment (Table 3, Figure 6) showing 8-62% reward improvement
  - History-length ablation: sweep N in {1, 20, 40, 80, unbounded} matching
    ProPS Figure 8a (reward as function of history context length)
  - Multi-LLM sweep: run all prompt variants across multiple Ollama models,
    producing a comparison table matching ProPS Figure 8b

All functions return the same dict format as llm_baseline.py and
baselines.py, so they slot directly into run_full_eval.py's baseline
registry without special-casing.

GRPO fine-tuning (ProPS Appendix B, Figure 8c) is explicitly excluded --
that requires LoRA/TRL training infrastructure and is marked as future work.
"""

import time
import re
import json
import logging
import numpy as np

log = logging.getLogger(__name__)

from envs import rollout_policy


# ─────────────────────────────────────────────────────────────────────────────
# Semantic task descriptions for ProPS+ (per-environment natural-language
# descriptions matching ProPS's task-description-augmented prompts)
# ─────────────────────────────────────────────────────────────────────────────

TASK_DESCRIPTIONS = {
    "CartPole-v1": (
        "A pole is attached by an un-actuated joint to a cart moving on a "
        "frictionless track. The policy controls a force applied to the cart "
        "to keep the pole from falling. Observations are [cart_pos, cart_vel, "
        "pole_angle, pole_ang_vel]. The policy is a linear function: "
        "action = sign(W @ obs) where W is a (2, 4) matrix (param_dim=8) or "
        "a (2, 5) matrix with bias (param_dim=10). Reward is +1 per step "
        "while the pole stays upright, up to 500 steps."
    ),
    "Swimmer-v4": (
        "A snake-like robot (2 links, 3 joints) must swim forward in viscous "
        "fluid by coordinating joint torques. Observations are 8 values "
        "including joint angles and angular velocities. The policy is a linear "
        "function mapping 8 observations to 2 continuous actions (joint "
        "torques in [-1, 1]). Higher forward velocity gives higher reward."
    ),
    "MountainCarContinuous-v0": (
        "A car must drive up a steep mountain by building momentum via "
        "oscillation. Observations are [position, velocity] (2 dims). "
        "The policy maps to 1 continuous action (engine force in [-1, 1]). "
        "Reaching the goal gives a large positive reward; slow progress "
        "is penalized."
    ),
    "InvertedPendulum-v4": (
        "A pendulum must stay upright on a cart. Observations are 4 values "
        "(cart pos/vel, pole angle/vel). The policy maps to 1 continuous "
        "force action. Reward is +1 per step while balanced; episode ends "
        "on failure."
    ),
    "Hopper-v4": (
        "A 2D one-legged robot must hop forward without falling. Observations "
        "are 11 values (joint angles and velocities). The policy maps to 3 "
        "continuous joint torques. Forward velocity gives positive reward; "
        "falling ends the episode."
    ),
    "FrozenLake-v1": (
        "An agent on a 4x4 ice grid must reach a goal without falling into "
        "holes. The ice is slippery (stochastic transitions). State is an "
        "integer 0-15. The policy maps each state to one of 4 actions "
        "(up/right/down/left). Reaching the goal gives reward +1."
    ),
    "Maze": (
        "An agent navigates a 4x4 grid maze with walls and holes. The state "
        "is an integer cell index 0-15. The policy maps each state to one of "
        "4 actions (up/right/down/left). Reaching the goal gives +1; falling "
        "in a hole gives -1; each step costs -0.01."
    ),
    "Nim": (
        "A combinatorial game with 3 piles of objects. On each turn, remove "
        "any positive number from one pile. The player who takes the last "
        "object wins. The action encodes (pile_index, n_to_remove). "
        "Win gives +1, loss gives -1. The key insight: XOR of pile sizes "
        "determines the winning strategy."
    ),
    "Navigation": (
        "A 2D continuous navigation task. The agent must reach a goal while "
        "avoiding circular obstacles. Observation is [x, y, vx, vy, gx, gy] "
        "(6 dims). The policy maps to 2 continuous acceleration values in "
        "[-1, 1]. Dense shaping reward plus +500 bonus on reaching the goal."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Expert behavioral hints per environment (ProPS Table 3 / Figure 6 protocol)
# These are representative of the kind of human-provided hints ProPS uses --
# not copied verbatim (the paper doesn't publish the exact hints), but
# functionally equivalent task-relevant heuristics.
# ─────────────────────────────────────────────────────────────────────────────

TASK_HINTS = {
    "CartPole-v1": (
        "Hint: To balance the pole, apply force in the same direction as the "
        "pole's angular velocity and tilt. If the pole is tilting right and "
        "falling rightward, push the cart right. Negative obs[2] means the "
        "pole leans left; respond with negative (left) force."
    ),
    "Swimmer-v4": (
        "Hint: Coordinated sinusoidal oscillation of the two joints drives "
        "forward locomotion. Try parameters that produce alternating positive "
        "and negative torques on the two joints. Larger magnitude torques "
        "generally produce faster swimming up to the saturation point."
    ),
    "MountainCarContinuous-v0": (
        "Hint: The car cannot drive up the hill directly. Build momentum by "
        "driving to the left first, then push right at the lowest point of "
        "the valley. The policy should push in the direction of current "
        "velocity (sign(obs[1])) rather than always pushing right."
    ),
    "InvertedPendulum-v4": (
        "Hint: Apply force proportional to and in the direction of the pole's "
        "tilt and angular velocity. A simple PD controller works well: "
        "force = k1 * pole_angle + k2 * pole_angular_velocity with positive "
        "k1 and k2."
    ),
    "Hopper-v4": (
        "Hint: Stable hopping requires coordinated timing of the three joints "
        "(hip, knee, ankle). The ankle joint (last action) provides the main "
        "push-off force; try keeping it near its maximum. The policy should "
        "not produce symmetric joint torques."
    ),
    "FrozenLake-v1": (
        "Hint: The goal is at cell 15 (bottom-right). From most cells, "
        "moving right (action 1) or down (action 2) is beneficial. Avoid "
        "cells 5, 7, 11, 12 (holes). Due to slippery ice, the agent may "
        "need to attempt actions repeatedly."
    ),
    "Maze": (
        "Hint: The goal is at the bottom-right corner. Navigate around holes. "
        "From the start (top-left), generally move right and down. "
        "Avoid hole cells: check which cells are dangerous and route around them."
    ),
    "Nim": (
        "Hint: The optimal Nim strategy is based on XOR (nim-sum) of pile "
        "sizes. If the nim-sum is 0, you are in a losing position; if nonzero, "
        "you can always move to a nim-sum=0 state. Try to encode parameters "
        "that implement XOR-based move selection."
    ),
    "Navigation": (
        "Hint: Move directly toward the goal while steering around obstacles. "
        "The acceleration should be proportional to (goal - current_position). "
        "When near an obstacle, steer perpendicular to it. The goal coordinates "
        "are the last two elements of the observation."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders for ProPS vs ProPS+ vs ProPS+ with hints
# ─────────────────────────────────────────────────────────────────────────────

def _build_system_prompt(env_name, param_dim, max_iterations, mode):
    """
    mode: "props" (numbers only), "props_plus" (with semantic description),
          "props_plus_hints" (with description + behavioral hints)
    """
    base = (
        f"You are a good global optimizer, helping me find the policy parameters "
        f"that maximize reward in a reinforcement learning environment.\n"
        f"I will give you parameter evaluations at each step. "
        f"Your goal is to find the best parameters within {max_iterations} iterations.\n\n"
        f"The parameter vector has {param_dim} floats.\n"
        f"Respond ONLY with a JSON array of {param_dim} floats, e.g. [0.1, -0.5, ...]\n"
        f"No explanation, markdown, or extra text."
    )

    if mode in ("props_plus", "props_plus_hints"):
        desc = TASK_DESCRIPTIONS.get(env_name, "")
        if desc:
            base = f"Environment Description:\n{desc}\n\n" + base

    if mode == "props_plus_hints":
        hint = TASK_HINTS.get(env_name, "")
        if hint:
            base += f"\n\n{hint}"

    return base


def _build_user_prompt(history_params, history_rewards, iteration,
                        max_iterations, max_history):
    lines = [f"Iteration {iteration}/{max_iterations}"]
    for p, r in list(zip(history_params, history_rewards))[-max_history:]:
        p_str = ", ".join(f"{x:.3f}" for x in p)
        lines.append(f"params=[{p_str}] reward={r:.3f}")
    lines.append("Propose the next params array (JSON array only):")
    return "\n".join(lines)


def _parse_response(text, param_dim):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, list) and len(obj) == param_dim:
            return np.array([float(x) for x in obj])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    m = re.search(r"\[([^\]]+)\]", text)
    if m:
        try:
            vals = [float(x.strip().strip('"').strip("'"))
                    for x in m.group(1).split(",")]
            if len(vals) == param_dim:
                return np.array(vals)
        except ValueError:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Core LLM optimizer with mode and history-length arguments
# ─────────────────────────────────────────────────────────────────────────────

def run_llm_optimizer_variant(
    env_name, param_dim, seed, max_iterations,
    mode="props",
    max_history=20,
    ollama_model="qwen2.5:7b",
    ollama_host="http://localhost:11434",
    n_episodes=5,
    timeout=240,
):
    """
    Unified LLM optimizer supporting ProPS/ProPS+/ProPS+ w/ hints modes.

    mode: "props"             -- numbers-only prompt (baseline, matches ProPS)
          "props_plus"        -- adds semantic environment description (ProPS+)
          "props_plus_hints"  -- adds description + behavioral hints (ProPS+ w/ hints)
    max_history: number of past (params, reward) pairs to include in context
                 (ablated in sweep_history_length below)
    """
    import requests

    timer = {"rollout_s": 0.0, "optimizer_s": 0.0, "llm_call_s": 0.0}
    t_start = time.perf_counter()

    rng = np.random.RandomState(seed)
    system_prompt = _build_system_prompt(env_name, param_dim, max_iterations, mode)

    history_params, history_rewards = [], []
    rewards = []
    best = -np.inf
    best_theta = None
    diagnostics = {"n_llm_calls": 0, "n_parse_failures": 0, "mode": mode,
                   "max_history": max_history, "model": ollama_model}

    # Warm-start
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

    for iteration in range(n_warmup, max_iterations):
        user_prompt = _build_user_prompt(
            history_params, history_rewards, iteration, max_iterations, max_history
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
            raise RuntimeError(f"Ollama call failed at {ollama_host}: {e}")
        except Exception as e:
            log.warning(f"LLM call failed (non-network): {e}")
            text = ""

        theta = _parse_response(text, param_dim)
        if theta is None:
            diagnostics["n_parse_failures"] += 1
            theta = (best_theta + rng.randn(param_dim) * 0.15
                     if best_theta is not None else rng.randn(param_dim))

        t0 = time.perf_counter()
        r = rollout_policy(env_name, theta, n_episodes, seed + iteration + 100)
        timer["rollout_s"] += time.perf_counter() - t0

        history_params.append(theta)
        history_rewards.append(r)
        rewards.append(r)
        if r > best:
            best, best_theta = r, theta.copy()

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


# ─────────────────────────────────────────────────────────────────────────────
# History-length ablation (ProPS Figure 8a)
# ─────────────────────────────────────────────────────────────────────────────

def sweep_history_length(
    env_name, param_dim, seed, max_iterations,
    history_lengths=(1, 20, 40, 80, None),
    mode="props_plus",
    ollama_model="qwen2.5:7b",
    ollama_host="http://localhost:11434",
    n_episodes=5,
):
    """
    Sweeps context history length N in {1, 20, 40, 80, None (unbounded)}.
    None maps to max_iterations (effectively unbounded), matching ProPS's
    "unbounded" condition in Figure 8a.

    Returns dict keyed by history length label.
    """
    results = {}
    for length in history_lengths:
        effective_length = length if length is not None else max_iterations
        label = str(length) if length is not None else "unbounded"
        log.info(f"History sweep: max_history={label}, mode={mode}")
        r = run_llm_optimizer_variant(
            env_name, param_dim, seed, max_iterations,
            mode=mode, max_history=effective_length,
            ollama_model=ollama_model, ollama_host=ollama_host,
            n_episodes=n_episodes,
        )
        results[label] = r
        log.info(f"  max_history={label}: best={r['best_reward']:.2f} "
                 f"mean={np.mean(r['rewards']):.2f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Multi-LLM comparison sweep (ProPS Figure 8b)
# ─────────────────────────────────────────────────────────────────────────────

def sweep_models(
    env_name, param_dim, seed, max_iterations,
    models=("qwen2.5:7b", "qwen2.5:14b"),
    mode="props_plus",
    ollama_host="http://localhost:11434",
    n_episodes=5,
):
    """
    Runs the same (env, mode) combination across multiple Ollama models,
    producing a comparison table matching ProPS Figure 8b's multi-LLM sweep.

    Models should be ones you have pulled on the cluster. Default is the two
    Qwen2.5 sizes available via your existing Ollama setup. Extend as needed:
    e.g. ("qwen2.5:7b", "qwen2.5:14b", "llama3.1:8b", "mistral:7b")
    """
    results = {}
    for model in models:
        log.info(f"Model sweep: {model}, env={env_name}, mode={mode}")
        r = run_llm_optimizer_variant(
            env_name, param_dim, seed, max_iterations,
            mode=mode, ollama_model=model, ollama_host=ollama_host,
            n_episodes=n_episodes,
        )
        results[model] = r
        log.info(f"  {model}: best={r['best_reward']:.2f} "
                 f"parse_failures={r['diagnostics']['parse_failure_rate']:.1%}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper for mode comparison (ProPS vs ProPS+ vs ProPS+ w/ hints) in one call
# ─────────────────────────────────────────────────────────────────────────────

def compare_prompt_modes(
    env_name, param_dim, seed, max_iterations,
    modes=("props", "props_plus", "props_plus_hints"),
    ollama_model="qwen2.5:7b",
    ollama_host="http://localhost:11434",
    n_episodes=5,
    max_history=20,
):
    """
    Run all three prompt modes on the same (env, seed), matching the
    ProPS / ProPS+ / ProPS+ w/ hints comparison in Tables 2-3.
    Returns dict keyed by mode name.
    """
    results = {}
    for mode in modes:
        log.info(f"Mode comparison: {mode}, env={env_name}")
        r = run_llm_optimizer_variant(
            env_name, param_dim, seed, max_iterations,
            mode=mode, max_history=max_history,
            ollama_model=ollama_model, ollama_host=ollama_host,
            n_episodes=n_episodes,
        )
        results[mode] = r
        log.info(f"  {mode}: best={r['best_reward']:.2f} "
                 f"mean={np.mean(r['rewards']):.2f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrappers matching baselines.py's call signature for CLI dispatch
# ─────────────────────────────────────────────────────────────────────────────

def run_props_plus(env_name, param_dim, seed, max_iterations, n_episodes=5,
                   ollama_model="qwen2.5:7b",
                   ollama_host="http://localhost:11434"):
    return run_llm_optimizer_variant(
        env_name, param_dim, seed, max_iterations,
        mode="props_plus", n_episodes=n_episodes,
        ollama_model=ollama_model, ollama_host=ollama_host,
    )


def run_props_plus_hints(env_name, param_dim, seed, max_iterations, n_episodes=5,
                          ollama_model="qwen2.5:7b",
                          ollama_host="http://localhost:11434"):
    return run_llm_optimizer_variant(
        env_name, param_dim, seed, max_iterations,
        mode="props_plus_hints", n_episodes=n_episodes,
        ollama_model=ollama_model, ollama_host=ollama_host,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt robustness ablation (ProPS Tables 11, 12)
# 3 paraphrased variations of both the base prompt and the environment
# description, to measure reward sensitivity to exact wording choices.
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_VARIATIONS = {
    "props": {
        # Original ProPS numbers-only prompt (already in _build_system_prompt)
        "original": None,  # None means use default
        "variation_1": (
            "You are an expert policy optimizer. Help me maximize the cumulative "
            "reward of a control agent by proposing policy parameter vectors. "
            "At each step I give you a history of parameter vectors and their "
            "rewards; you respond with only a JSON array of {param_dim} floats "
            "for the next candidate. No explanation, just the array."
        ),
        "variation_2": (
            "I need to optimize {param_dim} policy parameters to maximize reward. "
            "I will show you past (params, reward) pairs. Propose the next "
            "parameter vector as a JSON array of {param_dim} floats that you "
            "believe will achieve a higher reward than any seen so far. "
            "Respond only with the array, nothing else."
        ),
        "variation_3": (
            "Task: iterative policy parameter optimization. "
            "Input: history of {param_dim}-dimensional parameter vectors and "
            "scalar rewards. Output: one JSON array of {param_dim} floats "
            "predicting higher reward. No preamble, no explanation."
        ),
    },
    "props_plus": {
        "original": None,
        "variation_1": (
            "You are optimizing parameters for a reinforcement learning policy. "
            "The environment is described below. Use this context to make "
            "informed parameter proposals. Respond with a JSON array of "
            "{param_dim} floats only.\n\n{description}"
        ),
        "variation_2": (
            "Environment context:\n{description}\n\n"
            "Your task: propose the next {param_dim} policy parameters (JSON "
            "array only) to maximize cumulative reward, using the provided "
            "context and the reward history I will give you."
        ),
        "variation_3": (
            "Given the following environment:\n{description}\n\n"
            "Optimize a policy with {param_dim} parameters. "
            "At each iteration, review the reward history I provide and "
            "return only a JSON array of {param_dim} floats for the next "
            "parameter proposal."
        ),
    },
}


def _fill_prompt_template(template, env_name, param_dim):
    """Fill placeholders in a prompt variation template."""
    desc = TASK_DESCRIPTIONS.get(env_name, "")
    return template.format(param_dim=param_dim, description=desc)


def sweep_prompt_variations(
    env_name, param_dim, seed, max_iterations,
    mode="props",
    ollama_model="qwen2.5:7b",
    ollama_host="http://localhost:11434",
    n_episodes=5,
    max_history=20,
):
    """
    Runs the same (env, seed) across original + 3 prompt variations,
    matching ProPS Tables 11 and 12's prompt robustness ablation.

    mode: "props" (Table 11 -- no description, test instruction wording)
          "props_plus" (Table 12 -- with description, test description wording)
    """
    import requests
    variations = PROMPT_VARIATIONS[mode]
    results = {}

    for var_name, template in variations.items():
        if template is None:
            # Use the default system prompt from _build_system_prompt
            sys_prompt = _build_system_prompt(env_name, param_dim, max_iterations, mode)
        else:
            sys_prompt = _fill_prompt_template(template, env_name, param_dim)
            # Still inject task description at top if props_plus and description not already in template
            if mode == "props_plus" and "{description}" not in template:
                desc = TASK_DESCRIPTIONS.get(env_name, "")
                if desc:
                    sys_prompt = f"Environment Description:\n{desc}\n\n" + sys_prompt

        log.info(f"Prompt variation: {var_name} ({mode})")
        r = _run_with_custom_system_prompt(
            env_name, param_dim, seed, max_iterations,
            system_prompt=sys_prompt, max_history=max_history,
            ollama_model=ollama_model, ollama_host=ollama_host,
            n_episodes=n_episodes,
        )
        r["diagnostics"]["prompt_variation"] = var_name
        results[var_name] = r
        log.info(f"  {var_name}: best={r['best_reward']:.2f} mean={np.mean(r['rewards']):.2f}")

    return results


def _run_with_custom_system_prompt(
    env_name, param_dim, seed, max_iterations,
    system_prompt, max_history=20,
    ollama_model="qwen2.5:7b",
    ollama_host="http://localhost:11434",
    n_episodes=5,
    timeout=240,
):
    """Core runner with a fully custom system_prompt string."""
    import requests

    timer = {"rollout_s": 0.0, "optimizer_s": 0.0, "llm_call_s": 0.0}
    t_start = time.perf_counter()
    rng = np.random.RandomState(seed)

    history_params, history_rewards = [], []
    rewards = []
    best = -np.inf
    best_theta = None
    diagnostics = {"n_llm_calls": 0, "n_parse_failures": 0}

    n_warmup = min(5, max_iterations)
    for i in range(n_warmup):
        theta = rng.randn(param_dim)
        t0 = time.perf_counter()
        r = rollout_policy(env_name, theta, n_episodes, seed + i)
        timer["rollout_s"] += time.perf_counter() - t0
        history_params.append(theta)
        history_rewards.append(r)
        rewards.append(r)
        if r > best:
            best, best_theta = r, theta.copy()

    for iteration in range(n_warmup, max_iterations):
        user_prompt = _build_user_prompt(
            history_params, history_rewards, iteration, max_iterations, max_history
        )
        diagnostics["n_llm_calls"] += 1
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{ollama_host}/api/chat",
                json={"model": ollama_model,
                      "messages": [{"role": "system", "content": system_prompt},
                                   {"role": "user", "content": user_prompt}],
                      "options": {"temperature": 1.0}, "stream": False},
                timeout=timeout,
            )
            resp.raise_for_status()
            text = resp.json()["message"]["content"]
            timer["llm_call_s"] += time.perf_counter() - t0
        except Exception as e:
            log.warning(f"LLM call failed: {e}")
            text = ""

        theta = _parse_response(text, param_dim)
        if theta is None:
            diagnostics["n_parse_failures"] += 1
            theta = (best_theta + rng.randn(param_dim) * 0.15
                     if best_theta is not None else rng.randn(param_dim))

        t0 = time.perf_counter()
        r = rollout_policy(env_name, theta, n_episodes, seed + iteration + 100)
        timer["rollout_s"] += time.perf_counter() - t0
        history_params.append(theta)
        history_rewards.append(r)
        rewards.append(r)
        if r > best:
            best, best_theta = r, theta.copy()

    diagnostics["parse_failure_rate"] = (
        diagnostics["n_parse_failures"] / max(1, diagnostics["n_llm_calls"])
    )
    timer["total_s"] = time.perf_counter() - t_start
    return {"rewards": rewards[:max_iterations], "best_reward": float(best),
            "wallclock": timer, "diagnostics": diagnostics}


# ─────────────────────────────────────────────────────────────────────────────
# Hint ablation study (ProPS Figures 15-18)
# Systematically removes parts of the behavioral hint to measure each
# component's contribution. Three ablation conditions matching the paper:
#   ablation_1: remove the directional/action-specific guidance
#   ablation_2: remove the state-space description (env-specific)
#   ablation_3: empty hint (ProPS+ without any hint = baseline ProPS+)
# ─────────────────────────────────────────────────────────────────────────────

HINT_ABLATIONS = {
    "CartPole-v1": {
        "full": TASK_HINTS.get("CartPole-v1", ""),
        "ablation_1": "Hint: Balance the pole on the cart by applying appropriate forces.",
        "ablation_2": "Hint: The policy should respond to pole angle and angular velocity.",
        "ablation_3": "",  # no hint -- equivalent to plain ProPS+
    },
    "MountainCarContinuous-v0": {
        "full": TASK_HINTS.get("MountainCarContinuous-v0", ""),
        "ablation_1": "Hint: Build momentum to reach the goal at the top of the hill.",
        "ablation_2": "Hint: Push in the direction of current velocity.",
        "ablation_3": "",
    },
    "CliffWalking-v0": {
        "full": TASK_HINTS.get("CliffWalking-v0", "" ),
        "ablation_1": "Hint: Navigate to the goal while avoiding the cliff.",
        "ablation_2": "",
        "ablation_3": "",
    },
    "Reacher-v4": {
        "full": "Hint: Apply joint torques that guide the fingertip toward the target. "
                "Use shoulder torque for large movements and elbow torque for fine adjustments.",
        "ablation_1": "Hint: Move the arm to reach the target position.",
        "ablation_2": "",
        "ablation_3": "",
    },
    "Walker2d-v4": {
        "full": "Hint: Coordinate left and right limb movements in alternating patterns. "
                "Use action[3] when the torso leans forward; balance action[2] and action[4]/action[5] "
                "to stabilize and maintain walking.",
        "ablation_1": "Hint: Maintain alternating limb patterns for stable walking.",
        "ablation_2": "",
        "ablation_3": "",
    },
}
# Fill in TASK_HINTS for envs not in HINT_ABLATIONS
for _env, _hint in TASK_HINTS.items():
    if _env not in HINT_ABLATIONS:
        HINT_ABLATIONS[_env] = {"full": _hint, "ablation_1": "", "ablation_2": "", "ablation_3": ""}


def sweep_hint_ablations(
    env_name, param_dim, seed, max_iterations,
    ablations=("full", "ablation_1", "ablation_2", "ablation_3"),
    ollama_model="qwen2.5:7b",
    ollama_host="http://localhost:11434",
    n_episodes=5,
    max_history=20,
):
    """
    Runs ProPS+ with progressively stripped hints, matching Figures 15-18.
    Each condition uses the same environment description (ProPS+) but
    varies which components of the behavioral hint are included.
    """
    env_ablations = HINT_ABLATIONS.get(env_name, {})
    results = {}
    for abl_name in ablations:
        hint_text = env_ablations.get(abl_name, "")
        label = f"props_plus_hint_{abl_name}" if hint_text else "props_plus_no_hint"

        desc = TASK_DESCRIPTIONS.get(env_name, "")
        sys_prompt = (
            f"Environment Description:\n{desc}\n\n"
            f"You are a good global optimizer, helping me find policy parameters that "
            f"maximize reward. Respond with ONLY a JSON array of {param_dim} floats.\n"
            f"No explanation, markdown, or extra text."
        )
        if hint_text:
            sys_prompt += f"\n\n{hint_text}"

        log.info(f"Hint ablation: {abl_name} for {env_name}")
        r = _run_with_custom_system_prompt(
            env_name, param_dim, seed, max_iterations,
            system_prompt=sys_prompt, max_history=max_history,
            ollama_model=ollama_model, ollama_host=ollama_host,
            n_episodes=n_episodes,
        )
        r["diagnostics"]["hint_ablation"] = abl_name
        r["diagnostics"]["hint_text"] = hint_text[:80] + "..." if len(hint_text) > 80 else hint_text
        results[abl_name] = r
        log.info(f"  {abl_name}: best={r['best_reward']:.2f} mean={np.mean(r['rewards']):.2f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# In-context reasoning / justification mode (ProPS Figure 14, Appendix I)
# The LLM is asked to provide textual reasoning for each proposal, then
# the numerical vector is extracted from its response. This produces
# interpretable traces matching Figure 14b's episode justifications.
# ─────────────────────────────────────────────────────────────────────────────

JUSTIFICATION_SYSTEM_PROMPT = """You are a policy optimizer with reasoning capability.
At each iteration, I will provide the history of parameter vectors and their rewards.
You must:
1. First, briefly explain your reasoning for the next proposal (1-2 sentences starting with "Reasoning: ").
2. Then output the proposed parameter vector as a JSON array of {param_dim} floats on a new line.

Example response format:
Reasoning: I observed that increasing params[2] correlates with higher reward, so I will explore higher values there while keeping other parameters near the best-seen configuration.
[0.5, -0.3, 1.2, ...]

{description}"""


def run_with_justification(
    env_name, param_dim, seed, max_iterations,
    ollama_model="qwen2.5:7b",
    ollama_host="http://localhost:11434",
    n_episodes=5,
    max_history=20,
    timeout=240,
):
    """
    ProPS with in-context justification (Figure 14, Appendix I).
    Returns the same reward dict plus a 'reasoning_trace' list of the
    LLM's textual justifications per iteration -- this is the
    interpretability data that Figure 14b visualizes.
    """
    import requests

    desc = TASK_DESCRIPTIONS.get(env_name, "")
    system_prompt = JUSTIFICATION_SYSTEM_PROMPT.format(
        param_dim=param_dim,
        description=f"Environment: {desc}" if desc else "",
    )

    timer = {"rollout_s": 0.0, "optimizer_s": 0.0, "llm_call_s": 0.0}
    t_start = time.perf_counter()
    rng = np.random.RandomState(seed)
    history_params, history_rewards = [], []
    rewards = []
    best = -np.inf
    best_theta = None
    reasoning_trace = []
    diagnostics = {"n_llm_calls": 0, "n_parse_failures": 0}

    n_warmup = min(5, max_iterations)
    for i in range(n_warmup):
        theta = rng.randn(param_dim)
        t0 = time.perf_counter()
        r = rollout_policy(env_name, theta, n_episodes, seed + i)
        timer["rollout_s"] += time.perf_counter() - t0
        history_params.append(theta)
        history_rewards.append(r)
        rewards.append(r)
        if r > best:
            best, best_theta = r, theta.copy()
        reasoning_trace.append({"iteration": i, "reasoning": "warmup_random", "reward": r})

    for iteration in range(n_warmup, max_iterations):
        user_prompt = _build_user_prompt(
            history_params, history_rewards, iteration, max_iterations, max_history
        )
        diagnostics["n_llm_calls"] += 1
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{ollama_host}/api/chat",
                json={"model": ollama_model,
                      "messages": [{"role": "system", "content": system_prompt},
                                   {"role": "user", "content": user_prompt}],
                      "options": {"temperature": 1.0}, "stream": False},
                timeout=timeout,
            )
            resp.raise_for_status()
            text = resp.json()["message"]["content"]
            timer["llm_call_s"] += time.perf_counter() - t0
        except Exception as e:
            log.warning(f"LLM call failed: {e}")
            text = ""

        # Extract reasoning (before the JSON array) and the numerical proposal
        reasoning = ""
        import re as _re
        reasoning_match = _re.search(r"Reasoning:\s*(.+?)(?=\[|$)", text, _re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        theta = _parse_response(text, param_dim)
        if theta is None:
            diagnostics["n_parse_failures"] += 1
            theta = (best_theta + rng.randn(param_dim) * 0.15
                     if best_theta is not None else rng.randn(param_dim))

        t0 = time.perf_counter()
        r = rollout_policy(env_name, theta, n_episodes, seed + iteration + 100)
        timer["rollout_s"] += time.perf_counter() - t0
        history_params.append(theta)
        history_rewards.append(r)
        rewards.append(r)
        if r > best:
            best, best_theta = r, theta.copy()
        reasoning_trace.append({"iteration": iteration, "reasoning": reasoning, "reward": r})

    diagnostics["parse_failure_rate"] = (
        diagnostics["n_parse_failures"] / max(1, diagnostics["n_llm_calls"])
    )
    timer["total_s"] = time.perf_counter() - t_start
    return {
        "rewards": rewards[:max_iterations],
        "best_reward": float(best),
        "wallclock": timer,
        "diagnostics": diagnostics,
        "reasoning_trace": reasoning_trace,
    }
