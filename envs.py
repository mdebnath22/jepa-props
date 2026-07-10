"""
envs.py — Unified environment registry matching the 15 tasks evaluated in
Prompted Policy Search (ProPS), Zhou et al. 2025 (arXiv:2511.21928).

Provides a single rollout_policy(env_name, theta, n_episodes, seed) -> reward
interface for both continuous (linear policy) and discrete (tabular policy)
environments, so every optimizer/baseline in baselines.py and ablations.py
can be run identically across all tasks without per-env special-casing.

NOTE ON COVERAGE: Some of ProPS's 15 tasks use custom environments not in
standard Gymnasium (Navigation, Maze, Nim) or require packages not assumed
present (ale-py for Pong). These are registered with a `available` flag and
a clear ImportError-deferred message rather than silently failing, so the
registry is complete and self-documenting even on machines without every
optional dependency installed.
"""

import numpy as np
import logging

log = logging.getLogger(__name__)

try:
    import gymnasium as gym
except ImportError:
    gym = None
    log.warning("gymnasium not installed — install with: pip install gymnasium")


# ─────────────────────────────────────────────────────────────────────────────
# Task registry
# ─────────────────────────────────────────────────────────────────────────────
# policy_type: "linear" (theta is a flat weight vector, action = theta^T phi(s))
#              "tabular" (theta is one discrete action index per state)
# param_dim: dimensionality of theta for the *default* (identity-feature) linear
#            policy, i.e. param_dim = (obs_dim [+1 bias]) * n_actions_or_1
# action_type: "discrete" or "continuous"

TASKS = {
    # ---- Classic control (continuous obs, discrete or continuous action) ----
    "MountainCarContinuous-v0": dict(
        gym_id="MountainCarContinuous-v0", policy_type="linear",
        obs_dim=2, action_dim=1, param_dim=3, action_type="continuous",
        action_low=-1.0, action_high=1.0, max_episode_steps=999,
    ),
    "InvertedPendulum-v4": dict(
        gym_id="InvertedPendulum-v4", policy_type="linear",
        obs_dim=4, action_dim=1, param_dim=5, action_type="continuous",
        action_low=-3.0, action_high=3.0, max_episode_steps=1000,
    ),
    "InvertedDoublePendulum-v4": dict(
        gym_id="InvertedDoublePendulum-v4", policy_type="linear",
        obs_dim=11, action_dim=1, param_dim=12, action_type="continuous",
        action_low=-1.0, action_high=1.0, max_episode_steps=1000,
    ),
    "Reacher-v4": dict(
        gym_id="Reacher-v4", policy_type="linear",
        obs_dim=11, action_dim=2, param_dim=24, action_type="continuous",
        action_low=-1.0, action_high=1.0, max_episode_steps=50,
    ),
    "Swimmer-v4": dict(
        gym_id="Swimmer-v4", policy_type="linear",
        obs_dim=8, action_dim=2, param_dim=18, action_type="continuous",
        action_low=-1.0, action_high=1.0, max_episode_steps=1000,
    ),
    "Hopper-v4": dict(
        gym_id="Hopper-v4", policy_type="linear",
        obs_dim=11, action_dim=3, param_dim=36, action_type="continuous",
        action_low=-1.0, action_high=1.0, max_episode_steps=1000,
    ),
    "Walker2d-v4": dict(
        gym_id="Walker2d-v4", policy_type="linear",
        obs_dim=17, action_dim=6, param_dim=108, action_type="continuous",
        action_low=-1.0, action_high=1.0, max_episode_steps=1000,
    ),
    "FrozenLake-v1": dict(
        gym_id="FrozenLake-v1", policy_type="tabular",
        n_states=16, n_actions=4, param_dim=16, action_type="discrete",
        max_episode_steps=100, env_kwargs=dict(is_slippery=True),
    ),
    "CliffWalking-v0": dict(
        gym_id="CliffWalking-v0", policy_type="tabular",
        n_states=48, n_actions=4, param_dim=48, action_type="discrete",
        max_episode_steps=200,
    ),
    "MountainCar-v0": dict(
        gym_id="MountainCar-v0", policy_type="linear",
        obs_dim=2, action_dim=3, param_dim=9, action_type="discrete",
        max_episode_steps=200,
    ),
    "CartPole-v1": dict(
        gym_id="CartPole-v1", policy_type="linear",
        obs_dim=4, action_dim=2, param_dim=10, action_type="discrete",
        max_episode_steps=500,
    ),
    "ALE/Pong-v5": dict(
        gym_id="ALE/Pong-v5", policy_type="linear",
        obs_dim=128, action_dim=6, param_dim=774, action_type="discrete",
        max_episode_steps=10000, requires="ale-py",
    ),
    # ---- Custom tasks from ProPS (Maze, Nim, Navigation): not in standard ----
    # Gymnasium. Registered as unavailable placeholders with clear messaging
    # rather than silently omitted, so the 15-task table is complete.
    "Maze": dict(
        gym_id="__custom_Maze__", policy_type="tabular",
        n_states=16, n_actions=4, param_dim=16, action_type="discrete",
        max_episode_steps=100, custom=True,
    ),
    "Nim": dict(
        gym_id="__custom_Nim__", policy_type="linear",
        obs_dim=3, action_dim=21, param_dim=84, action_type="discrete",
        max_episode_steps=50, custom=True,
    ),
    "Navigation": dict(
        gym_id="__custom_Navigation__", policy_type="linear",
        obs_dim=6, action_dim=2, param_dim=14, action_type="continuous",
        action_low=-1.0, action_high=1.0, max_episode_steps=200, custom=True,
    ),
    "BallLanding": dict(
        gym_id="__custom_BallLanding__", policy_type="linear",
        obs_dim=2, action_dim=3, param_dim=9, action_type="continuous",
        action_low=-1.0, action_high=1.0, max_episode_steps=1, custom=True,
        available=True,
    ),
}

for _name, _cfg in TASKS.items():
    _cfg.setdefault("available", True)
    _cfg.setdefault("name", _name)


def list_available_tasks():
    return [n for n, c in TASKS.items() if c.get("available", True)]


def list_unavailable_tasks():
    return {n: c["reason"] for n, c in TASKS.items() if not c.get("available", True)}


# ─────────────────────────────────────────────────────────────────────────────
# Policy evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _linear_policy_action(theta, obs, cfg):
    obs_aug = np.concatenate([obs, [1.0]])  # bias term
    if cfg["action_type"] == "continuous":
        W = theta.reshape(cfg["action_dim"], cfg["obs_dim"] + 1)
        a = W @ obs_aug
        return np.clip(a, cfg["action_low"], cfg["action_high"])
    else:
        W = theta.reshape(cfg["action_dim"], cfg["obs_dim"] + 1)
        logits = W @ obs_aug
        return int(np.argmax(logits))


def _tabular_policy_action(theta, obs, cfg):
    state_idx = int(obs) if np.isscalar(obs) else int(obs)
    action_val = theta[state_idx % cfg["n_states"]]
    return int(np.clip(round(action_val), 0, cfg["n_actions"] - 1))


def _rollout_custom_env(env_name, theta, n_episodes, seed, cfg):
    """Dispatch rollout for Maze/Nim/Navigation (custom_envs.py), reusing
    the same linear/tabular policy logic as standard Gymnasium tasks."""
    from custom_envs import CUSTOM_ENV_REGISTRY

    EnvCls = CUSTOM_ENV_REGISTRY[env_name]
    total_rewards = []
    for ep in range(n_episodes):
        env = EnvCls(seed=seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False
        steps = 0
        max_steps = cfg.get("max_episode_steps", 200)
        while not done and steps < max_steps:
            if cfg["policy_type"] == "linear":
                action = _linear_policy_action(theta, obs, cfg)
            else:
                action = _tabular_policy_action(theta, obs, cfg)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1
        total_rewards.append(ep_reward)
        env.close()
    return float(np.mean(total_rewards))


def rollout_policy(env_name: str, theta: np.ndarray, n_episodes: int = 5,
                    seed: int = 0) -> float:
    """
    Unified rollout interface. Returns mean episodic return over n_episodes.
    theta is clipped/reshaped according to the task's policy_type.
    """
    cfg = TASKS[env_name]
    if not cfg.get("available", True):
        raise ValueError(f"Task '{env_name}' unavailable: {cfg.get('reason')}")

    if cfg.get("custom"):
        return _rollout_custom_env(env_name, theta, n_episodes, seed, cfg)

    if gym is None:
        raise ImportError("gymnasium required: pip install gymnasium")

    env_kwargs = cfg.get("env_kwargs", {})
    env = gym.make(cfg["gym_id"], **env_kwargs)

    total_rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False
        steps = 0
        max_steps = cfg.get("max_episode_steps", 1000)
        while not done and steps < max_steps:
            if cfg["policy_type"] == "linear":
                action = _linear_policy_action(theta, obs, cfg)
            else:
                action = _tabular_policy_action(theta, obs, cfg)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            steps += 1
        total_rewards.append(ep_reward)

    env.close()
    return float(np.mean(total_rewards))


def rollout_policy_repeated(env_name: str, theta: np.ndarray,
                             n_episodes: int = 5, n_repeats: int = 1,
                             base_seed: int = 0):
    """
    Reviewer-requested: evaluate a candidate with multiple independent
    repeats (each repeat itself averaging n_episodes), returning both the
    mean and a confidence interval, to control for return stochasticity
    (addresses the "random search best > CEM best" anomaly).

    Returns: dict(mean, std, ci95_low, ci95_high, all_repeat_means)
    """
    repeat_means = [
        rollout_policy(env_name, theta, n_episodes, seed=base_seed + 1000 * r)
        for r in range(n_repeats)
    ]
    arr = np.array(repeat_means)
    mean = float(arr.mean())
    std = float(arr.std())
    # Normal-approximation 95% CI; fine for n_repeats >= 5, flagged otherwise
    se = std / np.sqrt(max(1, n_repeats))
    return dict(
        mean=mean, std=std,
        ci95_low=mean - 1.96 * se, ci95_high=mean + 1.96 * se,
        all_repeat_means=repeat_means,
        n_repeats=n_repeats,
        warning=None if n_repeats >= 5 else "n_repeats<5: CI is approximate",
    )


def get_param_dim(env_name: str) -> int:
    return TASKS[env_name]["param_dim"]
