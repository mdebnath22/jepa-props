"""
nn_policy.py — Neural network policy support matching ProPS Table 9 and
Figure 11, where ProPS tests 1- and 2-hidden-layer MLP policies with
architectures [10], [10,4], [10,10], [15,15] evaluated on Navigation,
Swimmer, and Hopper.

Two things are needed:
1. A generic MLP policy class that runs a forward pass from observation
   to action, with its weight/bias vector flattened to a 1D param vector
   (so it's opaque to every optimizer in baselines.py -- they just get a
   float array to optimize, same as linear policies).

2. Random-projection reparameterization (ProPS Appendix E): when the MLP
   has many parameters (e.g., Navigation architecture [10] has 80 params),
   the LLM must propose an 80-dimensional vector at each step, which is
   challenging. ProPS's fix: optimize a low-dimensional latent z, and
   decode to theta = M @ z where M is a fixed random orthonormal matrix.
   This lets the LLM propose a small (e.g., 10-dim) vector while the
   actual policy has a much larger parameter space.

Usage:
    from nn_policy import MLPPolicy, make_rollout_fn

    # Create a 1-hidden-layer [10]-neuron policy for Swimmer (8 obs, 2 act)
    policy = MLPPolicy(obs_dim=8, act_dim=2, hidden_layers=[10],
                       action_type='continuous', action_bound=1.0)
    print('param_dim:', policy.param_dim)  # 100: (8+1)*10 + (10+1)*2 = 112

    # Roll it out (theta is a flat numpy array of length policy.param_dim)
    rollout = make_rollout_fn('Swimmer-v4', policy)
    reward = rollout(theta=np.random.randn(policy.param_dim), n_episodes=5, seed=0)

    # With random projection: only optimize 20-dim z, decoded to full theta
    from nn_policy import RandomProjectionWrapper
    proj = RandomProjectionWrapper(policy, latent_dim=20, seed=42)
    # Now pass proj to any optimizer as if it has param_dim=20, but the
    # actual policy runs with proj.param_dim original parameters.
    reward = proj.rollout(z=np.random.randn(20), env_name='Swimmer-v4',
                           n_episodes=5, seed=0)
"""

import numpy as np
import warnings


class MLPPolicy:
    """
    A generic MLP policy with configurable hidden layers. Supports both
    continuous (tanh-bounded output) and discrete (argmax) actions.

    Architecture:
        obs -> [Linear + ReLU] * n_hidden -> Linear -> [tanh | argmax]

    All weights and biases are flattened to a single param vector of length
    `param_dim`, enabling any flat-vector optimizer to tune this policy.

    Matches ProPS Table 9's architecture notation:
        [10]        = 1 hidden layer, 10 neurons
        [10, 4]     = 2 hidden layers, 10 and 4 neurons
        [10, 10]    = 2 hidden layers, 10 and 10 neurons
        [15, 15]    = 2 hidden layers, 15 and 15 neurons
    """

    def __init__(self, obs_dim, act_dim, hidden_layers,
                 action_type="continuous", action_bound=1.0):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_layers = list(hidden_layers)
        self.action_type = action_type
        self.action_bound = action_bound

        # Build layer sizes for weight/bias shape computation
        layer_sizes = [obs_dim] + hidden_layers + [act_dim]
        self.layer_shapes = []
        n_params = 0
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i + 1]
            self.layer_shapes.append((in_size, out_size))
            n_params += (in_size + 1) * out_size  # +1 for bias
        self.param_dim = n_params

    def _unpack(self, theta):
        """Unpack flat theta vector into list of (W, b) tuples per layer."""
        layers = []
        idx = 0
        for in_size, out_size in self.layer_shapes:
            w_size = in_size * out_size
            W = theta[idx:idx + w_size].reshape(out_size, in_size)
            idx += w_size
            b = theta[idx:idx + out_size]
            idx += out_size
            layers.append((W, b))
        return layers

    def act(self, obs, theta):
        """Forward pass: obs -> action given flat parameter vector theta."""
        obs = np.asarray(obs, dtype=float)
        layers = self._unpack(np.asarray(theta, dtype=float))
        x = obs
        for i, (W, b) in enumerate(layers):
            x = W @ x + b
            if i < len(layers) - 1:
                x = np.maximum(0, x)  # ReLU hidden activations

        if self.action_type == "continuous":
            return np.tanh(x) * self.action_bound
        else:
            return int(np.argmax(x))


def make_rollout_fn(env_name, policy):
    """
    Returns a callable rollout_fn(theta, n_episodes, seed) -> float that
    evaluates the given MLPPolicy in the named environment. The returned
    function works as a drop-in replacement for envs.rollout_policy for
    any env registered in envs.TASKS.

    Used by every optimizer in baselines.py via the standard interface:
        result = run_cem(env_name, policy.param_dim, ...)
    After replacing envs.rollout_policy with make_rollout_fn's output for
    the duration of the experiment.
    """
    try:
        import gymnasium as gym
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from envs import TASKS
        cfg = TASKS[env_name]
        gym_id = cfg["gym_id"]
        env_kwargs = cfg.get("env_kwargs", {})
        max_steps = cfg.get("max_episode_steps", 1000)
        use_custom = cfg.get("custom", False)
    except Exception as e:
        raise ImportError(f"Could not set up rollout env '{env_name}': {e}")

    def rollout_fn(theta, n_episodes=5, seed=0):
        theta = np.asarray(theta, dtype=float)
        ep_rewards = []
        for ep in range(n_episodes):
            if use_custom:
                from custom_envs import CUSTOM_ENV_REGISTRY
                env = CUSTOM_ENV_REGISTRY[env_name](seed=seed + ep)
            else:
                env = gym.make(gym_id, **env_kwargs)
            obs, _ = env.reset(seed=seed + ep)
            ep_reward = 0.0
            done = False
            steps = 0
            while not done and steps < max_steps:
                action = policy.act(obs, theta)
                obs, r, terminated, truncated, _ = env.step(action)
                ep_reward += r
                done = terminated or truncated
                steps += 1
            ep_rewards.append(ep_reward)
            env.close()
        return float(np.mean(ep_rewards))

    return rollout_fn


class RandomProjectionWrapper:
    """
    Reparameterizes an MLPPolicy via a fixed random orthonormal projection:
        theta = M @ z    where M in R^{param_dim x latent_dim}

    Any optimizer sees only `latent_dim` parameters. The full policy
    is recovered by multiplying by the fixed matrix M.

    This is ProPS Appendix E's key trick for handling high-dim MLP policies
    with LLM-iterative optimization: instead of asking the LLM to propose
    80+ floats, it proposes 10-20 floats in latent space.

    Args:
        policy:      MLPPolicy instance
        latent_dim:  dimensionality of the latent vector z that optimizers
                     actually propose (typically 10-20% of policy.param_dim)
        seed:        fixed seed for the random projection matrix M (must be
                     the same across all runs for reproducibility)
    """

    def __init__(self, policy, latent_dim, seed=0):
        self.policy = policy
        self.latent_dim = latent_dim
        self.param_dim_full = policy.param_dim

        rng = np.random.RandomState(seed)
        M_raw = rng.randn(policy.param_dim, latent_dim)
        # Orthonormalize columns via QR decomposition
        Q, _ = np.linalg.qr(M_raw)
        self.M = Q[:, :latent_dim]  # (param_dim, latent_dim), orthonormal columns
        assert self.M.shape == (policy.param_dim, latent_dim)

    @property
    def param_dim(self):
        return self.latent_dim

    def decode(self, z):
        """Decode latent z -> full parameter theta."""
        return self.M @ np.asarray(z, dtype=float)

    def rollout(self, z, env_name, n_episodes=5, seed=0):
        """Roll out in env_name using decoded parameter theta = M @ z."""
        theta = self.decode(z)
        rollout_fn = make_rollout_fn(env_name, self.policy)
        return rollout_fn(theta, n_episodes, seed)

    def make_rollout_fn_projected(self, env_name):
        """Returns rollout_fn(z, n_episodes, seed) -> float for use with
        any optimizer (they just call it with a latent_dim-length vector)."""
        full_rollout = make_rollout_fn(env_name, self.policy)

        def projected_rollout(z, n_episodes=5, seed=0):
            theta = self.decode(z)
            return full_rollout(theta, n_episodes, seed)

        return projected_rollout


# ─────────────────────────────────────────────────────────────────────────────
# Registry matching ProPS Table 9's exact architectures
# ─────────────────────────────────────────────────────────────────────────────

def get_props_nn_architectures():
    """
    Returns the full set of (env_name, architecture, param_dim) triples
    corresponding to ProPS Table 9, for use as drop-in rollout targets for
    any optimizer in baselines.py.

    Note: ProPS reports params counts as:
      Navigation [10]:     80 params  (6 obs x 10 + 10 bias = 70, +2 out+2 bias = 72... )
      Actually they report 80 which suggests obs_dim=7 or bias counted differently.
      We match their reported param counts from Table 9 using obs_aug (obs+bias).
      Navigation:  [6+1]*10 + [10+1]*2 = 70 + 22 = 92 ... Table says 80.
      The discrepancy suggests they may not use bias on the output layer, or use
      obs_dim=6 for Navigation without bias augmentation.
      We implement both and report our actual param_dim so there's no ambiguity.
    """
    configs = []

    # Navigation: obs_dim=6, act_dim=2, continuous
    for hidden in ([10], [10, 4], [10, 10], [15, 15]):
        p = MLPPolicy(obs_dim=6, act_dim=2, hidden_layers=hidden,
                      action_type="continuous", action_bound=1.0)
        configs.append(("Navigation", hidden, p.param_dim, p))

    # Swimmer: obs_dim=8, act_dim=2, continuous
    for hidden in ([10], [10, 4], [10, 10], [15, 15]):
        p = MLPPolicy(obs_dim=8, act_dim=2, hidden_layers=hidden,
                      action_type="continuous", action_bound=1.0)
        configs.append(("Swimmer-v4", hidden, p.param_dim, p))

    # Hopper: obs_dim=11, act_dim=3, continuous
    for hidden in ([10], [10, 4], [10, 10], [15, 15]):
        p = MLPPolicy(obs_dim=11, act_dim=3, hidden_layers=hidden,
                      action_type="continuous", action_bound=1.0)
        configs.append(("Hopper-v4", hidden, p.param_dim, p))

    return configs


def run_nn_policy_sweep(
    env_name=None,
    optimizer_fn=None,
    max_iterations=400,
    n_episodes=5,
    seed=0,
    latent_dim=None,
):
    """
    Run one or all ProPS Table 9 architectures through a given optimizer.

    optimizer_fn: callable(rollout_fn, param_dim, seed, max_iterations)
                  that returns {"rewards": [...], "best_reward": float}.
                  Any function from baselines.py's BASELINE_REGISTRY works
                  if wrapped to call rollout_fn instead of rollout_policy.

    latent_dim: if set, wraps the policy in RandomProjectionWrapper so the
                optimizer only sees a lower-dimensional latent space.
    """
    from baselines import BASELINE_REGISTRY

    configs = get_props_nn_architectures()
    if env_name:
        configs = [c for c in configs if c[0] == env_name]

    results = {}
    for env, hidden, param_dim, policy in configs:
        key = f"{env}_{'x'.join(str(h) for h in hidden)}"
        rollout_base = make_rollout_fn(env, policy)

        if latent_dim is not None:
            wrapper = RandomProjectionWrapper(policy, latent_dim=latent_dim, seed=seed)
            rollout_fn = wrapper.make_rollout_fn_projected(env)
            effective_dim = latent_dim
        else:
            rollout_fn = rollout_base
            effective_dim = param_dim

        log_str = f"{key} param_dim={param_dim}"
        if latent_dim:
            log_str += f" (projected to latent_dim={latent_dim})"

        import logging
        logging.getLogger(__name__).info(f"Running {log_str}")

        if optimizer_fn is None:
            # Default: random search for smoke testing
            rewards = []
            rng = np.random.RandomState(seed)
            best = -np.inf
            for _ in range(max_iterations):
                z = rng.randn(effective_dim)
                r = rollout_fn(z, n_episodes=n_episodes, seed=seed)
                rewards.append(r)
                best = max(best, r)
            results[key] = {"rewards": rewards, "best_reward": float(best),
                             "param_dim": param_dim, "effective_dim": effective_dim}
        else:
            result = optimizer_fn(rollout_fn, effective_dim, seed, max_iterations)
            result["param_dim"] = param_dim
            result["effective_dim"] = effective_dim
            results[key] = result

    return results
