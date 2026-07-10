"""
Corrected ablations.py helper functions, rewritten against JEPAAgent's
ACTUAL API (confirmed via grep on agent/jepa_agent.py):

  agent.history          -> HistoryBuffer instance
    .to_tensors(device)  -> (params_t, rewards_t)
    .get_stats()         -> (mean, std)  [numpy arrays over param_dim]
    .get_desired_reward(pct) -> float

  agent.context_encoder(params_norm, rewards_t) -> s_x  (1, d_model)
  agent.predictor(s_x, desired_r_t)             -> s_y_hat (1, d_model)
  agent.decoder(s_y_hat)                        -> theta_norm (1, param_dim)

This replaces the WRONG guessed API (_encode_history(), agent.noise_std
used as the noise tensor scale incorrectly, etc.) with the verified one,
mirroring propose()'s own internal logic exactly (agent/jepa_agent.py
lines 206-260ish) so the only difference in each ablated variant is the
CANDIDATE SELECTION RULE, not the encoding pipeline.
"""

import numpy as np
import torch
import logging

log = logging.getLogger(__name__)


def _encode_current_context(agent):
    """
    Mirrors the encoding steps at the start of JEPAAgent.propose(), exactly.
    Returns (s_x, mean, std, desired_r_t) ready for predictor/decoder calls.
    Returns None if history is empty (caller should fall back to random).
    """
    if len(agent.history) == 0:
        return None

    params_t, rewards_t = agent.history.to_tensors(agent.device)
    mean, std = agent.history.get_stats()

    params_norm = (
        params_t - torch.tensor(mean, device=agent.device, dtype=torch.float32)
    ) / torch.tensor(std, device=agent.device, dtype=torch.float32)

    s_x = agent.context_encoder(params_norm, rewards_t)  # (1, d_model)

    r_des = agent.history.get_desired_reward(agent.desired_pct)
    desired_r_t = torch.tensor([[r_des]], dtype=torch.float32, device=agent.device)

    return s_x, mean, std, desired_r_t


def _decode_candidate(agent, s_x, noise_scale, desired_r_t, mean, std):
    """One candidate: perturb s_x, predict, decode to theta. Returns
    (theta: np.ndarray, s_y_hat: torch.Tensor) so callers can score s_y_hat
    however they like (cosine sim, norm, etc.) without re-running predictor."""
    noise = torch.randn_like(s_x) * noise_scale
    s_x_noisy = s_x + noise
    s_y_hat = agent.predictor(s_x_noisy, desired_r_t)          # (1, d_model)
    theta_norm = agent.decoder(s_y_hat).squeeze(0).detach().cpu().numpy()
    theta = theta_norm * std + mean
    return theta, s_y_hat


def propose_best_of_n_fixed(agent, n_candidates, rng):
    """
    CORRECTED VERSION (replaces the buggy _propose_best_of_n in ablations.py).
    Decode n candidates, score each by predicted-latent norm as a simple
    confidence proxy, pick max. Mirrors propose()'s noise-scale schedule
    exactly so the only ablated variable is the selection rule.
    """
    ctx = _encode_current_context(agent)
    if ctx is None:
        return rng.randn(agent.param_dim)
    s_x, mean, std, desired_r_t = ctx

    best_theta, best_score = None, -np.inf
    for i in range(n_candidates):
        noise_scale = agent.noise_std * (1.0 + i * 2.0 / max(1, n_candidates - 1))
        theta, s_y_hat = _decode_candidate(agent, s_x, noise_scale, desired_r_t, mean, std)
        score = float(torch.norm(s_y_hat).item())
        if score > best_score:
            best_score, best_theta = score, theta

    return best_theta if best_theta is not None else rng.randn(agent.param_dim)


def propose_random_decode_fixed(agent, n_candidates, rng):
    """
    CORRECTED VERSION (replaces the buggy _propose_random_decode).
    Decode n candidates, pick uniformly at random — sanity floor for the
    selection-rule ablation.
    """
    ctx = _encode_current_context(agent)
    if ctx is None:
        return rng.randn(agent.param_dim)
    s_x, mean, std, desired_r_t = ctx

    candidates = []
    for i in range(n_candidates):
        noise_scale = agent.noise_std * (1.0 + i * 2.0 / max(1, n_candidates - 1))
        theta, _ = _decode_candidate(agent, s_x, noise_scale, desired_r_t, mean, std)
        candidates.append(theta)

    idx = rng.randint(0, len(candidates))
    return candidates[idx]


def run_latent_space_cem_fixed(env_name, param_dim, seed, max_iterations,
                                pretrained_ckpt="", n_eval_episodes=10,
                                popsize=20, elite_frac=0.2, device="cpu"):
    """
    CORRECTED VERSION of run_latent_space_cem, using the real JEPAAgent API.
    Runs CEM's elite-resampling update directly in JEPA's latent space:
    samples latent perturbations, decodes each to theta via agent.decoder,
    evaluates via rollout, fits a new Gaussian to elite LATENTS (not elite
    thetas), and resamples.
    """
    import sys, os, time
    sys.path.insert(0, os.getcwd())
    from agent.jepa_agent import JEPAAgent
    from runner.jepa_runner import rollout_policy

    np.random.seed(seed)
    torch.manual_seed(seed)

    agent = JEPAAgent(param_dim=param_dim, device=device)
    if pretrained_ckpt:
        try:
            ckpt = torch.load(pretrained_ckpt, map_location=device, weights_only=False)
            sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            agent.load_state_dict(sd, strict=False)
        except Exception as e:
            log.warning(f"checkpoint load failed: {e}")

    rng = np.random.RandomState(seed)
    rewards = []
    best = -np.inf
    t0 = time.perf_counter()

    n_warmup = max(10, param_dim)
    for i in range(n_warmup):
        theta = rng.randn(param_dim)
        r = rollout_policy(env_name, theta, n_eval_episodes, seed + i)
        agent.add_to_history(theta, r)
        rewards.append(r)
        best = max(best, r)

    ctx = _encode_current_context(agent)
    if ctx is None:
        # Should not happen after warmup, but guard anyway
        return {"rewards": rewards[:max_iterations], "best_reward": float(best),
                "wallclock": {"total_s": time.perf_counter() - t0}}
    s_x, mean, std, _ = ctx
    d_model = s_x.shape[-1]

    latent_mean = np.zeros(d_model)
    latent_std = np.ones(d_model) * 0.5
    n_elite = max(1, int(popsize * elite_frac))

    iteration = n_warmup
    while iteration < max_iterations:
        n_samples = min(popsize, max_iterations - iteration)
        latents = latent_mean[None, :] + rng.randn(n_samples, d_model) * latent_std[None, :]

        desired_r = (max(rewards[-20:]) if rewards else 0.0) * 1.05
        desired_r_t = torch.tensor([[desired_r]], dtype=torch.float32, device=device)

        sample_rewards = []
        for i in range(n_samples):
            z = torch.tensor(latents[i:i+1], dtype=torch.float32, device=device)
            s_y_hat = agent.predictor(z, desired_r_t)
            theta_norm = agent.decoder(s_y_hat).squeeze(0).detach().cpu().numpy()
            theta = theta_norm * std + mean

            r = rollout_policy(env_name, theta, n_eval_episodes, seed + iteration)
            agent.add_to_history(theta, r)
            rewards.append(r)
            sample_rewards.append(r)
            best = max(best, r)
            iteration += 1
            if iteration >= max_iterations:
                break

        if len(sample_rewards) > 0:
            sample_rewards = np.array(sample_rewards)
            elite_idx = np.argsort(sample_rewards)[-min(n_elite, len(sample_rewards)):]
            elite_latents = latents[:len(sample_rewards)][elite_idx]
            latent_mean = elite_latents.mean(axis=0)
            latent_std = elite_latents.std(axis=0) + 1e-3

    return {
        "rewards": rewards[:max_iterations],
        "best_reward": float(best),
        "wallclock": {"total_s": time.perf_counter() - t0},
    }
