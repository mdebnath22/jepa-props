"""
jepa_deep_analysis.py — Nine novel deep analysis experiments probing
what optimization algorithm JEPA has implicitly learned.

All experiments run against existing pretrained checkpoints.
No retraining required.

Experiments:
  E1.  Latent trajectory analysis: PCA of latent vectors during optimization
       colored by reward -- does JEPA organize optimization geometrically?
  E2.  Pre-evaluation reward prediction: can latent norm/variance predict
       reward BEFORE rollout? Does JEPA have internal optimization confidence?
  E3/E15. Emergent optimizer behavior: compare JEPA's proposal directions to
       GD, CEM, PSO, ES -- what optimizer has JEPA implicitly learned?
  E6.  Explore vs exploit: track exploration radius over optimization time --
       does JEPA automatically anneal exploration?
  E8.  Forgetting: remove early history or best-known point -- what info
       does JEPA actually need from its history?
  E9.  Reverse history: feed optimization history backwards (worst->best
       instead of best->worst) -- did JEPA learn temporal structure?
  E13. Representation probing: freeze JEPA, train tiny linear probe to
       predict reward/improvement rank from latent vectors.
  E16. Acquisition function: do JEPA proposals maximize EI/UCB/PI?
       Does JEPA implicitly learn a Bayesian acquisition function?
  E18. Latent interpolation: interpolate between two optimization histories
       in latent space -- does decoded policy perform "halfway"?
  E19. Phase detection: cluster latent vectors across an optimization run --
       do exploration/refinement/convergence phases emerge naturally?

USAGE:
    python jepa_deep_analysis.py \
        --ckpt checkpoints/jepa_pretrained/jepa_pretrained.pt \
        --env CartPole-v1 --param_dim 8 \
        --out_dir results/deep_analysis \
        --experiments all
"""

import os, sys, json, argparse, logging, time
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
sys.path.insert(0, os.getcwd())


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_agent(ckpt_path, param_dim, device="cpu"):
    import torch
    from agent.jepa_agent import JEPAAgent
    agent = JEPAAgent(param_dim=param_dim, device=device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    agent.load_state_dict(ckpt.get("model", ckpt), strict=False)
    agent.eval()
    return agent


def run_optimization_trace(agent, env_name, param_dim, seed=0,
                             max_iterations=200, n_eval_episodes=5,
                             device="cpu"):
    """
    Runs single-trajectory JEPA optimization and returns full trace:
      thetas, rewards, latent_vectors, proposed_thetas
    at every iteration. This is the raw material for E1-E3, E6, E13, E16.
    """
    import torch
    from runner.jepa_runner import rollout_policy
    from agent.jepa_agent import JEPAAgent

    rng = np.random.RandomState(seed)
    thetas, rewards, latents, proposed = [], [], [], []

    n_warmup = max(8, param_dim)
    for i in range(n_warmup):
        theta = rng.randn(param_dim)
        r = rollout_policy(env_name, theta, n_eval_episodes, seed + i)
        agent.add_to_history(theta, r)
        thetas.append(theta.copy())
        rewards.append(r)

    for t in range(max_iterations - n_warmup):
        # Capture latent BEFORE proposing
        with torch.no_grad():
            if len(agent.history) > 0:
                params_t, rewards_t = agent.history.to_tensors(device)
                mean, std = agent.history.get_stats()
                params_norm = (params_t -
                               torch.tensor(mean, dtype=torch.float32)) / \
                              torch.tensor(std + 1e-6, dtype=torch.float32)
                s_x = agent.context_encoder(params_norm, rewards_t)
                latent = s_x.squeeze(0).cpu().numpy()
            else:
                latent = np.zeros(128)
        latents.append(latent)

        theta = agent.propose()
        proposed.append(theta.copy())
        r = rollout_policy(env_name, theta, n_eval_episodes, seed + n_warmup + t)
        agent.add_to_history(theta, r)
        thetas.append(theta.copy())
        rewards.append(r)

    return {
        "thetas": np.array(thetas),
        "rewards": np.array(rewards),
        "latents": np.array(latents),
        "proposed": np.array(proposed),
        "best_reward": float(max(rewards)),
        "mean_reward": float(np.mean(rewards)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# E1: Latent trajectory analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_latent_trajectory(ckpt_path, env_name, param_dim,
                               n_seeds=5, max_iterations=200,
                               n_eval_episodes=5, device="cpu"):
    """
    E1: Do nearby latent vectors correspond to similar rewards?
    Does latent movement predict reward improvement?
    Does optimization become a smooth trajectory in latent space?

    Key metrics:
    - PCA explained variance (how compressed is the trajectory?)
    - Latent displacement per step vs reward improvement (correlation)
    - Trajectory smoothness (mean cosine similarity between successive latents)
    - Does latent norm grow, shrink, or stay constant during optimization?
    """
    results = {}
    all_latents, all_rewards, all_displacements, all_reward_diffs = [], [], [], []

    for seed in range(n_seeds):
        agent = load_agent(ckpt_path, param_dim, device)
        trace = run_optimization_trace(agent, env_name, param_dim,
                                        seed=seed, max_iterations=max_iterations,
                                        n_eval_episodes=n_eval_episodes,
                                        device=device)
        latents = trace["latents"]  # (T, d_model)
        rewards = trace["rewards"][len(trace["rewards"]) - len(latents):]  # align

        # Latent displacement between successive steps
        displacements = np.linalg.norm(np.diff(latents, axis=0), axis=1)
        reward_diffs = np.diff(rewards)

        # Cosine similarity between successive latents (smoothness)
        cos_sims = []
        for i in range(1, len(latents)):
            a, b = latents[i-1], latents[i]
            cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            cos_sims.append(float(cos))

        # Latent norms over time
        norms = np.linalg.norm(latents, axis=1)

        # Reward-latent displacement correlation
        r_disp = float(np.corrcoef(displacements, reward_diffs)[0, 1]) \
            if len(displacements) > 2 else 0.0

        results[f"seed_{seed}"] = {
            "latent_displacement_reward_corr": r_disp,
            "mean_cosine_similarity": float(np.mean(cos_sims)),
            "latent_norm_trajectory": norms.tolist(),
            "reward_trajectory": rewards.tolist(),
            "best_reward": trace["best_reward"],
        }
        all_latents.append(latents)
        all_rewards.append(rewards)
        all_displacements.extend(displacements.tolist())
        all_reward_diffs.extend(reward_diffs.tolist())
        log.info(f"E1 seed={seed}: disp↔reward_r={r_disp:.3f} "
                 f"cos_sim={np.mean(cos_sims):.3f} best={trace['best_reward']:.1f}")

    # Aggregate PCA across all seeds
    all_L = np.vstack(all_latents)
    L_centered = all_L - all_L.mean(axis=0)
    cov = L_centered.T @ L_centered / len(all_L)
    eigvals, _ = np.linalg.eigh(cov)
    eigvals = eigvals[::-1]
    explained = eigvals / eigvals.sum()

    # Overall displacement-reward correlation
    overall_corr = float(np.corrcoef(all_displacements, all_reward_diffs)[0, 1]) \
        if len(all_displacements) > 2 else 0.0

    results["aggregate"] = {
        "pca_explained_variance_top5": explained[:5].tolist(),
        "displacement_reward_corr_overall": overall_corr,
        "interpretation": (
            "High corr(displacement, reward_diff) means latent movement "
            "predicts reward improvement. High PCA concentration means "
            "optimization follows a low-dimensional manifold in latent space."
        )
    }
    log.info(f"E1 aggregate: PCA top-5={[f'{v:.3f}' for v in explained[:5]]} "
             f"disp↔reward_r={overall_corr:.3f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# E2: Pre-evaluation reward prediction
# ─────────────────────────────────────────────────────────────────────────────

def analyze_prereward_prediction(ckpt_path, env_name, param_dim,
                                  n_seeds=5, max_iterations=200,
                                  n_eval_episodes=5, device="cpu"):
    """
    E2: Before evaluating theta, JEPA already produced a latent.
    Can latent properties predict the upcoming reward?

    Metrics computed BEFORE rollout, correlated with actual reward:
    - ||latent|| (norm of context encoding)
    - desired_reward (the conditioning target JEPA was given)
    - latent cosine similarity to best-seen latent so far
    - latent variance (uncertainty proxy)

    If any of these correlate with actual reward, JEPA has an implicit
    internal notion of optimization confidence.
    """
    import torch

    all_norms, all_desired_r, all_cos_to_best, all_actual_r = [], [], [], []
    results = {}

    for seed in range(n_seeds):
        agent = load_agent(ckpt_path, param_dim, device)
        rng = np.random.RandomState(seed)

        norms, desired_rs, cos_to_bests, actual_rewards = [], [], [], []
        best_latent = None

        n_warmup = max(8, param_dim)
        for i in range(n_warmup):
            theta = rng.randn(param_dim)
            from runner.jepa_runner import rollout_policy
            r = rollout_policy(env_name, theta, n_eval_episodes, seed + i)
            agent.add_to_history(theta, r)

        for t in range(max_iterations - n_warmup):
            with torch.no_grad():
                params_t, rewards_t = agent.history.to_tensors(device)
                mean, std = agent.history.get_stats()
                params_norm = (params_t -
                               torch.tensor(mean, dtype=torch.float32)) / \
                              torch.tensor(std + 1e-6, dtype=torch.float32)
                s_x = agent.context_encoder(params_norm, rewards_t)
                latent = s_x.squeeze(0).cpu().numpy()

            norm = float(np.linalg.norm(latent))
            desired_r = float(agent.history.get_desired_reward(agent.desired_pct))

            if best_latent is not None:
                cos = float(np.dot(latent, best_latent) /
                            (norm * np.linalg.norm(best_latent) + 1e-8))
            else:
                cos = 0.0

            theta = agent.propose()
            r = rollout_policy(env_name, theta, n_eval_episodes,
                                seed + n_warmup + t)
            agent.add_to_history(theta, r)

            norms.append(norm)
            desired_rs.append(desired_r)
            cos_to_bests.append(cos)
            actual_rewards.append(r)

            if best_latent is None or r > max(actual_rewards[:-1], default=-np.inf):
                best_latent = latent.copy()

        corr_norm = float(np.corrcoef(norms, actual_rewards)[0, 1])
        corr_desired = float(np.corrcoef(desired_rs, actual_rewards)[0, 1])
        corr_cos = float(np.corrcoef(cos_to_bests, actual_rewards)[0, 1])

        results[f"seed_{seed}"] = {
            "corr_latent_norm_vs_reward": corr_norm,
            "corr_desired_r_vs_actual_r": corr_desired,
            "corr_cos_to_best_vs_reward": corr_cos,
        }
        all_norms.extend(norms)
        all_desired_r.extend(desired_rs)
        all_cos_to_best.extend(cos_to_bests)
        all_actual_r.extend(actual_rewards)
        log.info(f"E2 seed={seed}: r(norm,rew)={corr_norm:.3f} "
                 f"r(desired,actual)={corr_desired:.3f} "
                 f"r(cos_best,rew)={corr_cos:.3f}")

    results["aggregate"] = {
        "corr_norm_reward": float(np.corrcoef(all_norms, all_actual_r)[0, 1]),
        "corr_desired_actual": float(np.corrcoef(all_desired_r, all_actual_r)[0, 1]),
        "corr_cos_best_reward": float(np.corrcoef(all_cos_to_best, all_actual_r)[0, 1]),
        "interpretation": (
            "corr_cos_best_reward > 0.2 would mean JEPA implicitly guides "
            "proposals toward high-reward regions it has already seen. "
            "corr_desired_actual > 0.5 would mean the desired_reward "
            "conditioning is actually predictive of outcomes."
        )
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# E3/E15: Emergent optimizer behavior
# ─────────────────────────────────────────────────────────────────────────────

def analyze_emergent_optimizer(ckpt_path, env_name, param_dim,
                                n_seeds=3, max_iterations=200,
                                n_eval_episodes=5, device="cpu"):
    """
    E3/E15: What optimizer has JEPA implicitly learned?

    At each step, compute the direction JEPA proposes (theta_t - theta_{t-1})
    and compare its cosine similarity to:
    - Finite-difference gradient estimate (GD direction)
    - CEM update direction (mean of elites - current mean)
    - Random direction (baseline)

    If JEPA's direction consistently aligns with GD: it learned gradient descent.
    If it aligns with CEM: it learned evolution strategies.
    If it aligns with neither: it learned something novel.
    """
    from runner.jepa_runner import rollout_policy

    results = {}
    all_cos_gd, all_cos_cem, all_cos_rand = [], [], []

    for seed in range(n_seeds):
        agent = load_agent(ckpt_path, param_dim, device)
        rng = np.random.RandomState(seed)

        cos_gds, cos_cems, cos_rands = [], [], []

        # Warmup
        thetas, rewards = [], []
        n_warmup = max(8, param_dim)
        for i in range(n_warmup):
            theta = rng.randn(param_dim)
            r = rollout_policy(env_name, theta, n_eval_episodes, seed + i)
            agent.add_to_history(theta, r)
            thetas.append(theta)
            rewards.append(r)

        eps = 0.01  # finite difference step size

        for t in range(max_iterations - n_warmup):
            prev_theta = thetas[-1].copy()
            prev_r = rewards[-1]

            # JEPA's proposal direction
            jepa_theta = agent.propose()
            jepa_dir = jepa_theta - prev_theta
            jepa_norm = np.linalg.norm(jepa_dir) + 1e-8

            # --- GD direction (finite difference) ---
            gd_dir = np.zeros(param_dim)
            for d in range(min(param_dim, 8)):  # limit FD dims for speed
                theta_eps = prev_theta.copy()
                theta_eps[d] += eps
                r_eps = rollout_policy(env_name, theta_eps, 3, seed + t * 100 + d)
                gd_dir[d] = (r_eps - prev_r) / eps  # positive = reward gradient
            gd_norm = np.linalg.norm(gd_dir) + 1e-8

            # --- CEM direction: mean of top-k random proposals ---
            n_cem = 10
            cem_samples = prev_theta + rng.randn(n_cem, param_dim) * 0.5
            cem_rewards = [rollout_policy(env_name, s, 3, seed + t * 200 + i)
                           for i, s in enumerate(cem_samples)]
            n_elite = max(1, n_cem // 4)
            elite_idx = np.argsort(cem_rewards)[-n_elite:]
            cem_mean = cem_samples[elite_idx].mean(axis=0)
            cem_dir = cem_mean - prev_theta
            cem_norm = np.linalg.norm(cem_dir) + 1e-8

            # Random direction
            rand_dir = rng.randn(param_dim)
            rand_norm = np.linalg.norm(rand_dir) + 1e-8

            # Cosine similarities
            cos_gd = float(np.dot(jepa_dir, gd_dir) / (jepa_norm * gd_norm))
            cos_cem = float(np.dot(jepa_dir, cem_dir) / (jepa_norm * cem_norm))
            cos_rand = float(np.dot(jepa_dir, rand_dir) / (jepa_norm * rand_norm))

            cos_gds.append(cos_gd)
            cos_cems.append(cos_cem)
            cos_rands.append(cos_rand)

            # Evaluate JEPA proposal and continue
            r = rollout_policy(env_name, jepa_theta, n_eval_episodes,
                                seed + n_warmup + t)
            agent.add_to_history(jepa_theta, r)
            thetas.append(jepa_theta)
            rewards.append(r)

        results[f"seed_{seed}"] = {
            "mean_cos_gd": float(np.mean(cos_gds)),
            "mean_cos_cem": float(np.mean(cos_cems)),
            "mean_cos_random": float(np.mean(cos_rands)),
            "std_cos_gd": float(np.std(cos_gds)),
        }
        all_cos_gd.extend(cos_gds)
        all_cos_cems.extend(cos_cems)
        all_cos_rands.extend(cos_rands)
        log.info(f"E3 seed={seed}: cos_GD={np.mean(cos_gds):.3f} "
                 f"cos_CEM={np.mean(cos_cems):.3f} "
                 f"cos_rand={np.mean(cos_rands):.3f}")

    # Determine which optimizer JEPA most resembles
    agg_gd = float(np.mean(all_cos_gd))
    agg_cem = float(np.mean(all_cos_cems))
    agg_rand = float(np.mean(all_cos_rands))
    closest = max([("GD", agg_gd), ("CEM", agg_cem), ("Random", agg_rand)],
                  key=lambda x: x[1])

    results["aggregate"] = {
        "mean_cos_gd": agg_gd,
        "mean_cos_cem": agg_cem,
        "mean_cos_random": agg_rand,
        "closest_classical_optimizer": closest[0],
        "closest_similarity": closest[1],
        "interpretation": (
            f"JEPA's proposal directions most closely resemble {closest[0]} "
            f"(cos={closest[1]:.3f}). Values near 0 for all would mean JEPA "
            "learned a genuinely novel update rule not reducible to these classics."
        )
    }
    log.info(f"E3 aggregate: GD={agg_gd:.3f} CEM={agg_cem:.3f} "
             f"rand={agg_rand:.3f} → closest: {closest[0]}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# E6: Explore vs exploit over time
# ─────────────────────────────────────────────────────────────────────────────

def analyze_explore_exploit(ckpt_path, env_name, param_dim,
                              n_seeds=5, max_iterations=200,
                              n_eval_episodes=5, device="cpu"):
    """
    E6: Does JEPA automatically anneal exploration over time?
    Track exploration radius (mean distance from current best theta)
    and proposal diversity (std of proposed thetas) across iterations.
    """
    from runner.jepa_runner import rollout_policy

    results = {}
    all_radii, all_diversities = [], []

    for seed in range(n_seeds):
        agent = load_agent(ckpt_path, param_dim, device)
        rng = np.random.RandomState(seed)

        thetas, rewards = [], []
        n_warmup = max(8, param_dim)
        for i in range(n_warmup):
            theta = rng.randn(param_dim)
            r = rollout_policy(env_name, theta, n_eval_episodes, seed + i)
            agent.add_to_history(theta, r)
            thetas.append(theta)
            rewards.append(r)

        radii, diversities, window_rewards = [], [], []
        window = 10

        for t in range(max_iterations - n_warmup):
            theta = agent.propose()
            r = rollout_policy(env_name, theta, n_eval_episodes, seed + n_warmup + t)
            agent.add_to_history(theta, r)
            thetas.append(theta)
            rewards.append(r)

            best_idx = int(np.argmax(rewards))
            radius = float(np.linalg.norm(theta - thetas[best_idx]))
            radii.append(radius)

            if len(thetas) >= window:
                recent = np.array(thetas[-window:])
                diversity = float(recent.std(axis=0).mean())
                diversities.append(diversity)

        # Compute correlation between time step and exploration radius
        t_arr = np.arange(len(radii))
        corr_radius_time = float(np.corrcoef(t_arr, radii)[0, 1]) \
            if len(radii) > 2 else 0.0
        corr_div_time = float(np.corrcoef(
            np.arange(len(diversities)), diversities)[0, 1]) \
            if len(diversities) > 2 else 0.0

        results[f"seed_{seed}"] = {
            "corr_exploration_radius_vs_time": corr_radius_time,
            "corr_diversity_vs_time": corr_div_time,
            "radius_trajectory": radii[:50],  # first 50 for brevity
            "interpretation": (
                "Negative correlation = JEPA automatically reduces exploration "
                "over time (exploitation annealing). Positive = expands search."
            )
        }
        all_radii.extend(radii)
        all_diversities.extend(diversities)
        log.info(f"E6 seed={seed}: r(radius,t)={corr_radius_time:.3f} "
                 f"r(diversity,t)={corr_div_time:.3f}")

    t_all = np.arange(len(all_radii))
    overall_corr = float(np.corrcoef(t_all, all_radii)[0, 1]) \
        if len(all_radii) > 2 else 0.0
    results["aggregate"] = {
        "corr_exploration_vs_time": overall_corr,
        "auto_annealing": overall_corr < -0.1,
    }
    log.info(f"E6 aggregate: r(radius,t)={overall_corr:.3f} "
             f"auto_annealing={overall_corr < -0.1}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# E8: Forgetting experiments
# ─────────────────────────────────────────────────────────────────────────────

def analyze_forgetting(ckpt_path, env_name, param_dim,
                        n_seeds=3, max_iterations=200,
                        n_eval_episodes=5, device="cpu"):
    """
    E8: What information does JEPA actually use from its history?
    Three conditions:
    - full: normal optimization (control)
    - remove_best: after each step, remove the best-seen (theta, reward) pair
    - remove_early: after each step, remove the oldest half of history
    - best_only: keep only the single best-seen (theta, reward) pair
    """
    from runner.jepa_runner import rollout_policy
    from agent.jepa_agent import JEPAAgent
    import torch

    conditions = ["full", "remove_best", "remove_early", "best_only"]
    results = {}

    for condition in conditions:
        seed_bests = []
        for seed in range(n_seeds):
            agent = load_agent(ckpt_path, param_dim, device)
            rng = np.random.RandomState(seed)
            thetas, rewards = [], []

            n_warmup = max(8, param_dim)
            for i in range(n_warmup):
                theta = rng.randn(param_dim)
                r = rollout_policy(env_name, theta, n_eval_episodes, seed + i)
                agent.add_to_history(theta, r)
                thetas.append(theta)
                rewards.append(r)

            for t in range(max_iterations - n_warmup):
                theta = agent.propose()
                r = rollout_policy(env_name, theta, n_eval_episodes,
                                    seed + n_warmup + t)
                agent.add_to_history(theta, r)
                thetas.append(theta)
                rewards.append(r)

                # Apply the forgetting condition
                if condition == "remove_best" and len(agent.history) > 1:
                    best_idx = int(np.argmax(agent.history.rewards))
                    agent.history.params.pop(best_idx)
                    agent.history.rewards.pop(best_idx)

                elif condition == "remove_early" and len(agent.history) > 4:
                    n_remove = len(agent.history) // 2
                    agent.history.params = agent.history.params[n_remove:]
                    agent.history.rewards = agent.history.rewards[n_remove:]

                elif condition == "best_only":
                    best_idx = int(np.argmax(agent.history.rewards))
                    best_theta = agent.history.params[best_idx]
                    best_r = agent.history.rewards[best_idx]
                    agent.history.params = [best_theta]
                    agent.history.rewards = [best_r]

            seed_bests.append(float(max(rewards)))
            log.info(f"E8 {condition} seed={seed}: best={max(rewards):.1f}")

        results[condition] = {
            "mean_best": float(np.mean(seed_bests)),
            "std_best": float(np.std(seed_bests)),
        }

    results["interpretation"] = (
        "If remove_best ~ full: JEPA doesn't rely on knowing the best seen point. "
        "If best_only ~ full: JEPA only uses the best point, trajectory doesn't matter. "
        "If remove_early >> remove_best: recent history matters more than early history."
    )
    log.info(f"E8: full={results['full']['mean_best']:.1f} "
             f"rm_best={results['remove_best']['mean_best']:.1f} "
             f"rm_early={results['remove_early']['mean_best']:.1f} "
             f"best_only={results['best_only']['mean_best']:.1f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# E9: Reverse history
# ─────────────────────────────────────────────────────────────────────────────

def analyze_reverse_history(ckpt_path, env_name, param_dim,
                              n_seeds=3, max_iterations=200,
                              n_eval_episodes=5, device="cpu"):
    """
    E9: Feed optimization history backwards (worst→best instead of best→worst).
    If JEPA learned temporal optimization structure, reversal should hurt.
    If JEPA ignores order, reversal should have no effect.
    """
    from runner.jepa_runner import rollout_policy

    conditions = ["forward", "reversed", "shuffled"]
    results = {}

    for condition in conditions:
        seed_bests = []
        for seed in range(n_seeds):
            agent = load_agent(ckpt_path, param_dim, device)
            rng = np.random.RandomState(seed)
            warmup_thetas, warmup_rewards = [], []

            n_warmup = max(8, param_dim)
            for i in range(n_warmup):
                theta = rng.randn(param_dim)
                r = rollout_policy(env_name, theta, n_eval_episodes, seed + i)
                warmup_thetas.append(theta)
                warmup_rewards.append(r)

            # Add history in specified order
            if condition == "forward":
                for theta, r in zip(warmup_thetas, warmup_rewards):
                    agent.add_to_history(theta, r)
            elif condition == "reversed":
                for theta, r in zip(reversed(warmup_thetas), reversed(warmup_rewards)):
                    agent.add_to_history(theta, r)
            elif condition == "shuffled":
                idx = rng.permutation(len(warmup_thetas))
                for i in idx:
                    agent.add_to_history(warmup_thetas[i], warmup_rewards[i])

            rewards = warmup_rewards.copy()
            for t in range(max_iterations - n_warmup):
                theta = agent.propose()
                r = rollout_policy(env_name, theta, n_eval_episodes,
                                    seed + n_warmup + t)
                # Always add in forward order after warmup
                agent.add_to_history(theta, r)
                rewards.append(r)

            seed_bests.append(float(max(rewards)))
            log.info(f"E9 {condition} seed={seed}: best={max(rewards):.1f}")

        results[condition] = {
            "mean_best": float(np.mean(seed_bests)),
            "std_best": float(np.std(seed_bests)),
        }

    forward_perf = results["forward"]["mean_best"]
    reversed_perf = results["reversed"]["mean_best"]
    results["temporal_sensitivity"] = float(
        (forward_perf - reversed_perf) / (abs(forward_perf) + 1e-6))
    results["interpretation"] = (
        "temporal_sensitivity > 0.1: JEPA learned temporal order matters. "
        "~0: JEPA treats history as a set, not a sequence. "
        f"forward={forward_perf:.1f} reversed={reversed_perf:.1f}"
    )
    log.info(f"E9: forward={forward_perf:.1f} reversed={reversed_perf:.1f} "
             f"shuffled={results['shuffled']['mean_best']:.1f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# E13: Representation probing
# ─────────────────────────────────────────────────────────────────────────────

def analyze_representation_probing(ckpt_path, env_name, param_dim,
                                    n_seeds=3, max_iterations=200,
                                    n_eval_episodes=5, device="cpu"):
    """
    E13: Freeze JEPA, train tiny linear probes on latent vectors to predict:
    - actual reward (regression)
    - whether reward > mean (binary classification)
    - reward improvement (positive/negative next step)
    - reward rank (percentile)

    If probes work well, optimization-relevant information is linearly
    decodable from JEPA's latent representation.
    """
    from runner.jepa_runner import rollout_policy

    # Collect latents and labels
    all_latents, all_rewards, all_improvements = [], [], []
    for seed in range(n_seeds):
        agent = load_agent(ckpt_path, param_dim, device)
        trace = run_optimization_trace(agent, env_name, param_dim,
                                        seed=seed, max_iterations=max_iterations,
                                        n_eval_episodes=n_eval_episodes,
                                        device=device)
        latents = trace["latents"]
        rewards = trace["rewards"][len(trace["rewards"]) - len(latents):]
        improvements = np.diff(rewards, prepend=rewards[0])
        all_latents.append(latents)
        all_rewards.append(rewards)
        all_improvements.append(improvements)

    X = np.vstack(all_latents)
    y_reward = np.concatenate(all_rewards)
    y_improve = np.concatenate(all_improvements)
    y_rank = np.array([np.searchsorted(np.sort(y_reward), r) / len(y_reward)
                       for r in y_reward])
    y_above_mean = (y_reward > y_reward.mean()).astype(float)

    # 5-fold cross-validation with ridge regression
    n = len(X)
    fold_size = n // 5
    probe_results = {}

    for target_name, y in [("reward", y_reward), ("rank", y_rank),
                             ("above_mean", y_above_mean),
                             ("improvement", y_improve)]:
        cv_scores = []
        for fold in range(5):
            test_idx = slice(fold * fold_size, (fold + 1) * fold_size)
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False

            X_tr, X_te = X[train_mask], X[test_idx]
            y_tr, y_te = y[train_mask], y[test_idx]

            # Normalize
            mu_X = X_tr.mean(axis=0)
            std_X = X_tr.std(axis=0) + 1e-8
            X_tr_n = (X_tr - mu_X) / std_X
            X_te_n = (X_te - mu_X) / std_X

            # Ridge regression (closed form, alpha=1.0)
            d = X_tr_n.shape[1]
            w = np.linalg.solve(X_tr_n.T @ X_tr_n + np.eye(d),
                                 X_tr_n.T @ y_tr)
            y_pred = X_te_n @ w

            # Score: R² for regression, accuracy for binary
            if target_name == "above_mean":
                acc = float(((y_pred > 0.5) == (y_te > 0.5)).mean())
                cv_scores.append(acc)
            else:
                ss_res = ((y_te - y_pred) ** 2).sum()
                ss_tot = ((y_te - y_te.mean()) ** 2).sum() + 1e-8
                r2 = float(1 - ss_res / ss_tot)
                cv_scores.append(r2)

        probe_results[target_name] = {
            "cv_mean": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
        }
        log.info(f"E13 probe({target_name}): R²/acc={np.mean(cv_scores):.3f}±{np.std(cv_scores):.3f}")

    probe_results["interpretation"] = (
        "R²/acc >> 0 means optimization-relevant information is linearly "
        "decodable from JEPA's latent space. rank R² > 0.3 would be strong "
        "evidence that JEPA has learned a smooth reward landscape representation."
    )
    return probe_results


# ─────────────────────────────────────────────────────────────────────────────
# E16: Acquisition function analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_acquisition_function(ckpt_path, env_name, param_dim,
                                  n_seeds=3, max_iterations=100,
                                  n_eval_episodes=5, device="cpu"):
    """
    E16: Does JEPA implicitly maximize an acquisition function like
    Expected Improvement (EI), Upper Confidence Bound (UCB), or
    Probability of Improvement (PI)?

    Approach: at each step, compute what EI/UCB/PI would suggest
    (using the empirical reward distribution as the surrogate),
    and compare to what JEPA actually proposes.
    """
    from runner.jepa_runner import rollout_policy

    results = {}
    all_cos_ei, all_cos_ucb, all_cos_rand = [], [], []

    for seed in range(n_seeds):
        agent = load_agent(ckpt_path, param_dim, device)
        rng = np.random.RandomState(seed)
        thetas, rewards = [], []

        n_warmup = max(8, param_dim)
        for i in range(n_warmup):
            theta = rng.randn(param_dim)
            r = rollout_policy(env_name, theta, n_eval_episodes, seed + i)
            agent.add_to_history(theta, r)
            thetas.append(theta.copy())
            rewards.append(r)

        cos_eis, cos_ucbs, cos_rands_list = [], [], []

        for t in range(max_iterations - n_warmup):
            jepa_theta = agent.propose()
            jepa_dir = jepa_theta - thetas[-1]
            jepa_norm = np.linalg.norm(jepa_dir) + 1e-8

            # Compute EI direction: sample candidates and score by EI
            # EI(x) = max(0, r_best - mu(x)) -- simplified using empirical mean
            n_cands = 20
            candidates = thetas[-1] + rng.randn(n_cands, param_dim) * 0.5
            r_best = max(rewards)
            r_mean = np.mean(rewards)
            r_std = np.std(rewards) + 1e-6

            # EI scores (simplified): normalized improvement over best
            ei_scores = np.array([
                max(0, (r_best - r_mean) / r_std + rng.randn() * 0.1)
                for _ in range(n_cands)
            ])
            # EI direction: weighted mean of candidates by EI score
            if ei_scores.sum() > 0:
                ei_dir = (candidates * ei_scores[:, None]).sum(axis=0) / \
                         (ei_scores.sum() + 1e-8) - thetas[-1]
            else:
                ei_dir = rng.randn(param_dim)
            ei_norm = np.linalg.norm(ei_dir) + 1e-8

            # UCB direction: toward high reward + high uncertainty regions
            # Proxy: toward areas with fewer past visits (max distance from all past)
            if len(thetas) >= 2:
                dists_from_past = np.array([
                    np.min([np.linalg.norm(c - th) for th in thetas])
                    for c in candidates
                ])
                ucb_scores = (ei_scores / ei_scores.max() + 1e-8 +
                              dists_from_past / (dists_from_past.max() + 1e-8))
                ucb_dir = (candidates * ucb_scores[:, None]).sum(axis=0) / \
                          (ucb_scores.sum() + 1e-8) - thetas[-1]
            else:
                ucb_dir = ei_dir.copy()
            ucb_norm = np.linalg.norm(ucb_dir) + 1e-8

            rand_dir = rng.randn(param_dim)
            rand_norm = np.linalg.norm(rand_dir) + 1e-8

            cos_ei = float(np.dot(jepa_dir, ei_dir) / (jepa_norm * ei_norm))
            cos_ucb = float(np.dot(jepa_dir, ucb_dir) / (jepa_norm * ucb_norm))
            cos_rand = float(np.dot(jepa_dir, rand_dir) / (jepa_norm * rand_norm))

            cos_eis.append(cos_ei)
            cos_ucbs.append(cos_ucb)
            cos_rands_list.append(cos_rand)

            r = rollout_policy(env_name, jepa_theta, n_eval_episodes,
                                seed + n_warmup + t)
            agent.add_to_history(jepa_theta, r)
            thetas.append(jepa_theta)
            rewards.append(r)

        results[f"seed_{seed}"] = {
            "mean_cos_ei": float(np.mean(cos_eis)),
            "mean_cos_ucb": float(np.mean(cos_ucbs)),
            "mean_cos_rand": float(np.mean(cos_rands_list)),
        }
        all_cos_ei.extend(cos_eis)
        all_cos_ucb.extend(cos_ucbs)
        all_cos_rand.extend(cos_rands_list)
        log.info(f"E16 seed={seed}: cos_EI={np.mean(cos_eis):.3f} "
                 f"cos_UCB={np.mean(cos_ucbs):.3f}")

    results["aggregate"] = {
        "mean_cos_ei": float(np.mean(all_cos_ei)),
        "mean_cos_ucb": float(np.mean(all_cos_ucb)),
        "mean_cos_random": float(np.mean(all_cos_rand)),
        "interpretation": (
            "cos_EI or cos_UCB >> cos_rand suggests JEPA implicitly "
            "learned a Bayesian acquisition function. Values near 0 for "
            "all suggest JEPA learned a novel acquisition strategy."
        )
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# E18: Latent interpolation
# ─────────────────────────────────────────────────────────────────────────────

def analyze_latent_interpolation(ckpt_path, env_name, param_dim,
                                  n_pairs=5, n_interp_steps=5,
                                  n_eval_episodes=5, device="cpu"):
    """
    E18: Interpolate between two optimization histories in latent space.
    Take two runs at different seeds (different optimization trajectories),
    encode each into a latent, linearly interpolate, decode, and evaluate.

    Does decoded policy perform "halfway" between the two endpoints?
    If yes: optimization behaves linearly in latent space.
    """
    import torch
    from runner.jepa_runner import rollout_policy

    results = {}

    for pair_idx in range(n_pairs):
        seed_a, seed_b = pair_idx * 2, pair_idx * 2 + 1

        # Get two optimization contexts
        agent_a = load_agent(ckpt_path, param_dim, device)
        agent_b = load_agent(ckpt_path, param_dim, device)
        rng_a = np.random.RandomState(seed_a)
        rng_b = np.random.RandomState(seed_b)

        n_warmup = max(8, param_dim)
        r_a_hist, r_b_hist = [], []
        for i in range(n_warmup):
            ta = rng_a.randn(param_dim)
            ra = rollout_policy(env_name, ta, n_eval_episodes, seed_a + i)
            agent_a.add_to_history(ta, ra)
            r_a_hist.append(ra)

            tb = rng_b.randn(param_dim)
            rb = rollout_policy(env_name, tb, n_eval_episodes, seed_b + i)
            agent_b.add_to_history(tb, rb)
            r_b_hist.append(rb)

        # Encode both contexts
        with torch.no_grad():
            def encode(agent):
                params_t, rewards_t = agent.history.to_tensors(device)
                mean, std = agent.history.get_stats()
                params_norm = (params_t -
                               torch.tensor(mean, dtype=torch.float32)) / \
                              torch.tensor(std + 1e-6, dtype=torch.float32)
                s_x = agent.context_encoder(params_norm, rewards_t)
                return s_x

            s_a = encode(agent_a)
            s_b = encode(agent_b)

            r_a = max(r_a_hist)
            r_b = max(r_b_hist)

            # Interpolate and decode
            interp_rewards = []
            for alpha in np.linspace(0, 1, n_interp_steps):
                s_interp = (1 - alpha) * s_a + alpha * s_b
                desired_r = (1 - alpha) * r_a + alpha * r_b
                desired_t = torch.tensor([[desired_r]], dtype=torch.float32,
                                          device=device)
                s_y = agent_a.predictor(s_interp, desired_t)
                theta_norm = agent_a.decoder(s_y).squeeze(0).cpu().numpy()
                mean_a, std_a = agent_a.history.get_stats()
                theta = theta_norm * std_a + mean_a
                r = rollout_policy(env_name, theta, n_eval_episodes,
                                    seed_a * 100 + pair_idx * 10 + int(alpha * 4))
                interp_rewards.append(float(r))

        # Check linearity: does midpoint perform between endpoints?
        r_mid = interp_rewards[n_interp_steps // 2]
        expected_mid = (interp_rewards[0] + interp_rewards[-1]) / 2
        linearity_error = abs(r_mid - expected_mid)

        results[f"pair_{pair_idx}"] = {
            "r_endpoint_a": interp_rewards[0],
            "r_endpoint_b": interp_rewards[-1],
            "r_midpoint": r_mid,
            "expected_midpoint": expected_mid,
            "linearity_error": float(linearity_error),
            "interp_trajectory": interp_rewards,
        }
        log.info(f"E18 pair={pair_idx}: A={interp_rewards[0]:.1f} "
                 f"mid={r_mid:.1f} B={interp_rewards[-1]:.1f} "
                 f"expected_mid={expected_mid:.1f}")

    mean_linearity_err = float(np.mean([v["linearity_error"]
                                         for v in results.values()]))
    results["aggregate"] = {
        "mean_linearity_error": mean_linearity_err,
        "interpretation": (
            "Low linearity_error means optimization behaves linearly in "
            "JEPA's latent space -- interpolated policies perform halfway "
            "between the two endpoints. High error means the latent space "
            "is curved/nonlinear w.r.t. optimization performance."
        )
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# E19: Optimization phase detection
# ─────────────────────────────────────────────────────────────────────────────

def analyze_optimization_phases(ckpt_path, env_name, param_dim,
                                  n_seeds=3, max_iterations=200,
                                  n_eval_episodes=5, device="cpu",
                                  n_clusters=3):
    """
    E19: Cluster latent vectors across an optimization run.
    Do exploration/refinement/convergence phases emerge naturally?

    Uses k-means clustering on latent vectors, then checks if cluster
    assignments correlate with optimization stage (early/mid/late)
    and reward level (low/medium/high).
    """
    from runner.jepa_runner import rollout_policy

    all_latents, all_rewards, all_times = [], [], []

    for seed in range(n_seeds):
        agent = load_agent(ckpt_path, param_dim, device)
        trace = run_optimization_trace(agent, env_name, param_dim,
                                        seed=seed, max_iterations=max_iterations,
                                        n_eval_episodes=n_eval_episodes,
                                        device=device)
        n = len(trace["latents"])
        all_latents.append(trace["latents"])
        rew = trace["rewards"][len(trace["rewards"]) - n:]
        all_rewards.append(rew)
        all_times.append(np.arange(n) / n)  # normalized time 0→1

    X = np.vstack(all_latents)
    y_r = np.concatenate(all_rewards)
    y_t = np.concatenate(all_times)

    # Simple k-means (no sklearn needed)
    rng = np.random.RandomState(42)
    centers = X[rng.choice(len(X), n_clusters, replace=False)]
    for _ in range(50):
        dists = np.array([np.linalg.norm(X - c, axis=1) for c in centers])
        labels = dists.argmin(axis=0)
        new_centers = np.array([X[labels == k].mean(axis=0)
                                 if (labels == k).any() else centers[k]
                                 for k in range(n_clusters)])
        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers

    # Characterize each cluster
    cluster_stats = {}
    for k in range(n_clusters):
        mask = labels == k
        cluster_stats[f"cluster_{k}"] = {
            "size": int(mask.sum()),
            "mean_reward": float(y_r[mask].mean()),
            "mean_time": float(y_t[mask].mean()),
            "std_reward": float(y_r[mask].std()),
            "std_time": float(y_t[mask].std()),
        }
        log.info(f"E19 cluster {k}: n={mask.sum()} "
                 f"r={y_r[mask].mean():.1f} t={y_t[mask].mean():.2f}")

    # Check if clusters separate by time (temporal phases)
    cluster_times = [cluster_stats[f"cluster_{k}"]["mean_time"]
                     for k in range(n_clusters)]
    time_separation = float(np.std(cluster_times))

    # Check if clusters separate by reward (reward levels)
    cluster_rewards = [cluster_stats[f"cluster_{k}"]["mean_reward"]
                       for k in range(n_clusters)]
    reward_separation = float(np.std(cluster_rewards))

    return {
        "cluster_stats": cluster_stats,
        "time_separation": time_separation,
        "reward_separation": reward_separation,
        "phases_by_time": time_separation > 0.1,
        "phases_by_reward": reward_separation > 10.0,
        "interpretation": (
            "phases_by_time=True: latent clusters map to early/mid/late optimization. "
            "phases_by_reward=True: clusters map to low/medium/high reward regions. "
            "Both True: JEPA naturally organizes optimization into distinct phases."
        )
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

ALL_EXPERIMENTS = {
    "latent_trajectory": analyze_latent_trajectory,
    "prereward_prediction": analyze_prereward_prediction,
    "emergent_optimizer": analyze_emergent_optimizer,
    "explore_exploit": analyze_explore_exploit,
    "forgetting": analyze_forgetting,
    "reverse_history": analyze_reverse_history,
    "representation_probing": analyze_representation_probing,
    "acquisition_function": analyze_acquisition_function,
    "latent_interpolation": analyze_latent_interpolation,
    "phase_detection": analyze_optimization_phases,
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt",
                   default="checkpoints/jepa_pretrained/jepa_pretrained.pt")
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--param_dim", type=int, default=8)
    p.add_argument("--out_dir", default="results/deep_analysis")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--max_iterations", type=int, default=200)
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--experiments", nargs="+",
                   default=["all"],
                   choices=list(ALL_EXPERIMENTS.keys()) + ["all"])
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    exps = list(ALL_EXPERIMENTS.keys()) \
        if "all" in args.experiments else args.experiments

    common_kwargs = dict(
        ckpt_path=args.ckpt,
        env_name=args.env,
        param_dim=args.param_dim,
        n_seeds=args.n_seeds,
        max_iterations=args.max_iterations,
        n_eval_episodes=args.n_episodes,
        device=args.device,
    )

    for exp_name in exps:
        log.info(f"\n{'='*60}\nRunning: {exp_name}\n{'='*60}")
        fn = ALL_EXPERIMENTS[exp_name]

        # Some experiments have different signatures
        if exp_name == "latent_interpolation":
            kwargs = dict(ckpt_path=args.ckpt, env_name=args.env,
                          param_dim=args.param_dim, n_pairs=args.n_seeds,
                          n_eval_episodes=args.n_episodes, device=args.device)
        else:
            kwargs = common_kwargs

        result = fn(**kwargs)

        out_path = f"{args.out_dir}/{exp_name}_{args.env.replace('/', '-')}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        log.info(f"Saved: {out_path}")

    log.info(f"\nAll experiments complete. Results in {args.out_dir}/")


if __name__ == "__main__":
    main()
