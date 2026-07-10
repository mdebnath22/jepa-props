"""
baselines.py — Standard black-box optimization baselines, all sharing one
interface, for comparison against JEPA-based search.

Addresses reviewer request: "no comparisons against CMA-ES, PSO, NES, or
Bayesian optimization" and ProPS Appendix C's evolution-strategy comparison
(OpenAI-ES, (mu+lambda)-ES, Tabu Search).

Every function has signature:
    run_X(env_name, param_dim, seed, max_iterations, **kwargs) -> dict
returning:
    {"rewards": [...] (len == max_iterations, raw per-iteration reward),
     "best_reward": float,
     "wallclock": {"total_s": float, "rollout_s": float, "optimizer_s": float}}

wallclock separates rollout time from optimizer-internal time, directly
addressing the reviewer's request for profiling that doesn't conflate the
two (rollout typically dominates; this lets the paper report that honestly).
"""

import time
import numpy as np
import logging

log = logging.getLogger(__name__)

from envs import rollout_policy


def _timed_rollout(env_name, theta, n_episodes, seed, timer):
    t0 = time.perf_counter()
    r = rollout_policy(env_name, theta, n_episodes, seed)
    timer["rollout_s"] += time.perf_counter() - t0
    return r


# ─────────────────────────────────────────────────────────────────────────────
# CMA-ES
# ─────────────────────────────────────────────────────────────────────────────

def run_cmaes(env_name, param_dim, seed, max_iterations, n_episodes=5,
              sigma0=0.5, popsize=None):
    import cma
    timer = {"rollout_s": 0.0, "optimizer_s": 0.0}
    t_start = time.perf_counter()

    rng = np.random.RandomState(seed)
    x0 = rng.randn(param_dim)
    opts = {"seed": seed, "verbose": -9}
    if popsize:
        opts["popsize"] = popsize
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    rewards = []
    best = -np.inf
    iteration = 0
    while iteration < max_iterations and not es.stop():
        t0 = time.perf_counter()
        solutions = es.ask()
        timer["optimizer_s"] += time.perf_counter() - t0

        losses = []
        for theta in solutions:
            if iteration >= max_iterations:
                break
            r = _timed_rollout(env_name, np.array(theta), n_episodes,
                                seed + iteration, timer)
            rewards.append(r)
            best = max(best, r)
            losses.append(-r)  # cma minimizes
            iteration += 1

        if len(losses) == len(solutions):
            t0 = time.perf_counter()
            es.tell(solutions, losses)
            timer["optimizer_s"] += time.perf_counter() - t0
        else:
            break

    while len(rewards) < max_iterations:
        rewards.append(rewards[-1] if rewards else 0.0)

    timer["total_s"] = time.perf_counter() - t_start
    return {"rewards": rewards[:max_iterations], "best_reward": float(best),
            "wallclock": timer}


# ─────────────────────────────────────────────────────────────────────────────
# Plain PSO in parameter space (NO JEPA — the critical reviewer ablation)
# ─────────────────────────────────────────────────────────────────────────────

def run_pso_param_space(env_name, param_dim, seed, max_iterations,
                         n_episodes=5, n_particles=10, w=0.5, c1=1.5, c2=1.5,
                         force_terms="all"):
    """
    Standard particle swarm optimization run DIRECTLY in parameter space,
    no learned latent model involved at all. This isolates whether JEPA's
    latent prior provides any advantage beyond population structure itself.

    force_terms: "all" | "inertia_only" | "cognitive_only" | "social_only"
                 Ablates the three PSO terms individually (reviewer request).
    """
    timer = {"rollout_s": 0.0, "optimizer_s": 0.0}
    t_start = time.perf_counter()

    rng = np.random.RandomState(seed)
    t0 = time.perf_counter()
    positions = rng.randn(n_particles, param_dim)
    velocities = rng.randn(n_particles, param_dim) * 0.1
    personal_best_pos = positions.copy()
    personal_best_val = np.full(n_particles, -np.inf)
    global_best_pos = positions[0].copy()
    global_best_val = -np.inf
    timer["optimizer_s"] += time.perf_counter() - t0

    rewards = []
    iteration = 0
    while iteration < max_iterations:
        for i in range(n_particles):
            if iteration >= max_iterations:
                break
            r = _timed_rollout(env_name, positions[i], n_episodes,
                                seed + iteration, timer)
            rewards.append(r)
            iteration += 1

            t0 = time.perf_counter()
            if r > personal_best_val[i]:
                personal_best_val[i] = r
                personal_best_pos[i] = positions[i].copy()
            if r > global_best_val:
                global_best_val = r
                global_best_pos = positions[i].copy()
            timer["optimizer_s"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        r1 = rng.rand(n_particles, param_dim)
        r2 = rng.rand(n_particles, param_dim)

        inertia_term = w * velocities
        cognitive_term = c1 * r1 * (personal_best_pos - positions)
        social_term = c2 * r2 * (global_best_pos[None, :] - positions)

        if force_terms == "inertia_only":
            velocities = inertia_term
        elif force_terms == "cognitive_only":
            velocities = cognitive_term
        elif force_terms == "social_only":
            velocities = social_term
        else:  # "all"
            velocities = inertia_term + cognitive_term + social_term

        positions = positions + velocities
        timer["optimizer_s"] += time.perf_counter() - t0

    timer["total_s"] = time.perf_counter() - t_start
    return {"rewards": rewards[:max_iterations], "best_reward": float(global_best_val),
            "wallclock": timer}


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian Optimization (GP-EI via scikit-optimize)
# ─────────────────────────────────────────────────────────────────────────────

def run_bayesopt(env_name, param_dim, seed, max_iterations, n_episodes=5,
                  bound=3.0, max_gp_iterations=100):
    """
    NOTE: full GP-EI refitting scales poorly with iteration count (each
    .tell() refits the GP on the full history, making cost grow roughly
    cubically). To keep wall-clock tractable across many tasks/seeds, BO
    is run for min(max_iterations, max_gp_iterations) iterations using GP,
    then padded with the GP's final best-known point repeated (NOT fresh
    random) for any remaining iterations, so reward curves remain length
    max_iterations and "best" is unaffected, while wall-clock stays bounded.
    Report max_gp_iterations explicitly in any paper text using this baseline.
    """
    from skopt import Optimizer
    from skopt.space import Real

    timer = {"rollout_s": 0.0, "optimizer_s": 0.0}
    t_start = time.perf_counter()

    gp_iterations = min(max_iterations, max_gp_iterations)

    space = [Real(-bound, bound) for _ in range(param_dim)]
    t0 = time.perf_counter()
    opt = Optimizer(space, base_estimator="GP", acq_func="EI",
                     random_state=seed, n_initial_points=min(10, gp_iterations))
    timer["optimizer_s"] += time.perf_counter() - t0

    rewards = []
    best = -np.inf
    for iteration in range(gp_iterations):
        t0 = time.perf_counter()
        x = opt.ask()
        timer["optimizer_s"] += time.perf_counter() - t0

        theta = np.array(x)
        r = _timed_rollout(env_name, theta, n_episodes, seed + iteration, timer)
        rewards.append(r)
        best = max(best, r)

        t0 = time.perf_counter()
        opt.tell(x, -r)  # skopt minimizes
        timer["optimizer_s"] += time.perf_counter() - t0

    # Pad remaining iterations (if max_iterations > gp_iterations) by
    # repeating the last reward -- keeps array length consistent for
    # downstream aggregation without further (slow) GP calls.
    while len(rewards) < max_iterations:
        rewards.append(rewards[-1] if rewards else 0.0)

    timer["total_s"] = time.perf_counter() - t_start
    timer["gp_iterations_actually_run"] = gp_iterations
    return {"rewards": rewards[:max_iterations], "best_reward": float(best), "wallclock": timer}


# ─────────────────────────────────────────────────────────────────────────────
# NES / OpenAI-ES (Salimans et al. 2017) — matches ProPS Appendix C
# ─────────────────────────────────────────────────────────────────────────────

def run_openai_es(env_name, param_dim, seed, max_iterations, n_episodes=5,
                   popsize=20, sigma=0.1, lr=0.03):
    timer = {"rollout_s": 0.0, "optimizer_s": 0.0}
    t_start = time.perf_counter()

    rng = np.random.RandomState(seed)
    theta = rng.randn(param_dim)
    rewards = []
    best = -np.inf
    iteration = 0

    while iteration < max_iterations:
        t0 = time.perf_counter()
        n_samples = min(popsize, max_iterations - iteration)
        eps = rng.randn(n_samples, param_dim)
        timer["optimizer_s"] += time.perf_counter() - t0

        R = np.zeros(n_samples)
        for i in range(n_samples):
            candidate = theta + sigma * eps[i]
            r = _timed_rollout(env_name, candidate, n_episodes,
                                seed + iteration, timer)
            R[i] = r
            rewards.append(r)
            best = max(best, r)
            iteration += 1

        t0 = time.perf_counter()
        A = (R - R.mean()) / (R.std() + 1e-8)
        grad_est = (eps.T @ A) / (n_samples * sigma)
        theta = theta + lr * grad_est
        timer["optimizer_s"] += time.perf_counter() - t0

    timer["total_s"] = time.perf_counter() - t_start
    return {"rewards": rewards[:max_iterations], "best_reward": float(best),
            "wallclock": timer}


# ─────────────────────────────────────────────────────────────────────────────
# (mu + lambda)-ES (Schwefel 1977) — matches ProPS Appendix C
# ─────────────────────────────────────────────────────────────────────────────

def run_mu_plus_lambda_es(env_name, param_dim, seed, max_iterations,
                           n_episodes=5, mu=5, lam=15, sigma=0.3):
    timer = {"rollout_s": 0.0, "optimizer_s": 0.0}
    t_start = time.perf_counter()

    rng = np.random.RandomState(seed)
    t0 = time.perf_counter()
    population = rng.randn(mu, param_dim)
    timer["optimizer_s"] += time.perf_counter() - t0

    rewards = []
    best = -np.inf
    iteration = 0

    parent_fitness = []
    for i in range(mu):
        if iteration >= max_iterations:
            break
        r = _timed_rollout(env_name, population[i], n_episodes,
                            seed + iteration, timer)
        rewards.append(r)
        parent_fitness.append(r)
        best = max(best, r)
        iteration += 1
    parent_fitness = np.array(parent_fitness)

    while iteration < max_iterations:
        t0 = time.perf_counter()
        n_offspring = min(lam, max_iterations - iteration)
        parent_idx = rng.randint(0, len(population), size=n_offspring)
        offspring = population[parent_idx] + rng.randn(n_offspring, param_dim) * sigma
        timer["optimizer_s"] += time.perf_counter() - t0

        offspring_fitness = []
        for i in range(n_offspring):
            r = _timed_rollout(env_name, offspring[i], n_episodes,
                                seed + iteration, timer)
            rewards.append(r)
            offspring_fitness.append(r)
            best = max(best, r)
            iteration += 1
        offspring_fitness = np.array(offspring_fitness)

        t0 = time.perf_counter()
        combined_pop = np.vstack([population[:len(parent_fitness)], offspring])
        combined_fit = np.concatenate([parent_fitness, offspring_fitness])
        top_idx = np.argsort(combined_fit)[-mu:]
        population = combined_pop[top_idx]
        parent_fitness = combined_fit[top_idx]
        timer["optimizer_s"] += time.perf_counter() - t0

    timer["total_s"] = time.perf_counter() - t_start
    return {"rewards": rewards[:max_iterations], "best_reward": float(best),
            "wallclock": timer}


# ─────────────────────────────────────────────────────────────────────────────
# Tabu Search (Glover 1986) — matches ProPS Appendix C
# ─────────────────────────────────────────────────────────────────────────────

def run_tabu_search(env_name, param_dim, seed, max_iterations, n_episodes=5,
                     step_size=0.3, tabu_size=20, n_neighbors=8):
    timer = {"rollout_s": 0.0, "optimizer_s": 0.0}
    t_start = time.perf_counter()

    rng = np.random.RandomState(seed)
    current = rng.randn(param_dim)
    best_theta = current.copy()

    r0 = _timed_rollout(env_name, current, n_episodes, seed, timer)
    rewards = [r0]
    best = r0
    tabu_list = [tuple(np.round(current, 2))]
    iteration = 1

    while iteration < max_iterations:
        t0 = time.perf_counter()
        neighbors = [current + rng.randn(param_dim) * step_size
                     for _ in range(min(n_neighbors, max_iterations - iteration))]
        timer["optimizer_s"] += time.perf_counter() - t0

        neighbor_rewards = []
        for nb in neighbors:
            r = _timed_rollout(env_name, nb, n_episodes, seed + iteration, timer)
            rewards.append(r)
            neighbor_rewards.append(r)
            best = max(best, r)
            iteration += 1

        t0 = time.perf_counter()
        order = np.argsort(neighbor_rewards)[::-1]
        moved = False
        for idx in order:
            key = tuple(np.round(neighbors[idx], 2))
            if key not in tabu_list:
                current = neighbors[idx]
                if neighbor_rewards[idx] > best:
                    best_theta = current.copy()
                tabu_list.append(key)
                if len(tabu_list) > tabu_size:
                    tabu_list.pop(0)
                moved = True
                break
        if not moved and neighbors:
            current = neighbors[order[0]]
        timer["optimizer_s"] += time.perf_counter() - t0

    timer["total_s"] = time.perf_counter() - t_start
    return {"rewards": rewards[:max_iterations], "best_reward": float(best),
            "wallclock": timer}


# ─────────────────────────────────────────────────────────────────────────────
# CEM and Random (centralized here for consistency with other baselines)
# ─────────────────────────────────────────────────────────────────────────────

def run_cem(env_name, param_dim, seed, max_iterations, n_episodes=5,
            popsize=20, elite_frac=0.2, init_std=1.0):
    timer = {"rollout_s": 0.0, "optimizer_s": 0.0}
    t_start = time.perf_counter()

    rng = np.random.RandomState(seed)
    mean = np.zeros(param_dim)
    std = np.ones(param_dim) * init_std
    n_elite = max(1, int(popsize * elite_frac))

    rewards = []
    best = -np.inf
    iteration = 0

    while iteration < max_iterations:
        t0 = time.perf_counter()
        n_samples = min(popsize, max_iterations - iteration)
        samples = mean[None, :] + rng.randn(n_samples, param_dim) * std[None, :]
        timer["optimizer_s"] += time.perf_counter() - t0

        sample_rewards = []
        for i in range(n_samples):
            r = _timed_rollout(env_name, samples[i], n_episodes,
                                seed + iteration, timer)
            rewards.append(r)
            sample_rewards.append(r)
            best = max(best, r)
            iteration += 1

        t0 = time.perf_counter()
        sample_rewards = np.array(sample_rewards)
        elite_idx = np.argsort(sample_rewards)[-n_elite:]
        elite = samples[elite_idx]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-6
        timer["optimizer_s"] += time.perf_counter() - t0

    timer["total_s"] = time.perf_counter() - t_start
    return {"rewards": rewards[:max_iterations], "best_reward": float(best),
            "wallclock": timer}


def run_random_search(env_name, param_dim, seed, max_iterations, n_episodes=5,
                       scale=1.0):
    timer = {"rollout_s": 0.0, "optimizer_s": 0.0}
    t_start = time.perf_counter()

    rng = np.random.RandomState(seed)
    rewards = []
    best = -np.inf
    for iteration in range(max_iterations):
        t0 = time.perf_counter()
        theta = rng.randn(param_dim) * scale
        timer["optimizer_s"] += time.perf_counter() - t0

        r = _timed_rollout(env_name, theta, n_episodes, seed + iteration, timer)
        rewards.append(r)
        best = max(best, r)

    timer["total_s"] = time.perf_counter() - t_start
    return {"rewards": rewards, "best_reward": float(best), "wallclock": timer}


# ─────────────────────────────────────────────────────────────────────────────
# Registry for CLI dispatch
# ─────────────────────────────────────────────────────────────────────────────

BASELINE_REGISTRY = {
    "cmaes": run_cmaes,
    "pso": run_pso_param_space,
    "bayesopt": run_bayesopt,
    "openai_es": run_openai_es,
    "mu_plus_lambda_es": run_mu_plus_lambda_es,
    "tabu_search": run_tabu_search,
    "cem": run_cem,
    "random": run_random_search,
}
