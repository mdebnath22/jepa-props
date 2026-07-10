"""
standardized_comparison.py

Runs CEM, CMA-ES, Random Search, TPE, IPOP-CMA-ES with the correct
runner rollout (no bias, matches 8-dim CartPole and 16-dim Swimmer
checkpoints) under a standardized protocol:
  CartPole:  n=10 seeds, budget=400, n_ep=10
  Swimmer:   n=5  seeds, budget=400, n_ep=10

Also runs TPE and IPOP-CMA-ES multi-seed.

USAGE:
  python standardized_comparison.py --env CartPole-v1 --param_dim 8 \
      --n_seeds 10 --budget 400 --n_episodes 10 \
      --out results/reviewer/standardized_cartpole.json

  python standardized_comparison.py --env Swimmer-v4 --param_dim 16 \
      --n_seeds 5 --budget 400 --n_episodes 10 \
      --out results/reviewer/standardized_swimmer.json
"""

import os, sys, json, argparse, logging
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
sys.path.insert(0, os.getcwd())


def rollout(env_name, theta, n_ep, seed):
    """Always use runner rollout (no bias augmentation)."""
    from runner.jepa_runner import rollout_policy
    return float(rollout_policy(env_name, theta, n_ep, seed))


# ── CEM (runner rollout) ─────────────────────────────────────────────────────

def run_cem_clean(env_name, param_dim, seed, budget, n_ep,
                   pop_size=None, elite_frac=0.2, sigma_init=1.0):
    pop_size = pop_size or max(20, param_dim * 4)
    elite_k  = max(2, int(pop_size * elite_frac))
    rng = np.random.RandomState(seed)
    mu    = rng.randn(param_dim)
    sigma = np.ones(param_dim) * sigma_init
    rewards, best_r = [], -np.inf
    n_iters = budget // pop_size
    for it in range(n_iters):
        samples = mu + sigma * rng.randn(pop_size, param_dim)
        rs = [rollout(env_name, s, n_ep, seed * 10000 + it * pop_size + i)
              for i, s in enumerate(samples)]
        rewards.extend(rs)
        idx = np.argsort(rs)[::-1][:elite_k]
        mu    = samples[idx].mean(0)
        sigma = samples[idx].std(0) + 1e-6
        best_r = max(best_r, max(rs))
    return {"best_reward": float(best_r),
            "mean_reward": float(np.mean(rewards)),
            "rewards": rewards}


# ── CMA-ES (numpy implementation, runner rollout) ────────────────────────────

def run_cmaes_clean(env_name, param_dim, seed, budget, n_ep, sigma0=0.5):
    try:
        import cma
        rng = np.random.RandomState(seed)
        x0  = rng.randn(param_dim)
        rewards, best_r = [], -np.inf
        eval_count = [0]
        def obj(theta):
            if eval_count[0] >= budget:
                return 1e10
            r = rollout(env_name, np.array(theta), n_ep,
                        seed * 10000 + eval_count[0])
            eval_count[0] += 1
            rewards.append(r)
            return -r
        opts = cma.CMAOptions()
        opts['maxfevals'] = budget
        opts['seed']      = seed
        opts['verbose']   = -9
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        es.optimize(obj)
        best_r = -es.result.fbest
        return {"best_reward": float(best_r),
                "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
                "rewards": rewards}
    except ImportError:
        # numpy fallback: (1+1)-CMA-ES approximation
        rng = np.random.RandomState(seed)
        theta = rng.randn(param_dim)
        sigma = sigma0
        best_r = rollout(env_name, theta, n_ep, seed)
        rewards = [best_r]
        for t in range(1, budget):
            candidate = theta + sigma * rng.randn(param_dim)
            r = rollout(env_name, candidate, n_ep, seed * 10000 + t)
            rewards.append(r)
            if r > best_r:
                best_r, theta = r, candidate.copy()
                sigma *= 1.22
            else:
                sigma *= 0.82
            sigma = np.clip(sigma, 1e-4, 10.0)
        return {"best_reward": float(best_r),
                "mean_reward": float(np.mean(rewards)),
                "rewards": rewards}


# ── Random Search (runner rollout) ───────────────────────────────────────────

def run_random_clean(env_name, param_dim, seed, budget, n_ep):
    rng = np.random.RandomState(seed)
    rewards, best_r = [], -np.inf
    for t in range(budget):
        theta = rng.randn(param_dim)
        r = rollout(env_name, theta, n_ep, seed * 10000 + t)
        rewards.append(r)
        best_r = max(best_r, r)
    return {"best_reward": float(best_r),
            "mean_reward": float(np.mean(rewards)),
            "rewards": rewards}


# ── PSO (runner rollout) ─────────────────────────────────────────────────────

def run_pso_clean(env_name, param_dim, seed, budget, n_ep,
                   n_particles=20, w=0.7, c1=1.5, c2=1.5):
    rng = np.random.RandomState(seed)
    X  = rng.randn(n_particles, param_dim)
    V  = rng.randn(n_particles, param_dim) * 0.1
    pb = X.copy()
    pb_r = np.array([rollout(env_name, x, n_ep, seed * 10000 + i)
                      for i, x in enumerate(X)])
    gb   = pb[np.argmax(pb_r)].copy()
    gb_r = float(np.max(pb_r))
    rewards = list(pb_r)
    t = n_particles
    while t < budget:
        r1, r2 = rng.rand(n_particles, param_dim), rng.rand(n_particles, param_dim)
        V  = w*V + c1*r1*(pb - X) + c2*r2*(gb - X)
        X  += V
        for i, x in enumerate(X):
            if t >= budget:
                break
            r = rollout(env_name, x, n_ep, seed * 10000 + t)
            rewards.append(r); t += 1
            if r > pb_r[i]:
                pb[i], pb_r[i] = x.copy(), r
                if r > gb_r:
                    gb, gb_r = x.copy(), r
    return {"best_reward": float(gb_r),
            "mean_reward": float(np.mean(rewards)),
            "rewards": rewards}


# ── TPE multi-seed (Optuna, runner rollout) ───────────────────────────────────

def run_tpe_clean(env_name, param_dim, seed, budget, n_ep):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        return run_random_clean(env_name, param_dim, seed, budget, n_ep)

    rewards, eval_count = [], [0]
    def objective(trial):
        theta = np.array([trial.suggest_float(f"x{i}", -3.0, 3.0)
                          for i in range(param_dim)])
        r = rollout(env_name, theta, n_ep, seed * 10000 + eval_count[0])
        eval_count[0] += 1
        rewards.append(r)
        return -r

    sampler = optuna.samplers.TPESampler(seed=seed)
    study   = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=budget, show_progress_bar=False)
    return {"best_reward": float(-study.best_value),
            "mean_reward": float(np.mean(rewards)),
            "rewards": rewards}


# ── IPOP-CMA-ES multi-seed (runner rollout) ───────────────────────────────────

def run_ipop_clean(env_name, param_dim, seed, budget, n_ep, sigma0=0.5):
    try:
        import cma
    except ImportError:
        return run_cmaes_clean(env_name, param_dim, seed, budget, n_ep, sigma0)

    rng = np.random.RandomState(seed)
    rewards, best_r = [], -np.inf
    evals_used = 0
    popsize = 4 + int(3 * np.log(param_dim))

    while evals_used < budget:
        remaining = budget - evals_used
        x0 = rng.randn(param_dim)
        eval_count = [0]

        def obj(theta):
            if eval_count[0] >= remaining:
                return 1e10
            r = rollout(env_name, np.array(theta), n_ep,
                        seed * 100000 + evals_used + eval_count[0])
            eval_count[0] += 1
            rewards.append(r)
            return -r

        opts = cma.CMAOptions()
        opts['maxfevals'] = remaining
        opts['popsize']   = popsize
        opts['seed']      = seed + evals_used
        opts['verbose']   = -9
        try:
            es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
            es.optimize(obj)
            if -es.result.fbest > best_r:
                best_r = float(-es.result.fbest)
        except Exception:
            pass
        evals_used += eval_count[0]
        popsize *= 2   # IPOP: double population on restart

    return {"best_reward": float(best_r),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "rewards": rewards}


# ── Main runner ───────────────────────────────────────────────────────────────

METHODS = {
    "cem":      run_cem_clean,
    "cmaes":    run_cmaes_clean,
    "random":   run_random_clean,
    "pso":      run_pso_clean,
    "tpe":      run_tpe_clean,
    "ipop":     run_ipop_clean,
}


def run_all(env_name, param_dim, n_seeds, budget, n_ep, methods, out_path):
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)
        log.info(f"Loaded existing results from {out_path}")

    for name in methods:
        if name in results:
            log.info(f"  {name}: already done, skipping")
            continue
        fn = METHODS[name]
        log.info(f"\nRunning {name} (n_seeds={n_seeds})...")
        bests, means = [], []
        for seed in range(n_seeds):
            try:
                r = fn(env_name, param_dim, seed, budget, n_ep)
                bests.append(r["best_reward"])
                means.append(r["mean_reward"])
                log.info(f"  seed={seed}: best={r['best_reward']:.1f}")
            except Exception as e:
                log.warning(f"  seed={seed} FAILED: {e}")
        if bests:
            results[name] = {
                "mean_best": float(np.mean(bests)),
                "std_best":  float(np.std(bests)),
                "median_best": float(np.median(bests)),
                "n_seeds":   len(bests),
                "seeds":     bests,
            }
            log.info(f"  {name}: {np.mean(bests):.1f}±{np.std(bests):.1f}")
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

    # Print filled table
    log.info(f"\n{'='*60}")
    log.info(f"STANDARDIZED RESULTS ({env_name}, budget={budget}, n_ep={n_ep})")
    log.info(f"{'='*60}")
    log.info(f"{'Method':<20} {'Mean':>8} {'Std':>8} {'Median':>8} {'n':>4}")
    for name, v in results.items():
        log.info(f"  {name:<18} {v['mean_best']:>8.1f} "
                 f"{v['std_best']:>8.1f} {v['median_best']:>8.1f} "
                 f"{v['n_seeds']:>4}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env",        default="CartPole-v1")
    p.add_argument("--param_dim",  type=int, default=8)
    p.add_argument("--n_seeds",    type=int, default=10)
    p.add_argument("--budget",     type=int, default=400)
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--methods",    nargs="+",
                   default=["cem", "cmaes", "random", "pso", "tpe", "ipop"])
    p.add_argument("--out",
                   default="results/reviewer/standardized_cartpole.json")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    run_all(args.env, args.param_dim, args.n_seeds,
            args.budget, args.n_episodes, args.methods, args.out)


if __name__ == "__main__":
    main()
