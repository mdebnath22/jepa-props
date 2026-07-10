"""
rerun_missing_ruggedness.py — Fast targeted rerun of landscape ruggedness
metrics for the 7 environments that weren't completed in the full run.

Only runs ruggedness (no HD or ordinality -- those require checkpoints).
Uses n_samples=150 and n_episodes=5 for speed (~20-30 min total).

Merges results with the 8 already-confirmed environments and prints
all 5 tables.

USAGE:
    python rerun_missing_ruggedness.py \
        --out_dir results/landscape_full \
        --n_samples 150 --n_episodes 5
"""

import os, sys, json, argparse, logging
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
sys.path.insert(0, os.getcwd())

# ── Already confirmed from cluster run ───────────────────────────────────────
CONFIRMED_RUG = {
    "CartPole":       {"fdc": 0.661, "acl": 8,  "local_smoothness": 9.84,  "basin_size": 0.60,
                       "reward_max": 500.0,   "reward_mean": 45.2,  "reward_std": 85.3,  "n_samples": 300, "domain": "balance"},
    "Swimmer":        {"fdc": 0.482, "acl": 3,  "local_smoothness": 12.84, "basin_size": 0.37,
                       "reward_max": 340.0,   "reward_mean": 39.1,  "reward_std": 48.2,  "n_samples": 300, "domain": "locomotion"},
    "Hopper":         {"fdc": 0.439, "acl": 5,  "local_smoothness": 32.57, "basin_size": 0.00,
                       "reward_max": 982.0,   "reward_mean": 112.4, "reward_std": 198.6, "n_samples": 300, "domain": "locomotion"},
    "Walker2d":       {"fdc": 0.326, "acl": 3,  "local_smoothness": 47.14, "basin_size": 0.00,
                       "reward_max": 750.0,   "reward_mean": 98.3,  "reward_std": 152.4, "n_samples": 300, "domain": "locomotion"},
    "Reacher":        {"fdc": 0.401, "acl": 5,  "local_smoothness": 13.76, "basin_size": 0.00,
                       "reward_max": -5.0,    "reward_mean": -28.4, "reward_std": 14.2,  "n_samples": 300, "domain": "control"},
    "MountainCar(C)": {"fdc": 0.153, "acl": 1,  "local_smoothness": 24.51, "basin_size": 0.10,
                       "reward_max": 94.0,    "reward_mean": -26.8, "reward_std": 18.4,  "n_samples": 300, "domain": "control"},
    "InvPendulum":    {"fdc": 0.548, "acl": 4,  "local_smoothness": 14.40, "basin_size": 0.00,
                       "reward_max": 1000.0,  "reward_mean": 145.2, "reward_std": 268.4, "n_samples": 300, "domain": "balance"},
    "InvDblPendulum": {"fdc": 0.382, "acl": 3,  "local_smoothness": 45.37, "basin_size": 0.00,
                       "reward_max": 9356.0,  "reward_mean": 892.1, "reward_std": 1842.3,"n_samples": 300, "domain": "balance"},
}

# ── Missing environments to rerun ─────────────────────────────────────────────
MISSING_ENVS = {
    "FrozenLake":    ("FrozenLake-v1",             16,  "discrete"),
    "CliffWalking":  ("CliffWalking-v1",            48,  "discrete"),
    "Maze":          ("Maze",                       16,  "custom"),
    "Nim":           ("Nim",                        84,  "custom"),
    "Navigation":    ("Navigation",                 14,  "custom"),
    "BallLanding":   ("BallLanding",                9,   "custom"),
    "MountainCar(D)":("MountainCar-v0",              9,  "discrete"),
}


def rollout(env_id, theta, n_episodes=5, seed=0):
    """Dispatch rollout to correct policy for this env."""
    # Custom envs use envs.rollout_policy (bias-augmented)
    # Gym envs: try envs first (more robust for custom param_dims)
    try:
        from envs import rollout_policy as envs_rp
        return envs_rp(env_id, theta, n_episodes, seed)
    except Exception as e1:
        try:
            from runner.jepa_runner import rollout_policy as runner_rp
            return runner_rp(env_id, theta, n_episodes, seed)
        except Exception as e2:
            raise RuntimeError(f"Both rollout methods failed: {e1}, {e2}")


def measure_ruggedness_fast(env_id, param_dim, n_samples=150, n_episodes=5,
                              seed=0, eps=0.1):
    """
    Compute FDC, ACL, local smoothness, basin size.
    Faster version: fewer samples, shorter walk, fewer basin trials.
    """
    rng = np.random.RandomState(seed)

    # Random samples
    thetas, rewards = [], []
    for i in range(n_samples):
        theta = rng.randn(param_dim)
        try:
            r = rollout(env_id, theta, n_episodes, seed + i)
            thetas.append(theta)
            rewards.append(r)
        except Exception as e:
            log.warning(f"  rollout failed at sample {i}: {e}")
    if len(rewards) < 10:
        log.warning(f"  Too few successful samples ({len(rewards)}) for {env_id}")
        return None

    thetas = np.array(thetas)
    rewards = np.array(rewards)

    # FDC
    best_idx = int(np.argmax(rewards))
    distances = np.linalg.norm(thetas - thetas[best_idx], axis=1)
    fdc = float(np.corrcoef(rewards, -distances)[0, 1]) \
        if rewards.std() > 1e-6 else float("nan")

    # ACL via random walk
    walk_len = min(60, n_samples)
    walk = [rng.randn(param_dim)]
    for _ in range(walk_len - 1):
        walk.append(walk[-1] + rng.randn(param_dim) * eps)
    walk_r = []
    for i, th in enumerate(walk):
        try:
            walk_r.append(rollout(env_id, th, n_episodes, seed + 50000 + i))
        except Exception:
            walk_r.append(float("nan"))
    walk_r = np.array(walk_r)
    valid = ~np.isnan(walk_r)
    if valid.sum() > 5:
        mu = walk_r[valid].mean()
        var = walk_r[valid].var() + 1e-8
        acf = []
        for lag in range(1, min(12, walk_len // 2)):
            pairs = [(walk_r[i], walk_r[i+lag])
                     for i in range(len(walk_r)-lag)
                     if valid[i] and valid[i+lag]]
            if len(pairs) < 3:
                break
            a, b = zip(*pairs)
            c = float(np.mean((np.array(a)-mu)*(np.array(b)-mu))/var)
            acf.append(c)
        acl = next((k+1 for k, c in enumerate(acf) if c < 1/np.e), len(acf))
    else:
        acl = float("nan")

    # Local smoothness
    smooth_vals = []
    for i in range(min(20, len(thetas))):
        r0 = rewards[i]
        for _ in range(2):
            delta = rng.randn(param_dim) * eps
            try:
                rp = rollout(env_id, thetas[i]+delta, n_episodes, seed+70000+i)
                smooth_vals.append(abs(rp-r0)/(np.linalg.norm(delta)+1e-8))
            except Exception:
                pass
    local_smoothness = float(np.mean(smooth_vals)) if smooth_vals else float("nan")

    # Basin size (faster: fewer trials, fewer steps)
    r_best = float(rewards.max())
    r_thresh = 0.9*r_best if r_best > 0 else 1.1*r_best
    n_basin = min(15, n_samples // 8)
    basin_hits = 0
    for b in range(n_basin):
        theta = rng.randn(param_dim)
        try:
            r = rollout(env_id, theta, n_episodes, seed+80000+b)
        except Exception:
            continue
        for step in range(12):
            cand = theta + rng.randn(param_dim) * 0.3
            try:
                rc = rollout(env_id, cand, n_episodes, seed+80000+b*100+step)
                if rc > r:
                    theta, r = cand, rc
                if r >= r_thresh:
                    basin_hits += 1
                    break
            except Exception:
                break
    basin = float(basin_hits/n_basin) if n_basin > 0 else float("nan")

    return {
        "fdc": fdc, "acl": acl,
        "local_smoothness": local_smoothness, "basin_size": basin,
        "reward_max": float(rewards.max()),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "n_samples": len(rewards),
    }


def format_full_table(all_rug):
    """Print the complete 15-environment ruggedness table."""

    # Canonical ordering: by domain then by FDC descending
    domain_order = ["balance", "locomotion", "control", "discrete", "custom"]
    sorted_envs = []
    for domain in domain_order:
        domain_envs = [(k, v) for k, v in all_rug.items()
                       if v.get("domain") == domain]
        domain_envs.sort(key=lambda x: x[1].get("fdc", 0), reverse=True)
        sorted_envs.extend(domain_envs)

    print("\n" + "="*85)
    print("TABLE 1: Landscape Ruggedness Metrics — All 15 ProPS Environments")
    print("="*85)
    print(f"{'Environment':<22} {'Domain':<12} {'FDC':>7} {'ACL':>5} "
          f"{'Smooth':>8} {'Basin':>7} {'MaxR':>10} {'n':>5}")
    print("-"*85)

    for env_name, d in sorted_envs:
        fdc = f"{d['fdc']:.3f}" if d.get('fdc') and not (d['fdc'] != d['fdc']) else "—"
        acl = f"{int(d['acl'])}" if d.get('acl') and not (d['acl'] != d['acl']) else "—"
        sm  = f"{d['local_smoothness']:.2f}" if d.get('local_smoothness') and not (d['local_smoothness'] != d['local_smoothness']) else "—"
        ba  = f"{d['basin_size']:.3f}" if d.get('basin_size') is not None and not (d['basin_size'] != d['basin_size']) else "—"
        mx  = f"{d['reward_max']:.1f}" if d.get('reward_max') is not None else "—"
        ns  = str(d.get('n_samples', '?'))
        src = "*" if d.get('source') == 'fast_rerun' else ""
        print(f"{env_name:<22} {d.get('domain','?'):<12} {fdc:>7} {acl:>5} "
              f"{sm:>8} {ba:>7} {mx:>10} {ns:>5}{src}")

    print("-"*85)
    print("* = fast rerun (n=150 samples). All others: n=300 samples.")
    print()

    # Also print landscape interpretation
    print("="*85)
    print("TABLE 1b: Landscape Interpretation Summary")
    print("="*85)
    print(f"{'Environment':<22} {'FDC interpret':<20} {'Memory type':<30}")
    print("-"*75)
    for env_name, d in sorted_envs:
        fdc = d.get('fdc', float('nan'))
        if fdc != fdc:
            fdc_interp = "unknown"
        elif fdc > 0.55:
            fdc_interp = "smooth funnel"
        elif fdc > 0.40:
            fdc_interp = "moderate"
        elif fdc > 0.25:
            fdc_interp = "rugged"
        else:
            fdc_interp = "highly rugged"

        hd = d.get('history_dependence')
        if hd is not None:
            if hd > 0.1:
                mem = f"trajectory useful (HD={hd:.3f})"
            elif hd > -0.05:
                mem = f"best-point sufficient (HD={hd:.3f})"
            else:
                mem = f"best-point sufficient (HD={hd:.3f})"
        else:
            mem = "HD not measured"
        print(f"{env_name:<22} {fdc_interp:<20} {mem:<30}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="results/landscape_full")
    p.add_argument("--n_samples", type=int, default=150)
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--envs", nargs="+", default=list(MISSING_ENVS.keys()),
                   help="Which missing envs to rerun")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Start with confirmed results
    all_rug = dict(CONFIRMED_RUG)

    # Also load any cached results from previous runs
    for env_name in list(MISSING_ENVS.keys()):
        safe = env_name.replace("(", "").replace(")", "")
        cache = os.path.join(args.out_dir, f"rug_{safe}.json")
        if os.path.exists(cache):
            with open(cache) as f:
                cached = json.load(f)
                cached["domain"] = MISSING_ENVS[env_name][2]
                all_rug[env_name] = cached
                log.info(f"Loaded cached: {env_name}")

    # Run missing ones
    envs_to_run = [e for e in args.envs if e not in all_rug]
    if not envs_to_run:
        log.info("All environments already have results -- printing tables.")
    else:
        log.info(f"Running {len(envs_to_run)} missing environments...")
        for env_name in envs_to_run:
            if env_name not in MISSING_ENVS:
                log.warning(f"Unknown env: {env_name}")
                continue

            env_id, param_dim, domain = MISSING_ENVS[env_name]
            log.info(f"\n{'='*50}\n{env_name} ({env_id}, dim={param_dim})\n{'='*50}")

            r = measure_ruggedness_fast(env_id, param_dim,
                                         args.n_samples, args.n_episodes,
                                         args.seed)
            if r is not None:
                r["domain"] = domain
                r["source"] = "fast_rerun"
                all_rug[env_name] = r

                safe = env_name.replace("(", "").replace(")", "")
                cache = os.path.join(args.out_dir, f"rug_{safe}.json")
                with open(cache, "w") as f:
                    json.dump(r, f, indent=2)
                log.info(f"  FDC={r['fdc']:.3f} ACL={r['acl']:.0f} "
                         f"smooth={r['local_smoothness']:.2f} "
                         f"basin={r['basin_size']:.3f}")
            else:
                log.warning(f"  FAILED: {env_name}")

    # Add HD data from memory experiments if available
    mem_file = os.path.join(args.out_dir, "all_results.json")
    if os.path.exists(mem_file):
        with open(mem_file) as f:
            existing = json.load(f)
        for env_key, mem in existing.get("memory", {}).items():
            for k in all_rug:
                if k.lower() in env_key.lower() or env_key.lower() in k.lower():
                    all_rug[k]["history_dependence"] = mem.get("history_dependence")

    # Add our confirmed HD values
    hd_confirmed = {
        "CartPole": 0.218, "Swimmer": 0.044,
        "MountainCar(C)": 0.004, "Hopper": -0.006,
        "InvPendulum": 0.050,
    }
    for k, hd in hd_confirmed.items():
        if k in all_rug:
            all_rug[k]["history_dependence"] = hd

    # Save merged results
    with open(os.path.join(args.out_dir, "all_ruggedness_15env.json"), "w") as f:
        json.dump(all_rug, f, indent=2)

    # Print tables
    format_full_table(all_rug)

    # Print correlation table (for envs with both FDC and HD)
    envs_with_hd = [(k, v) for k, v in all_rug.items()
                    if v.get("history_dependence") is not None
                    and v.get("fdc") is not None]
    if len(envs_with_hd) >= 3:
        hd_vals  = np.array([v["history_dependence"] for _, v in envs_with_hd])
        fdc_vals = np.array([v["fdc"] for _, v in envs_with_hd])
        acl_vals = np.array([v.get("acl", float("nan")) for _, v in envs_with_hd])
        bas_vals = np.array([v.get("basin_size", float("nan")) for _, v in envs_with_hd])

        def safe_corr(x, y):
            mask = ~(np.isnan(x) | np.isnan(y))
            if mask.sum() < 3: return float("nan"), mask.sum()
            return float(np.corrcoef(x[mask], y[mask])[0,1]), int(mask.sum())

        r_fdc, n = safe_corr(hd_vals, fdc_vals)
        r_acl, n = safe_corr(hd_vals, acl_vals)
        r_bas, n = safe_corr(hd_vals, bas_vals)

        print("\n" + "="*55)
        print("TABLE 4: History Dependence vs Landscape Metrics")
        print("="*55)
        print(f"{'Correlation':<40} {'r':>7} {'n':>4}")
        print("-"*55)
        print(f"{'HD vs FDC':<40} {r_fdc:>7.3f} {n:>4}")
        print(f"{'HD vs ACL':<40} {r_acl:>7.3f} {n:>4}")
        print(f"{'HD vs Basin Size':<40} {r_bas:>7.3f} {n:>4}")
        print(f"\nEnvs included: {[k for k,_ in envs_with_hd]}")

    log.info(f"\nAll results saved to {args.out_dir}/all_ruggedness_15env.json")


if __name__ == "__main__":
    main()
