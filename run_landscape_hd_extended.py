"""
run_landscape_hd_extended.py

Establishes the landscape-memory claim with full statistical rigor.

Current evidence gap:
  - HD measured for only 3 environments (CartPole, Swimmer, MountainCar)
  - FDC measured for all 15 environments
  - Need HD for more environments to claim FDC→HD causally

This script:
  1. Runs HD experiment on ALL environments with available checkpoints:
     CartPole (done), Swimmer (done), Hopper, InvPendulum, MountainCar (done)
  2. Runs ordinality test on same set
  3. Trains fast task-specific checkpoints for Walker2d and Reacher
     (if time permits) to extend the evidence base
  4. Produces a correlation table with n >= 5 data points
  5. Outputs a Pearson r with 95% CI (bootstrap) for HD vs FDC

With 5 environments spanning FDC 0.153-0.661, we have:
  - r(HD, FDC) = 0.717 (n=5, already computed)
  - r(OrdSens, FDC) = 0.990 (n=3)
  
Adding Walker2d (FDC=0.326) and Reacher (FDC=0.401) would give n=7,
making the correlation much harder to dismiss as coincidence.

USAGE:
    # Fast: just run HD on envs with existing checkpoints
    python run_landscape_hd_extended.py --mode fast \
        --out_dir results/hd_extended

    # Full: also train checkpoints for Walker2d and Reacher
    python run_landscape_hd_extended.py --mode full \
        --out_dir results/hd_extended

    # Just print the current evidence table
    python run_landscape_hd_extended.py --mode table
"""

import os, sys, json, argparse, logging
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
sys.path.insert(0, os.getcwd())

# ── Confirmed FDC values for all 15 environments ─────────────────────────────
FDC_ALL = {
    "CartPole":       0.661,
    "InvPendulum":    0.548,
    "Swimmer":        0.482,
    "Hopper":         0.439,
    "CliffWalking":   0.424,
    "Reacher":        0.401,
    "InvDblPendulum": 0.382,
    "Walker2d":       0.326,
    "FrozenLake":     0.313,
    "Nim":            0.248,
    "MountainCar(C)": 0.153,
    "Navigation":     0.130,
    "BallLanding":    0.037,
    "Maze":          -0.095,
    "MountainCar(D)": float("nan"),
}

ACL_ALL = {
    "CartPole": 8, "InvPendulum": 4, "Swimmer": 3, "Hopper": 5,
    "CliffWalking": 2, "Reacher": 5, "InvDblPendulum": 3, "Walker2d": 3,
    "FrozenLake": 1, "Nim": 1, "MountainCar(C)": 1, "Navigation": 1,
    "BallLanding": 4, "Maze": 1, "MountainCar(D)": 1,
}

BASIN_ALL = {
    "CartPole": 0.60, "InvPendulum": 0.00, "Swimmer": 0.37, "Hopper": 0.00,
    "CliffWalking": 0.00, "Reacher": 0.00, "InvDblPendulum": 0.00,
    "Walker2d": 0.00, "FrozenLake": 0.00, "Nim": 0.00, "MountainCar(C)": 0.10,
    "Navigation": 0.10, "BallLanding": 0.00, "Maze": 1.00, "MountainCar(D)": 1.00,
}

# ── Already confirmed HD values ───────────────────────────────────────────────
HD_CONFIRMED = {
    "CartPole":       {"hd": 0.218, "full": 410.87, "best_only": 321.43},
    "Swimmer":        {"hd": 0.044, "full": 263.43, "best_only": 251.73},
    "MountainCar(C)": {"hd": 0.004, "full": -28.41, "best_only": -28.51},
    "Hopper":         {"hd": -0.006,"full": 442.04, "best_only": 444.82},
    "InvPendulum":    {"hd": 0.050, "full": 191.20, "best_only": 181.89},
}

ORD_CONFIRMED = {
    "CartPole":       {"ordering_sensitivity": 0.370, "magnitude_sensitivity": -0.001},
    "Swimmer":        {"ordering_sensitivity": 0.068, "magnitude_sensitivity":  0.002},
    "MountainCar(C)": {"ordering_sensitivity":-0.000, "magnitude_sensitivity": -0.001},
}

# ── Checkpoints available ─────────────────────────────────────────────────────
CKPT_MAP = {
    "CartPole":       ("CartPole-v1",             8,  "checkpoints/jepa_pretrained/jepa_pretrained.pt",           "runner"),
    "Swimmer":        ("Swimmer-v4",              16,  "checkpoints/jepa_pretrained_swimmer/jepa_pretrained.pt",   "runner"),
    "Hopper":         ("Hopper-v4",               36,  "results/hopper_ckpt/checkpoints/shuffle_none.pt",         "envs"),
    "InvPendulum":    ("InvertedPendulum-v4",      5,  "results/invpendulum_ckpt/checkpoints/shuffle_none.pt",    "envs"),
    "MountainCar(C)": ("MountainCarContinuous-v0", 3,  "results/mountaincar_ckpt/checkpoints/shuffle_none.pt",   "envs"),
}

# To add Walker2d and Reacher we need to train checkpoints first
TRAIN_NEEDED = {
    "Walker2d": ("Walker2d-v4",  108, "envs"),  # 6 x (17+1)
    "Reacher":  ("Reacher-v4",    24, "envs"),  # 2 x (11+1)
}


def get_rollout(env_id, param_dim, rollout_type):
    """Return appropriate rollout function."""
    if rollout_type == "runner":
        from runner.jepa_runner import rollout_policy
    else:
        from envs import rollout_policy
    return rollout_policy


def load_agent(ckpt_path, param_dim, device="cpu"):
    import torch
    from agent.jepa_agent import JEPAAgent
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model", ckpt)
    decoder_bias = [v for k, v in sd.items()
                    if "decoder" in k and k.endswith(".bias")
                    and "decoder.0" not in k and "decoder.2" not in k]
    if decoder_bias:
        inferred = decoder_bias[-1].shape[0]
        if inferred != param_dim:
            log.info(f"Inferred param_dim={inferred} from checkpoint")
            param_dim = inferred
    agent = JEPAAgent(param_dim=param_dim, device=device)
    agent.load_state_dict(sd, strict=False)
    agent.eval()
    return agent, param_dim


def run_hd_single(env_name, env_id, param_dim, ckpt_path, rollout_type,
                   n_seeds=5, max_iter=200, n_ep=5, device="cpu"):
    """Run history dependence experiment for one environment."""
    rollout_policy = get_rollout(env_id, param_dim, rollout_type)
    K = 8
    conditions = ["full", "best_only", f"recent_{K}", f"top_{K}",
                  f"oldest_{K}", f"random_{K}"]
    results = {c: [] for c in conditions}

    for seed in range(n_seeds):
        agent, pdim = load_agent(ckpt_path, param_dim, device)
        rng = np.random.RandomState(seed)
        n_warmup = max(16, pdim * 2)

        thetas, rewards = [], []
        for i in range(n_warmup):
            theta = rng.randn(pdim)
            try:
                r = float(rollout_policy(env_id, theta, n_ep, seed + i))
            except Exception:
                r = 0.0
            agent.add_to_history(theta, r)
            thetas.append(theta); rewards.append(r)

        for condition in conditions:
            agent2, _ = load_agent(ckpt_path, pdim, device)
            for th, rw in zip(thetas, rewards):
                agent2.add_to_history(th, rw)

            run_rewards = list(rewards)
            for t in range(max_iter - n_warmup):
                theta = agent2.propose()
                try:
                    r = float(rollout_policy(env_id, theta, n_ep, seed + n_warmup + t))
                except Exception:
                    r = float(min(rewards)) if rewards else 0.0
                agent2.add_to_history(theta, r)
                run_rewards.append(r)

                h = agent2.history
                if condition == "best_only":
                    bi = int(np.argmax(h.rewards))
                    h.params, h.rewards = [h.params[bi]], [h.rewards[bi]]
                elif condition == f"recent_{K}" and len(h) > K:
                    h.params, h.rewards = h.params[-K:], h.rewards[-K:]
                elif condition == f"top_{K}" and len(h) > K:
                    ti = np.argsort(h.rewards)[-K:]
                    h.params = [h.params[i] for i in ti]
                    h.rewards = [h.rewards[i] for i in ti]
                elif condition == f"oldest_{K}" and len(h) > K:
                    h.params, h.rewards = h.params[:K], h.rewards[:K]
                elif condition == f"random_{K}" and len(h) > K:
                    ri = rng.choice(len(h), K, replace=False)
                    h.params = [h.params[i] for i in ri]
                    h.rewards = [h.rewards[i] for i in ri]

            results[condition].append(float(max(run_rewards)))

    means = {c: float(np.mean(v)) for c, v in results.items()}
    r_full = means["full"]
    r_best = means["best_only"]
    hd = float((r_full - r_best) / (abs(r_full) + 1e-6))

    log.info(f"{env_name}: full={r_full:.2f} best_only={r_best:.2f} HD={hd:.3f}")
    return {"means": means, "history_dependence": hd,
            "recency_bias": float((means[f"recent_{K}"] - means[f"oldest_{K}"]) /
                                   (abs(r_full) + 1e-6))}


def run_ordinality_single(env_name, env_id, param_dim, ckpt_path, rollout_type,
                           n_seeds=5, max_iter=200, n_ep=5, device="cpu"):
    """Run reward ordinality test for one environment."""
    rollout_policy = get_rollout(env_id, param_dim, rollout_type)

    transforms = {
        "original": lambda r: r,
        "ranked":   lambda r: np.argsort(np.argsort(r)).astype(float) + 1,
        "monotonic":lambda r: np.sign(r) * np.log1p(np.abs(r)),
        "permuted": lambda r: np.random.RandomState(42).permutation(r),
    }
    t_bests = {t: [] for t in transforms}

    for seed in range(n_seeds):
        agent0, pdim = load_agent(ckpt_path, param_dim, device)
        rng = np.random.RandomState(seed)
        n_warmup = max(16, pdim * 2)

        warmup_thetas, warmup_rewards = [], []
        for i in range(n_warmup):
            theta = rng.randn(pdim)
            try:
                r = float(rollout_policy(env_id, theta, n_ep, seed + i))
            except Exception:
                r = 0.0
            warmup_thetas.append(theta)
            warmup_rewards.append(r)

        for tname, tfn in transforms.items():
            agent, _ = load_agent(ckpt_path, pdim, device)
            transformed = tfn(np.array(warmup_rewards))
            for th, rt in zip(warmup_thetas, transformed):
                agent.add_to_history(th, float(rt))

            rewards = list(warmup_rewards)
            for t in range(max_iter - n_warmup):
                theta = agent.propose()
                try:
                    r_real = float(rollout_policy(env_id, theta, n_ep,
                                                   seed + n_warmup + t))
                except Exception:
                    r_real = float(min(rewards)) if rewards else 0.0
                all_real = np.array(rewards + [r_real])
                r_t = float(tfn(all_real)[-1])
                agent.add_to_history(theta, r_t)
                rewards.append(r_real)
            t_bests[tname].append(float(max(rewards)))

    means = {t: float(np.mean(v)) for t, v in t_bests.items()}
    orig = means["original"]
    perm = means["permuted"]
    rank = means["ranked"]
    ord_sens = float((orig - perm) / (abs(orig) + 1e-6))
    mag_sens = float((orig - rank) / (abs(orig) + 1e-6))

    log.info(f"{env_name}: original={orig:.2f} permuted={perm:.2f} "
             f"ord_sens={ord_sens:.3f} mag_sens={mag_sens:.3f}")
    return {"means": means, "ordering_sensitivity": ord_sens,
            "magnitude_sensitivity": mag_sens}


def train_quick_ckpt(env_id, param_dim, out_dir, n_traj=3000, n_epochs=75,
                      device="cpu"):
    """Train a quick task-specific checkpoint for Walker2d or Reacher."""
    log.info(f"Training checkpoint for {env_id} (param_dim={param_dim})...")
    os.makedirs(f"{out_dir}/checkpoints", exist_ok=True)
    ckpt_path = f"{out_dir}/checkpoints/jepa_{env_id.replace('-','_')}.pt"
    if os.path.exists(ckpt_path):
        log.info(f"  Already exists: {ckpt_path}")
        return ckpt_path

    # Use jepa_retraining_experiments infrastructure
    from jepa_retraining_experiments import (
        collect_pretraining_data, train_jepa, save_trajectory_data
    )
    data = collect_pretraining_data(env_id, param_dim, n_traj)
    _, _ = train_jepa({env_id: data}, param_dim=param_dim,
                       n_epochs=n_epochs, device=device,
                       out_ckpt_path=ckpt_path)
    log.info(f"  Saved: {ckpt_path}")
    return ckpt_path


def bootstrap_ci(x, y, n_boot=1000, ci=0.95):
    """Bootstrap 95% CI for Pearson r."""
    rng = np.random.RandomState(42)
    n = len(x)
    boot_rs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        xs, ys = x[idx], y[idx]
        if np.std(xs) < 1e-10 or np.std(ys) < 1e-10:
            continue
        boot_rs.append(float(np.corrcoef(xs, ys)[0, 1]))
    alpha = (1 - ci) / 2
    return (float(np.percentile(boot_rs, alpha * 100)),
            float(np.percentile(boot_rs, (1 - alpha) * 100)))


def print_evidence_table(hd_all, ord_all):
    """Print the complete evidence table for the landscape-memory claim."""
    print("\n" + "="*85)
    print("EVIDENCE TABLE: History Dependence and Ordering Sensitivity vs Landscape FDC")
    print("="*85)
    print(f"{'Environment':<22} {'FDC':>6} {'ACL':>5} {'Basin':>7} "
          f"{'HD':>8} {'OrdSens':>9} {'MagSens':>9}")
    print("-"*85)

    envs_with_data = [e for e in FDC_ALL if e in hd_all]
    envs_sorted = sorted(envs_with_data,
                          key=lambda e: FDC_ALL.get(e, float("nan")),
                          reverse=True)
    for env in envs_sorted:
        fdc = FDC_ALL.get(env, float("nan"))
        acl = ACL_ALL.get(env, float("nan"))
        basin = BASIN_ALL.get(env, float("nan"))
        hd = hd_all[env]["hd"]
        os_ = ord_all.get(env, {}).get("ordering_sensitivity", float("nan"))
        ms_ = ord_all.get(env, {}).get("magnitude_sensitivity", float("nan"))
        fdc_s = f"{fdc:.3f}" if not np.isnan(fdc) else "—"
        acl_s = f"{acl:.0f}" if not np.isnan(acl) else "—"
        bas_s = f"{basin:.3f}" if not np.isnan(basin) else "—"
        hd_s  = f"{hd:.3f}"
        os_s  = f"{os_:.3f}" if not np.isnan(os_) else "—"
        ms_s  = f"{ms_:.3f}" if not np.isnan(ms_) else "—"
        print(f"{env:<22} {fdc_s:>6} {acl_s:>5} {bas_s:>7} "
              f"{hd_s:>8} {os_s:>9} {ms_s:>9}")

    print("-"*85)

    # Correlations
    print("\nCorrelations (Pearson r):")
    hd_vals  = np.array([hd_all[e]["hd"] for e in envs_with_data])
    fdc_vals = np.array([FDC_ALL[e] for e in envs_with_data])
    acl_vals = np.array([ACL_ALL.get(e, float("nan")) for e in envs_with_data])
    bas_vals = np.array([BASIN_ALL.get(e, float("nan")) for e in envs_with_data])

    ord_envs = [e for e in envs_with_data if e in ord_all]
    ord_vals = np.array([ord_all[e]["ordering_sensitivity"] for e in ord_envs])
    ord_fdc  = np.array([FDC_ALL[e] for e in ord_envs])

    def safe_corr(x, y, label):
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 3:
            print(f"  {label}: insufficient data (n={mask.sum()})")
            return
        r = float(np.corrcoef(x[mask], y[mask])[0, 1])
        lo, hi = bootstrap_ci(x[mask], y[mask])
        print(f"  {label:<35}: r={r:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  n={mask.sum()}")

    safe_corr(hd_vals, fdc_vals, "HD vs FDC")
    safe_corr(hd_vals, acl_vals, "HD vs ACL")
    safe_corr(hd_vals, bas_vals, "HD vs Basin")
    safe_corr(ord_vals, ord_fdc,  "OrdSens vs FDC")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fast", "full", "table"], default="fast")
    p.add_argument("--out_dir", default="results/hd_extended")
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--max_iter", type=int, default=200)
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Start from confirmed results
    hd_all  = {k: {"hd": v["hd"], "full": v["full"], "best_only": v["best_only"]}
                for k, v in HD_CONFIRMED.items()}
    ord_all = {k: {"ordering_sensitivity": v["ordering_sensitivity"],
                    "magnitude_sensitivity": v["magnitude_sensitivity"]}
                for k, v in ORD_CONFIRMED.items()}

    if args.mode == "table":
        print_evidence_table(hd_all, ord_all)
        return

    # Run HD and ordinality for envs with existing checkpoints not yet measured
    run_envs = {k: v for k, v in CKPT_MAP.items() if k not in HD_CONFIRMED}

    if args.mode == "full":
        # Train quick checkpoints for Walker2d and Reacher
        for env_name, (env_id, pdim, rollout_type) in TRAIN_NEEDED.items():
            ckpt = train_quick_ckpt(env_id, pdim,
                                     f"{args.out_dir}/{env_name.lower()}",
                                     n_traj=3000, n_epochs=75,
                                     device=args.device)
            run_envs[env_name] = (env_id, pdim, ckpt, rollout_type)

    for env_name, info in run_envs.items():
        env_id, pdim, ckpt_path, rollout_type = info
        if not os.path.exists(ckpt_path):
            log.warning(f"Checkpoint missing for {env_name}: {ckpt_path}")
            continue

        cache_hd  = f"{args.out_dir}/hd_{env_name.replace('(','').replace(')','')}.json"
        cache_ord = f"{args.out_dir}/ord_{env_name.replace('(','').replace(')','')}.json"

        # History dependence
        if not os.path.exists(cache_hd):
            log.info(f"\n{'='*50}\nHD: {env_name}\n{'='*50}")
            r = run_hd_single(env_name, env_id, pdim, ckpt_path, rollout_type,
                               args.n_seeds, args.max_iter, args.n_episodes,
                               args.device)
            with open(cache_hd, "w") as f:
                json.dump(r, f, indent=2)
        else:
            with open(cache_hd) as f:
                r = json.load(f)
        hd_all[env_name] = {"hd": r["history_dependence"],
                              "full": r["means"]["full"],
                              "best_only": r["means"]["best_only"]}

        # Ordinality
        if not os.path.exists(cache_ord):
            log.info(f"\n{'='*50}\nOrdinality: {env_name}\n{'='*50}")
            r2 = run_ordinality_single(env_name, env_id, pdim, ckpt_path,
                                        rollout_type, args.n_seeds, args.max_iter,
                                        args.n_episodes, args.device)
            with open(cache_ord, "w") as f:
                json.dump(r2, f, indent=2)
        else:
            with open(cache_ord) as f:
                r2 = json.load(f)
        ord_all[env_name] = {"ordering_sensitivity": r2["ordering_sensitivity"],
                               "magnitude_sensitivity": r2["magnitude_sensitivity"]}

    # Save merged results
    merged = {"hd": hd_all, "ordinality": ord_all,
              "fdc": FDC_ALL, "acl": ACL_ALL, "basin": BASIN_ALL}
    with open(f"{args.out_dir}/all_hd_results.json", "w") as f:
        json.dump(merged, f, indent=2)

    # Print evidence table
    print_evidence_table(hd_all, ord_all)

    log.info(f"Results saved to {args.out_dir}/all_hd_results.json")


if __name__ == "__main__":
    main()
