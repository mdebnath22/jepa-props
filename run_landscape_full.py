"""
run_landscape_full.py — Runs landscape analysis across all 15 ProPS environments
and produces 5 publication-ready tables.

Table 1: Landscape ruggedness metrics for all 15 environments
Table 2: History Dependence by memory condition (5 envs with checkpoints)
Table 3: Reward Ordinality Test results (5 envs with checkpoints)
Table 4: Correlations between History Dependence and landscape metrics
Table 5: Full JEPA performance summary across all experiments

USAGE:
    python run_landscape_full.py \
        --ckpt_cartpole checkpoints/jepa_pretrained/jepa_pretrained.pt \
        --ckpt_swimmer  checkpoints/jepa_pretrained_swimmer/jepa_pretrained.pt \
        --ckpt_hopper   results/hopper_ckpt/checkpoints/shuffle_none.pt \
        --ckpt_mc       results/mountaincar_ckpt/checkpoints/shuffle_none.pt \
        --ckpt_ip       results/invpendulum_ckpt/checkpoints/shuffle_none.pt \
        --out_dir results/landscape_full \
        --n_seeds 5 --n_samples 300 --n_episodes 10
"""

import os, sys, json, argparse, logging
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
sys.path.insert(0, os.getcwd())

# ─────────────────────────────────────────────────────────────────────────────
# Environment registry — all 15 ProPS domains
# ─────────────────────────────────────────────────────────────────────────────

ENV_REGISTRY = {
    # (display_name, gym_id_or_custom, param_dim, domain_type, ckpt_key)
    "CartPole":          ("CartPole-v1",                 8,  "balance",     "cartpole"),
    "Swimmer":           ("Swimmer-v4",                  16, "locomotion",  "swimmer"),
    "Hopper":            ("Hopper-v4",                   36, "locomotion",  "hopper"),
    "Walker2d":          ("Walker2d-v4",                 24, "locomotion",  None),
    "Reacher":           ("Reacher-v4",                  28, "control",     None),
    "MountainCar(C)":    ("MountainCarContinuous-v0",    3,  "control",     "mc"),
    "InvPendulum":       ("InvertedPendulum-v4",         5,  "balance",     "ip"),
    "InvDblPendulum":    ("InvertedDoublePendulum-v4",   9,  "balance",     None),
    "FrozenLake":        ("FrozenLake-v1",               16, "discrete",    None),
    "CliffWalking":      ("CliffWalking-v0",             48, "discrete",    None),
    "Maze":              ("Maze",                        16, "custom",      None),
    "Nim":               ("Nim",                        84, "custom",      None),
    "Navigation":        ("Navigation",                  14, "custom",      None),
    "BallLanding":       ("BallLanding",                  9, "custom",      None),
    "MountainCar(D)":    ("MountainCar-v0",              64, "discrete",    None),
}


def get_rollout_fn(env_name, param_dim):
    """Returns appropriate rollout function for this env's param_dim."""
    try:
        from runner.jepa_runner import rollout_policy as runner_rp
        def rp(env, theta, n_ep, seed):
            try:
                return runner_rp(env, theta, n_ep, seed)
            except Exception:
                from envs import rollout_policy as envs_rp
                return envs_rp(env, theta, n_ep, seed)
        return rp
    except Exception:
        from envs import rollout_policy as envs_rp
        return envs_rp


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
            log.info(f"param_dim={inferred} inferred from checkpoint (arg={param_dim})")
            param_dim = inferred
    agent = JEPAAgent(param_dim=param_dim, device=device)
    agent.load_state_dict(sd, strict=False)
    agent.eval()
    return agent, param_dim


# ─────────────────────────────────────────────────────────────────────────────
# Ruggedness (runs on ALL 15 environments)
# ─────────────────────────────────────────────────────────────────────────────

def measure_ruggedness(env_id, param_dim, n_samples=300, n_episodes=5,
                        seed=0, eps=0.1):
    rollout = get_rollout_fn(env_id, param_dim)
    rng = np.random.RandomState(seed)

    # Collect random samples
    thetas, rewards = [], []
    for i in range(n_samples):
        theta = rng.randn(param_dim)
        try:
            r = rollout(env_id, theta, n_episodes, seed + i)
            thetas.append(theta)
            rewards.append(r)
        except Exception as e:
            log.warning(f"  rollout failed: {e}")
    if len(rewards) < 10:
        return None
    thetas = np.array(thetas)
    rewards = np.array(rewards)

    # FDC
    best_idx = int(np.argmax(rewards))
    distances = np.linalg.norm(thetas - thetas[best_idx], axis=1)
    fdc = float(np.corrcoef(rewards, -distances)[0, 1]) \
        if rewards.std() > 1e-6 else float("nan")

    # Random walk ACL
    walk_len = min(80, n_samples)
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
        for lag in range(1, min(15, walk_len // 2)):
            i1 = np.where(valid)[0]
            pairs = [(walk_r[i], walk_r[i+lag]) for i in i1 if i+lag < len(walk_r) and valid[i+lag]]
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
    for i in range(min(30, len(thetas))):
        r0 = rewards[i]
        for _ in range(2):
            delta = rng.randn(param_dim) * eps
            try:
                rp = rollout(env_id, thetas[i] + delta, n_episodes, seed + 70000 + i)
                smooth_vals.append(abs(rp - r0) / (np.linalg.norm(delta) + 1e-8))
            except Exception:
                pass
    local_smoothness = float(np.mean(smooth_vals)) if smooth_vals else float("nan")

    # Basin size
    r_best = float(rewards.max())
    r_thresh = 0.9 * r_best if r_best > 0 else 1.1 * r_best
    n_basin = min(20, n_samples // 10)
    basin_hits = 0
    for b in range(n_basin):
        theta = rng.randn(param_dim)
        try:
            r = rollout(env_id, theta, n_episodes, seed + 80000 + b)
        except Exception:
            continue
        for step in range(15):
            cand = theta + rng.randn(param_dim) * 0.3
            try:
                rc = rollout(env_id, cand, n_episodes, seed + 80000 + b*100 + step)
                if rc > r:
                    theta, r = cand, rc
                if r >= r_thresh:
                    basin_hits += 1
                    break
            except Exception:
                break
    basin = float(basin_hits / n_basin) if n_basin > 0 else float("nan")

    return {
        "fdc": fdc, "acl": acl,
        "local_smoothness": local_smoothness, "basin_size": basin,
        "reward_mean": float(rewards.mean()), "reward_std": float(rewards.std()),
        "reward_max": float(rewards.max()), "n_samples": len(rewards),
    }


# ─────────────────────────────────────────────────────────────────────────────
# History Dependence (runs on envs WITH checkpoints only)
# ─────────────────────────────────────────────────────────────────────────────

def measure_history_dependence(ckpt_path, env_id, param_dim,
                                n_seeds=5, max_iter=200, n_episodes=5,
                                device="cpu"):
    rollout = get_rollout_fn(env_id, param_dim)
    K = 8
    conditions = ["full", "best_only", f"recent_{K}", f"top_{K}",
                  f"oldest_{K}", f"random_{K}"]
    cond_bests = {c: [] for c in conditions}

    for seed in range(n_seeds):
        agent, pdim = load_agent(ckpt_path, param_dim, device)
        rng = np.random.RandomState(seed)
        n_warmup = max(16, pdim * 2)

        thetas, rewards = [], []
        for i in range(n_warmup):
            theta = rng.randn(pdim)
            try:
                r = rollout(env_id, theta, n_episodes, seed + i)
            except Exception:
                r = 0.0
            agent.add_to_history(theta, r)
            thetas.append(theta)
            rewards.append(r)

        for condition in conditions:
            # Reset agent state for each condition (same warmup)
            agent2, _ = load_agent(ckpt_path, param_dim, device)
            for th, rw in zip(thetas, rewards):
                agent2.add_to_history(th, rw)

            run_rewards = list(rewards)
            for t in range(max_iter - n_warmup):
                theta = agent2.propose()
                try:
                    r = rollout(env_id, theta, n_episodes, seed + n_warmup + t)
                except Exception:
                    r = min(rewards) if rewards else 0.0
                agent2.add_to_history(theta, r)
                run_rewards.append(r)

                # Apply memory condition
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

            cond_bests[condition].append(float(max(run_rewards)))

    results = {c: {"mean": float(np.mean(v)), "std": float(np.std(v))}
               for c, v in cond_bests.items()}
    r_full = results["full"]["mean"]
    r_best = results["best_only"]["mean"]
    hd = float((r_full - r_best) / (abs(r_full) + 1e-6))
    results["history_dependence"] = hd
    results["recency_bias"] = float(
        (results[f"recent_{K}"]["mean"] - results[f"oldest_{K}"]["mean"]) /
        (abs(r_full) + 1e-6))
    results["quality_bias"] = float(
        (results[f"top_{K}"]["mean"] - results[f"oldest_{K}"]["mean"]) /
        (abs(r_full) + 1e-6))
    log.info(f"  HD={hd:.3f} full={r_full:.1f} best_only={r_best:.1f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Reward Ordinality (runs on envs WITH checkpoints only)
# ─────────────────────────────────────────────────────────────────────────────

def measure_ordinality(ckpt_path, env_id, param_dim,
                        n_seeds=5, max_iter=200, n_episodes=5,
                        device="cpu"):
    rollout = get_rollout_fn(env_id, param_dim)

    transforms = {
        "original": lambda r: r,
        "ranked": lambda r: np.argsort(np.argsort(r)).astype(float) + 1,
        "monotonic": lambda r: np.sign(r) * np.log1p(np.abs(r)),
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
                r = rollout(env_id, theta, n_episodes, seed + i)
            except Exception:
                r = 0.0
            warmup_thetas.append(theta)
            warmup_rewards.append(r)

        for tname, tfn in transforms.items():
            agent, _ = load_agent(ckpt_path, param_dim, device)
            transformed = tfn(np.array(warmup_rewards))
            for th, rt in zip(warmup_thetas, transformed):
                agent.add_to_history(th, float(rt))

            rewards = list(warmup_rewards)
            for t in range(max_iter - n_warmup):
                theta = agent.propose()
                try:
                    r_real = rollout(env_id, theta, n_episodes, seed + n_warmup + t)
                except Exception:
                    r_real = min(rewards) if rewards else 0.0
                all_real = np.array(rewards + [r_real])
                r_transformed = float(tfn(all_real)[-1])
                agent.add_to_history(theta, r_transformed)
                rewards.append(r_real)
            t_bests[tname].append(float(max(rewards)))

    results = {t: {"mean": float(np.mean(v)), "std": float(np.std(v))}
               for t, v in t_bests.items()}
    orig = results["original"]["mean"]
    perm = results["permuted"]["mean"]
    rank = results["ranked"]["mean"]
    ordering_sensitivity = float((orig - perm) / (abs(orig) + 1e-6))
    magnitude_sensitivity = float((orig - rank) / (abs(orig) + 1e-6))
    results["ordering_sensitivity"] = ordering_sensitivity
    results["magnitude_sensitivity"] = magnitude_sensitivity
    log.info(f"  ord_sens={ordering_sensitivity:.3f} mag_sens={magnitude_sensitivity:.3f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Table formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_table1_ruggedness(all_rug):
    """Table 1: Landscape metrics for all 15 environments."""
    rows = [
        f"{'Environment':<22} {'Domain':<12} {'FDC':>6} {'ACL':>5} "
        f"{'Smooth':>8} {'Basin':>7} {'MaxR':>8}"
    ]
    rows.append("-" * 75)
    for name, (env_id, pdim, domain, _) in ENV_REGISTRY.items():
        d = all_rug.get(name, {})
        fdc = f"{d['fdc']:.3f}" if d.get('fdc') is not None and not np.isnan(d.get('fdc', float('nan'))) else "—"
        acl = f"{d['acl']:.0f}" if d.get('acl') is not None and not np.isnan(d.get('acl', float('nan'))) else "—"
        sm = f"{d['local_smoothness']:.2f}" if d.get('local_smoothness') is not None and not np.isnan(d.get('local_smoothness', float('nan'))) else "—"
        ba = f"{d['basin_size']:.3f}" if d.get('basin_size') is not None and not np.isnan(d.get('basin_size', float('nan'))) else "—"
        mx = f"{d['reward_max']:.1f}" if d.get('reward_max') is not None else "—"
        rows.append(f"{name:<22} {domain:<12} {fdc:>6} {acl:>5} {sm:>8} {ba:>7} {mx:>8}")
    return "\n".join(rows)


def format_table2_memory(all_mem):
    """Table 2: History Dependence by condition."""
    K = 8
    conditions = ["full", "best_only", f"recent_{K}", f"top_{K}", f"oldest_{K}"]
    rows = [f"{'Environment':<22} " + " ".join(f"{c:>12}" for c in ["HD"] + conditions)]
    rows.append("-" * (22 + 13 * (len(conditions) + 1)))
    for name, d in all_mem.items():
        hd = f"{d['history_dependence']:.3f}"
        vals = []
        for c in conditions:
            v = d.get(c, {}).get("mean")
            vals.append(f"{v:.1f}" if v is not None else "—")
        rows.append(f"{name:<22} {hd:>12} " + " ".join(f"{v:>12}" for v in vals))
    return "\n".join(rows)


def format_table3_ordinality(all_ord):
    """Table 3: Reward ordinality test results."""
    rows = [f"{'Environment':<22} {'Original':>10} {'Ranked':>8} {'Monotonic':>10} "
            f"{'Permuted':>9} {'OrdSens':>8} {'MagSens':>8}"]
    rows.append("-" * 80)
    for name, d in all_ord.items():
        orig = d.get("original", {}).get("mean", float("nan"))
        rank = d.get("ranked", {}).get("mean", float("nan"))
        mono = d.get("monotonic", {}).get("mean", float("nan"))
        perm = d.get("permuted", {}).get("mean", float("nan"))
        os_ = d.get("ordering_sensitivity", float("nan"))
        ms_ = d.get("magnitude_sensitivity", float("nan"))
        rows.append(
            f"{name:<22} {orig:>10.2f} {rank:>8.2f} {mono:>10.2f} "
            f"{perm:>9.2f} {os_:>8.3f} {ms_:>8.3f}"
        )
    return "\n".join(rows)


def format_table4_correlations(all_rug, all_mem, all_ord):
    """Table 4: Correlations between HD/OrdSens and landscape metrics."""
    envs = [e for e in all_rug if e in all_mem]
    if len(envs) < 3:
        return f"Insufficient data for correlation analysis (need ≥3 envs, have {len(envs)})"

    hd = np.array([all_mem[e]["history_dependence"] for e in envs])
    fdc = np.array([all_rug[e].get("fdc", float("nan")) for e in envs])
    acl = np.array([all_rug[e].get("acl", float("nan")) for e in envs])
    basin = np.array([all_rug[e].get("basin_size", float("nan")) for e in envs])
    smooth = np.array([all_rug[e].get("local_smoothness", float("nan")) for e in envs])

    ord_envs = [e for e in envs if e in all_ord]
    ord_sens = np.array([all_ord[e]["ordering_sensitivity"] for e in ord_envs])
    ord_fdc = np.array([all_rug[e].get("fdc", float("nan")) for e in ord_envs])

    def corr(x, y):
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 3:
            return float("nan"), mask.sum()
        return float(np.corrcoef(x[mask], y[mask])[0, 1]), int(mask.sum())

    rows = [f"{'Metric Pair':<40} {'r':>7} {'n':>4}"]
    rows.append("-" * 55)
    for pair, x, y in [
        ("HD vs FDC", hd, fdc),
        ("HD vs ACL", hd, acl),
        ("HD vs Basin Size", hd, basin),
        ("HD vs -Smoothness (neg)", hd, -smooth),
        ("OrdSens vs FDC", ord_sens, ord_fdc),
    ]:
        r, n = corr(x, y)
        rows.append(f"{pair:<40} {r:>7.3f} {n:>4}")

    return "\n".join(rows)


def format_table5_summary(all_rug, all_mem, all_ord):
    """Table 5: Full summary combining all metrics per environment."""
    rows = [
        f"{'Environment':<22} {'Domain':<12} {'FDC':>6} {'ACL':>5} "
        f"{'HD':>7} {'OrdSens':>8} {'MagSens':>8} {'HasJEPA':>8}"
    ]
    rows.append("-" * 85)
    for name, (env_id, pdim, domain, ckpt_key) in ENV_REGISTRY.items():
        rug = all_rug.get(name, {})
        mem = all_mem.get(name, {})
        ord_ = all_ord.get(name, {})
        fdc = f"{rug['fdc']:.3f}" if rug.get('fdc') is not None and not np.isnan(rug.get('fdc', float('nan'))) else "—"
        acl = f"{rug['acl']:.0f}" if rug.get('acl') is not None and not np.isnan(rug.get('acl', float('nan'))) else "—"
        hd = f"{mem['history_dependence']:.3f}" if mem.get('history_dependence') is not None else "—"
        os_ = f"{ord_['ordering_sensitivity']:.3f}" if ord_.get('ordering_sensitivity') is not None else "—"
        ms_ = f"{ord_['magnitude_sensitivity']:.3f}" if ord_.get('magnitude_sensitivity') is not None else "—"
        has_j = "✓" if ckpt_key else "—"
        rows.append(f"{name:<22} {domain:<12} {fdc:>6} {acl:>5} {hd:>7} {os_:>8} {ms_:>8} {has_j:>8}")
    return "\n".join(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_cartpole", default="checkpoints/jepa_pretrained/jepa_pretrained.pt")
    p.add_argument("--ckpt_swimmer",  default="checkpoints/jepa_pretrained_swimmer/jepa_pretrained.pt")
    p.add_argument("--ckpt_hopper",   default="results/hopper_ckpt/checkpoints/shuffle_none.pt")
    p.add_argument("--ckpt_mc",       default="results/mountaincar_ckpt/checkpoints/shuffle_none.pt")
    p.add_argument("--ckpt_ip",       default="results/invpendulum_ckpt/checkpoints/shuffle_none.pt")
    p.add_argument("--out_dir",       default="results/landscape_full")
    p.add_argument("--device",        default="cpu")
    p.add_argument("--n_seeds",       type=int, default=5)
    p.add_argument("--n_samples",     type=int, default=300)
    p.add_argument("--n_episodes",    type=int, default=10)
    p.add_argument("--max_iter",      type=int, default=200)
    p.add_argument("--skip_ruggedness", action="store_true")
    p.add_argument("--skip_memory",     action="store_true")
    p.add_argument("--skip_ordinality", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_map = {
        "cartpole": args.ckpt_cartpole,
        "swimmer":  args.ckpt_swimmer,
        "hopper":   args.ckpt_hopper,
        "mc":       args.ckpt_mc,
        "ip":       args.ckpt_ip,
    }

    all_rug, all_mem, all_ord = {}, {}, {}

    # ── Table 1: Ruggedness for all 15 envs ──────────────────────────────────
    if not args.skip_ruggedness:
        log.info("\n" + "="*60 + "\nTABLE 1: Landscape Ruggedness (all 15 envs)\n" + "="*60)
        for name, (env_id, pdim, domain, ckpt_key) in ENV_REGISTRY.items():
            log.info(f"\n--- {name} ({env_id}, param_dim={pdim}) ---")
            cache = f"{args.out_dir}/rug_{name.replace('/', '_').replace('(','').replace(')','')}.json"
            if os.path.exists(cache):
                with open(cache) as f:
                    all_rug[name] = json.load(f)
                log.info(f"  Loaded from cache")
            else:
                r = measure_ruggedness(env_id, pdim, args.n_samples,
                                        args.n_episodes, seed=0)
                if r is not None:
                    all_rug[name] = r
                    with open(cache, "w") as f:
                        json.dump(r, f, indent=2)
                else:
                    log.warning(f"  FAILED -- skipping {name}")

    # ── Tables 2-3: HD and Ordinality for envs with checkpoints ──────────────
    for name, (env_id, pdim, domain, ckpt_key) in ENV_REGISTRY.items():
        if ckpt_key is None:
            continue
        ckpt = ckpt_map.get(ckpt_key)
        if not ckpt or not os.path.exists(ckpt):
            log.warning(f"No checkpoint for {name} (key={ckpt_key}) -- skipping HD/Ord")
            continue

        if not args.skip_memory:
            log.info(f"\n--- HD: {name} ---")
            cache = f"{args.out_dir}/mem_{name.replace('/', '_').replace('(','').replace(')','')}.json"
            if os.path.exists(cache):
                with open(cache) as f:
                    all_mem[name] = json.load(f)
                log.info(f"  Loaded from cache")
            else:
                r = measure_history_dependence(ckpt, env_id, pdim,
                                                args.n_seeds, args.max_iter,
                                                args.n_episodes, args.device)
                all_mem[name] = r
                with open(cache, "w") as f:
                    json.dump(r, f, indent=2)

        if not args.skip_ordinality:
            log.info(f"\n--- Ordinality: {name} ---")
            cache = f"{args.out_dir}/ord_{name.replace('/', '_').replace('(','').replace(')','')}.json"
            if os.path.exists(cache):
                with open(cache) as f:
                    all_ord[name] = json.load(f)
                log.info(f"  Loaded from cache")
            else:
                r = measure_ordinality(ckpt, env_id, pdim,
                                        args.n_seeds, args.max_iter,
                                        args.n_episodes, args.device)
                all_ord[name] = r
                with open(cache, "w") as f:
                    json.dump(r, f, indent=2)

    # ── Print all 5 tables ────────────────────────────────────────────────────
    tables = {
        "Table 1: Landscape Ruggedness (all 15 environments)":
            format_table1_ruggedness(all_rug),
        "Table 2: History Dependence by Memory Condition":
            format_table2_memory(all_mem),
        "Table 3: Reward Ordinality Test":
            format_table3_ordinality(all_ord),
        "Table 4: Correlations — History Dependence vs Landscape Metrics":
            format_table4_correlations(all_rug, all_mem, all_ord),
        "Table 5: Full Summary":
            format_table5_summary(all_rug, all_mem, all_ord),
    }

    out_text = ""
    for title, table in tables.items():
        block = f"\n{'='*75}\n{title}\n{'='*75}\n{table}\n"
        print(block)
        out_text += block

    with open(f"{args.out_dir}/all_tables.txt", "w") as f:
        f.write(out_text)

    # Save combined JSON
    with open(f"{args.out_dir}/all_results.json", "w") as f:
        json.dump({"ruggedness": all_rug, "memory": all_mem,
                   "ordinality": all_ord}, f, indent=2)

    log.info(f"\nAll tables saved to {args.out_dir}/all_tables.txt")
    log.info(f"All results saved to {args.out_dir}/all_results.json")


if __name__ == "__main__":
    main()
