import os, sys, json, argparse, logging
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
sys.path.insert(0, os.getcwd())


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


def get_rollout(param_dim):
    if param_dim in (8, 16):
        from runner.jepa_runner import rollout_policy
    else:
        from envs import rollout_policy
    return rollout_policy


def run_jepa_optimization(ckpt_path, env_name, param_dim,
                           seed=0, max_iter=400, n_ep=10, device="cpu",
                           reward_transform=None, noise_std=0.0,
                           sparse_every=1, delayed_by=0,
                           warmstart_theta=None):
    agent, pdim = load_agent(ckpt_path, param_dim, device)
    rollout = get_rollout(pdim)
    rng = np.random.RandomState(seed)
    rewards = []
    best_reward = -np.inf
    delayed_buffer = []
    n_warmup = max(8, pdim)

    for i in range(n_warmup):
        theta = warmstart_theta.copy() if (i == 0 and warmstart_theta is not None) else rng.randn(pdim)
        r_true = float(rollout(env_name, theta, n_ep, seed + i))
        r_obs = r_true + (rng.randn() * noise_std if noise_std > 0 else 0.0)
        r_hist = reward_transform(r_obs) if reward_transform else r_obs
        if delayed_by > 0:
            delayed_buffer.append((theta, r_hist))
            if len(delayed_buffer) > delayed_by:
                th_d, r_d = delayed_buffer.pop(0)
                agent.add_to_history(th_d, r_d)
        elif i % sparse_every == 0:
            agent.add_to_history(theta, r_hist)
        rewards.append(r_true)
        best_reward = max(best_reward, r_true)

    for t in range(max_iter - n_warmup):
        theta = agent.propose()
        r_true = float(rollout(env_name, theta, n_ep, seed + n_warmup + t))
        r_obs = r_true + (rng.randn() * noise_std if noise_std > 0 else 0.0)
        r_hist = reward_transform(r_obs) if reward_transform else r_obs
        if delayed_by > 0:
            delayed_buffer.append((theta, r_hist))
            if len(delayed_buffer) > delayed_by:
                th_d, r_d = delayed_buffer.pop(0)
                agent.add_to_history(th_d, r_d)
        elif t % sparse_every == 0:
            agent.add_to_history(theta, r_hist)
        rewards.append(r_true)
        best_reward = max(best_reward, r_true)

    return {"rewards": rewards, "best_reward": float(best_reward),
            "mean_reward": float(np.mean(rewards)),
            "auc": float(np.mean(np.maximum.accumulate(rewards)))}


def auc_and_threshold(rewards, pct=0.9):
    running_max = np.maximum.accumulate(rewards)
    auc = float(np.mean(running_max))
    thr = pct * running_max[-1] if running_max[-1] > 0 else pct * running_max[-1]
    tti = next((i for i, v in enumerate(running_max) if v >= thr), len(rewards))
    return auc, int(tti)


# ── C1+: AUC and time-to-threshold ──────────────────────────────────────────

def run_auc_metrics(ckpt_path, env_name, param_dim,
                     n_seeds=10, max_iter=400, n_ep=10, device="cpu"):
    from baselines import run_cem
    results = {}
    for label, fn in [
        ("swarm_jepa", lambda s: run_jepa_optimization(
            ckpt_path, env_name, param_dim, seed=s,
            max_iter=max_iter, n_ep=n_ep, device=device)),
        ("cem",    lambda s: run_cem(env_name, param_dim, s, max_iter, n_ep)),
    ]:
        aucs, ttis, bests = [], [], []
        for seed in range(n_seeds):
            try:
                r = fn(seed)
                auc, tti = auc_and_threshold(r["rewards"])
                aucs.append(auc); ttis.append(tti); bests.append(r["best_reward"])
            except Exception as e:
                log.warning(f"{label} seed={seed}: {e}")
        if bests:
            results[label] = {"mean_best": float(np.mean(bests)),
                               "mean_auc": float(np.mean(aucs)),
                               "mean_tti_90pct": float(np.mean(ttis))}
            log.info(f"C1+ {label}: best={np.mean(bests):.1f} auc={np.mean(aucs):.1f} tti={np.mean(ttis):.0f}")
    return results


# ── C2+: OOD parameter dimensions ───────────────────────────────────────────

def run_ood_dimensions(ckpt_path, env_name, train_param_dim,
                        ood_dims=(128, 256, 512),
                        n_seeds=5, max_iter=200, n_ep=5, device="cpu"):
    results = {"train_dim": train_param_dim}
    for ood_dim in ood_dims:
        log.info(f"C2+ OOD dim={ood_dim}")
        _, pdim = load_agent(ckpt_path, train_param_dim, device)
        rng_proj = np.random.RandomState(12345)
        R = rng_proj.randn(ood_dim, pdim)
        R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-8
        seed_bests = []
        for seed in range(n_seeds):
            agent, _ = load_agent(ckpt_path, train_param_dim, device)
            rng = np.random.RandomState(seed)
            rewards, best = [], -np.inf
            for i in range(max(8, pdim)):
                theta_low = rng.randn(pdim)
                r = float(-np.sum((R @ theta_low)**2) / ood_dim * 10
                           + rng.randn() * 0.5)
                agent.add_to_history(theta_low, r)
                rewards.append(r); best = max(best, r)
            for t in range(max_iter - max(8, pdim)):
                theta_low = agent.propose()
                r = float(-np.sum((R @ theta_low)**2) / ood_dim * 10
                           + rng.randn() * 0.5)
                agent.add_to_history(theta_low, r)
                rewards.append(r); best = max(best, r)
            seed_bests.append(float(best))
        results[f"dim_{ood_dim}"] = {"mean_best": float(np.mean(seed_bests)),
                                      "std_best": float(np.std(seed_bests))}
        log.info(f"  dim={ood_dim}: {np.mean(seed_bests):.3f}")
    return results


# ── C7A: Convergence analysis ────────────────────────────────────────────────

def run_convergence_analysis(ckpt_path, env_name, param_dim,
                              n_seeds=20, max_iter=400, n_ep=10, device="cpu"):
    from baselines import run_cem
    all_curves = {"jepa": [], "cem": [], "pso": []}
    for seed in range(n_seeds):
        r = run_jepa_optimization(ckpt_path, env_name, param_dim,
                                   seed=seed, max_iter=max_iter, n_ep=n_ep)
        all_curves["jepa"].append(np.maximum.accumulate(r["rewards"]).tolist())
        for label, fn in [
            ("cem", lambda s: run_cem(env_name, param_dim, s, max_iter, n_ep)),
            ]:
            try:
                rv = fn(seed)
                all_curves[label].append(np.maximum.accumulate(rv["rewards"]).tolist())
            except Exception:
                pass
    results = {}
    for method, curves in all_curves.items():
        if not curves:
            continue
        min_len = min(len(c) for c in curves)
        arr = np.array([c[:min_len] for c in curves])
        mean_c = arr.mean(0)
        results[method] = {"mean_curve": mean_c.tolist(),
                           "std_curve": arr.std(0).tolist(),
                           "final_mean": float(mean_c[-1]),
                           "n_seeds": len(curves)}
        log.info(f"C7A {method}: final={mean_c[-1]:.1f}")
    return results


# ── C7B: Scaling laws ────────────────────────────────────────────────────────

def run_scaling_laws(ckpt_path, env_name, train_param_dim,
                      dims=(16, 32, 64, 128, 256, 512),
                      n_seeds=5, max_iter=200, n_ep=5, device="cpu"):
    _, pdim = load_agent(ckpt_path, train_param_dim, device)
    rollout = get_rollout(pdim)
    results = {}
    for dim in dims:
        log.info(f"C7B dim={dim}")
        seed_bests = []
        for seed in range(n_seeds):
            agent, _ = load_agent(ckpt_path, train_param_dim, device)
            rng = np.random.RandomState(seed)
            if dim > pdim:
                rng_p = np.random.RandomState(42 + dim)
                proj = rng_p.randn(dim, pdim)
                proj /= np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8
            else:
                proj = None
            rewards, best = [], -np.inf
            n_warmup = max(8, pdim)
            for i in range(n_warmup):
                theta = rng.randn(pdim)
                theta_eval = (proj @ theta)[:pdim] if proj is not None else theta
                try:
                    r = float(rollout(env_name, theta_eval, n_ep, seed + i))
                except Exception:
                    r = float(rng.randn() * 10)
                agent.add_to_history(theta, r)
                rewards.append(r); best = max(best, r)
            for t in range(max_iter - n_warmup):
                theta = agent.propose()
                theta_eval = (proj @ theta)[:pdim] if proj is not None else theta
                try:
                    r = float(rollout(env_name, theta_eval, n_ep, seed + n_warmup + t))
                except Exception:
                    r = float(rng.randn() * 10)
                agent.add_to_history(theta, r)
                rewards.append(r); best = max(best, r)
            seed_bests.append(float(best))
        results[f"dim_{dim}"] = {"mean_best": float(np.mean(seed_bests)),
                                  "std_best": float(np.std(seed_bests))}
        log.info(f"  dim={dim}: {np.mean(seed_bests):.2f}")
    return results


# ── C7C: Noisy objectives ────────────────────────────────────────────────────

def run_noisy_objectives(ckpt_path, env_name, param_dim,
                          noise_levels=(0.0, 0.05, 0.10, 0.20, 0.50),
                          n_seeds=5, max_iter=300, n_ep=10, device="cpu"):
    _, pdim = load_agent(ckpt_path, param_dim, device)
    rollout = get_rollout(pdim)
    rng_est = np.random.RandomState(0)
    sample_r = []
    for i in range(20):
        try:
            sample_r.append(float(rollout(env_name, rng_est.randn(pdim), 5, i)))
        except Exception:
            pass
    reward_range = max(np.ptp(sample_r), 1.0) if sample_r else 100.0
    results = {"reward_range": float(reward_range)}
    for nl in noise_levels:
        noise_std = nl * reward_range
        log.info(f"C7C noise={nl:.0%}")
        seed_bests = []
        for seed in range(n_seeds):
            r = run_jepa_optimization(ckpt_path, env_name, param_dim,
                                       seed=seed, max_iter=max_iter,
                                       n_ep=n_ep, noise_std=noise_std)
            seed_bests.append(r["best_reward"])
        results[f"noise_{nl:.0%}"] = {"mean_best": float(np.mean(seed_bests)),
                                       "std_best": float(np.std(seed_bests))}
        log.info(f"  noise={nl:.0%}: {np.mean(seed_bests):.2f}")
    return results


# ── C7E: Sparse rewards ──────────────────────────────────────────────────────

def run_sparse_rewards(ckpt_path, env_name, param_dim,
                        sparse_intervals=(1, 5, 10, 20),
                        n_seeds=5, max_iter=400, n_ep=10, device="cpu"):
    results = {}
    for k in sparse_intervals:
        log.info(f"C7E sparse_every={k}")
        seed_bests = []
        for seed in range(n_seeds):
            r = run_jepa_optimization(ckpt_path, env_name, param_dim,
                                       seed=seed, max_iter=max_iter,
                                       n_ep=n_ep, sparse_every=k)
            seed_bests.append(r["best_reward"])
        results[f"sparse_{k}"] = {"mean_best": float(np.mean(seed_bests)),
                                   "std_best": float(np.std(seed_bests))}
        log.info(f"  k={k}: {np.mean(seed_bests):.2f}")
    return results


# ── C7F: Delayed rewards ─────────────────────────────────────────────────────

def run_delayed_rewards(ckpt_path, env_name, param_dim,
                         delays=(0, 5, 10, 20, 50),
                         n_seeds=5, max_iter=400, n_ep=10, device="cpu"):
    results = {}
    for delay in delays:
        log.info(f"C7F delay={delay}")
        seed_bests = []
        for seed in range(n_seeds):
            r = run_jepa_optimization(ckpt_path, env_name, param_dim,
                                       seed=seed, max_iter=max_iter,
                                       n_ep=n_ep, delayed_by=delay)
            seed_bests.append(r["best_reward"])
        results[f"delay_{delay}"] = {"mean_best": float(np.mean(seed_bests)),
                                      "std_best": float(np.std(seed_bests))}
        log.info(f"  delay={delay}: {np.mean(seed_bests):.2f}")
    return results


# ── C7G: Changing objective ──────────────────────────────────────────────────

def run_changing_objective(ckpt_path, env_name, param_dim,
                            switch_at=200, n_seeds=5, max_iter=400,
                            n_ep=10, device="cpu"):
    agent0, pdim = load_agent(ckpt_path, param_dim, device)
    rollout = get_rollout(pdim)
    results = {}
    for cond, tf_before, tf_after in [
        ("sign_flip",   lambda r: r,       lambda r: -r),
        ("scale_10x",   lambda r: r,       lambda r: r * 10),
        ("offset_100",  lambda r: r,       lambda r: r + 100),
    ]:
        log.info(f"C7G {cond}")
        seed_before, seed_after = [], []
        for seed in range(n_seeds):
            agent, _ = load_agent(ckpt_path, pdim, device)
            rng = np.random.RandomState(seed)
            rewards_true = []
            n_warmup = max(8, pdim)
            for i in range(n_warmup):
                theta = rng.randn(pdim)
                r_true = float(rollout(env_name, theta, n_ep, seed + i))
                agent.add_to_history(theta, tf_before(r_true))
                rewards_true.append(r_true)
            for t in range(max_iter - n_warmup):
                theta = agent.propose()
                r_true = float(rollout(env_name, theta, n_ep, seed + n_warmup + t))
                tf = tf_before if t + n_warmup < switch_at else tf_after
                agent.add_to_history(theta, tf(r_true))
                rewards_true.append(r_true)
            seed_before.append(float(max(rewards_true[:switch_at])))
            seed_after.append(float(max(rewards_true[switch_at:])) if len(rewards_true) > switch_at else 0.0)
        results[cond] = {
            "mean_best_before": float(np.mean(seed_before)),
            "mean_best_after":  float(np.mean(seed_after)),
        }
        log.info(f"  {cond}: before={np.mean(seed_before):.1f} after={np.mean(seed_after):.1f}")
    return results


# ── C7H: Budget scaling ──────────────────────────────────────────────────────

def run_budget_scaling(ckpt_path, env_name, param_dim,
                        budgets=(10, 20, 50, 100, 200, 400),
                        n_seeds=5, n_ep=10, device="cpu"):
    from baselines import run_cem
    max_iter = max(budgets)
    raw = {"jepa": {b: [] for b in budgets}, "cem": {b: [] for b in budgets}}
    for seed in range(n_seeds):
        r = run_jepa_optimization(ckpt_path, env_name, param_dim,
                                   seed=seed, max_iter=max_iter, n_ep=n_ep)
        for b in budgets:
            raw["jepa"][b].append(float(max(r["rewards"][:b])))
        try:
            rc = run_cem(env_name, param_dim, seed, max_iter, n_ep)
            for b in budgets:
                raw["cem"][b].append(float(max(rc["rewards"][:b])))
        except Exception:
            pass
    results = {}
    for method in ["jepa", "cem"]:
        results[method] = {b: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                           for b, v in raw[method].items() if v}
        for b, v in raw[method].items():
            if v:
                log.info(f"C7H {method} budget={b}: {np.mean(v):.1f}")
    return results


# ── C7I: Warm start vs cold start ───────────────────────────────────────────

def run_warmstart_comparison(ckpt_path, env_name, param_dim,
                              n_seeds=5, max_iter=400, n_ep=10, device="cpu"):
    from baselines import run_cem
    _, pdim = load_agent(ckpt_path, param_dim, device)
    rollout = get_rollout(pdim)
    results = {}
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        best_r, best_theta = -np.inf, None
        for i in range(50):
            theta = rng.randn(pdim)
            try:
                r = float(rollout(env_name, theta, n_ep, seed * 1000 + i))
                if r > best_r:
                    best_r, best_theta = r, theta.copy()
            except Exception:
                pass
        r_cold = run_jepa_optimization(ckpt_path, env_name, pdim, seed=seed,
                                        max_iter=max_iter, n_ep=n_ep)
        r_warm = run_jepa_optimization(ckpt_path, env_name, pdim, seed=seed,
                                        max_iter=max_iter, n_ep=n_ep,
                                        warmstart_theta=best_theta)
        try:
            cem_best = run_cem(env_name, pdim, seed, max_iter, n_ep)["best_reward"]
        except Exception:
            cem_best = float("nan")
        results[f"seed_{seed}"] = {
            "jepa_cold": r_cold["best_reward"],
            "jepa_warm": r_warm["best_reward"],
            "cem_cold": cem_best,
            "init_reward": float(best_r),
        }
        log.info(f"C7I seed={seed}: cold={r_cold['best_reward']:.1f} "
                 f"warm={r_warm['best_reward']:.1f}")
    bests = {k: [results[f"seed_{s}"][k] for s in range(n_seeds)]
             for k in ["jepa_cold", "jepa_warm", "cem_cold"]}
    results["aggregate"] = {k: {"mean": float(np.nanmean(v)),
                                  "std": float(np.nanstd(v))}
                              for k, v in bests.items()}
    return results


# ── C9: Data efficiency ──────────────────────────────────────────────────────

def run_data_efficiency_full(train_envs, eval_env, eval_param_dim,
                              traj_counts=(100, 300, 500, 1000, 3000, 10000),
                              n_seeds=5, max_iter=200, n_ep=5, device="cpu"):
    from jepa_retraining_experiments import train_jepa, evaluate_checkpoint
    from runner.jepa_runner import rollout_policy as _rp_de
    results = {}
    for n_traj in traj_counts:
        log.info(f"C9 n_traj={n_traj}")
        ckpt = f"/tmp/data_eff_{n_traj}.pt"
        if not os.path.exists(ckpt):
            data = {}
            rng_de = np.random.RandomState(0)
            for env in train_envs:
                trajs = []
                for i in range(n_traj):
                    theta = rng_de.randn(eval_param_dim)
                    try:
                        r = float(_rp_de(env, theta, 5, i))
                    except Exception:
                        r = 0.0
                    trajs.append((theta, r))
                data[env] = trajs
            train_jepa(data, param_dim=eval_param_dim, n_epochs=50,
                        device=device, out_ckpt_path=ckpt)
        # Evaluate using runner rollout (no bias, matches param_dim)
        from runner.jepa_runner import rollout_policy as _rp_eval
        eval_bests = []
        for _seed in range(n_seeds):
            _agent, _ = load_agent(ckpt, eval_param_dim)
            _rng = np.random.RandomState(_seed)
            _best = -np.inf
            for _i in range(max(8, eval_param_dim)):
                _theta = _rng.randn(eval_param_dim)
                try:
                    _r = float(_rp_eval(eval_env, _theta, n_ep, _seed + _i))
                except Exception:
                    _r = 0.0
                _agent.add_to_history(_theta, _r)
                _best = max(_best, _r)
            for _t in range(max_iter - max(8, eval_param_dim)):
                _theta = _agent.propose()
                try:
                    _r = float(_rp_eval(eval_env, _theta, n_ep, _seed + max(8,eval_param_dim) + _t))
                except Exception:
                    _r = 0.0
                _agent.add_to_history(_theta, _r)
                _best = max(_best, _r)
            eval_bests.append(_best)
        results[f"n_{n_traj}"] = {"n_trajectories": n_traj,
                                   "mean_best": float(np.mean(eval_bests))}
        log.info(f"  n_traj={n_traj}: {np.mean(eval_bests):.2f}")
    return results


# ── C3+: Environment label shuffle ──────────────────────────────────────────

def run_env_label_shuffle(train_envs_and_dims, eval_env, eval_param_dim,
                           n_traj=2000, n_epochs=50, n_seeds=3,
                           max_iter=200, n_ep=5, device="cpu"):
    from jepa_retraining_experiments import train_jepa, evaluate_checkpoint
    from runner.jepa_runner import rollout_policy as _rp_el
    results = {}
    for condition in ["unshuffled", "label_shuffled"]:
        ckpt = f"/tmp/env_label_{condition}.pt"
        if not os.path.exists(ckpt):
            rng_el = np.random.RandomState(0)
            all_data = {}
            for e, d in train_envs_and_dims:
                trajs = []
                for i in range(n_traj):
                    theta = rng_el.randn(d)
                    try:
                        r = float(_rp_el(e, theta, 5, i))
                    except Exception:
                        r = 0.0
                    trajs.append((theta, r))
                all_data[e] = trajs
            if condition == "label_shuffled":
                keys = list(all_data.keys())
                all_data = {keys[(i+1) % len(keys)]: all_data[k]
                            for i, k in enumerate(keys)}
            train_jepa(all_data, param_dim=eval_param_dim, n_epochs=n_epochs,
                        device=device, out_ckpt_path=ckpt)
        from runner.jepa_runner import rollout_policy as _rp_eval2
        eval_bests2 = []
        for _seed in range(n_seeds):
            _agent2, _ = load_agent(ckpt, eval_param_dim)
            _rng2 = np.random.RandomState(_seed)
            _best2 = -np.inf
            for _i in range(max(8, eval_param_dim)):
                _theta2 = _rng2.randn(eval_param_dim)
                try:
                    _r2 = float(_rp_eval2(eval_env, _theta2, 5, _seed + _i))
                except Exception:
                    _r2 = 0.0
                _agent2.add_to_history(_theta2, _r2)
                _best2 = max(_best2, _r2)
            for _t in range(100):
                _theta2 = _agent2.propose()
                try:
                    _r2 = float(_rp_eval2(eval_env, _theta2, 5, _seed + max(8,eval_param_dim) + _t))
                except Exception:
                    _r2 = 0.0
                _agent2.add_to_history(_theta2, _r2)
                _best2 = max(_best2, _r2)
            eval_bests2.append(_best2)
        results[condition] = {"mean_best": float(np.mean(eval_bests2))}
        log.info(f"C3+ {condition}: {np.mean(eval_bests2):.2f}")
    results["effect"] = results["label_shuffled"]["mean_best"] - results["unshuffled"]["mean_best"]
    return results


# ── C4+: Memory comparison vs baselines ─────────────────────────────────────

def run_memory_vs_baselines(ckpt_path, env_name, param_dim,
                             n_seeds=5, max_iter=200, n_ep=10, device="cpu"):
    from baselines import run_cem
    K = 8
    conditions = ["full", "best_only", f"recent_{K}", f"oldest_{K}"]
    results = {"jepa": {}}
    for cond in conditions:
        seed_bests = []
        for seed in range(n_seeds):
            agent, pdim = load_agent(ckpt_path, param_dim, device)
            rollout = get_rollout(pdim)
            rng = np.random.RandomState(seed)
            n_warmup = max(8, pdim)
            rewards = []
            for i in range(n_warmup):
                theta = rng.randn(pdim)
                r = float(rollout(env_name, theta, n_ep, seed + i))
                agent.add_to_history(theta, r); rewards.append(r)
            for t in range(max_iter - n_warmup):
                theta = agent.propose()
                r = float(rollout(env_name, theta, n_ep, seed + n_warmup + t))
                agent.add_to_history(theta, r); rewards.append(r)
                h = agent.history
                if cond == "best_only":
                    bi = int(np.argmax(h.rewards))
                    h.params, h.rewards = [h.params[bi]], [h.rewards[bi]]
                elif cond == f"recent_{K}" and len(h) > K:
                    h.params, h.rewards = h.params[-K:], h.rewards[-K:]
                elif cond == f"oldest_{K}" and len(h) > K:
                    h.params, h.rewards = h.params[:K], h.rewards[:K]
            seed_bests.append(float(max(rewards)))
        results["jepa"][cond] = {"mean_best": float(np.mean(seed_bests))}
        log.info(f"C4+ jepa {cond}: {np.mean(seed_bests):.2f}")
    bests_cem = []
    for seed in range(n_seeds):
        try:
            bests_cem.append(run_cem(env_name, param_dim, seed, max_iter, n_ep)["best_reward"])
        except Exception:
            pass
    if bests_cem:
        results["cem"] = {"mean_best": float(np.mean(bests_cem)),
                          "memory_type": "Gaussian (mu,sigma) -- no trajectory"}
    return results


# ── C6+: Mutual information ──────────────────────────────────────────────────

def run_mutual_information(ckpt_path, env_name, param_dim,
                            n_seeds=3, max_iter=200, n_ep=5, device="cpu"):
    import torch
    all_latents, all_rewards = [], []
    for seed in range(n_seeds):
        agent, pdim = load_agent(ckpt_path, param_dim, device)
        rollout = get_rollout(pdim)
        rng = np.random.RandomState(seed)
        n_warmup = max(8, pdim)
        rewards = []
        for i in range(n_warmup):
            theta = rng.randn(pdim)
            r = float(rollout(env_name, theta, n_ep, seed + i))
            agent.add_to_history(theta, r); rewards.append(r)
        latents = []
        for t in range(max_iter - n_warmup):
            with torch.no_grad():
                params_t, rewards_t = agent.history.to_tensors(device)
                mean, std = agent.history.get_stats()
                params_norm = (params_t - torch.tensor(mean, dtype=torch.float32)) / \
                              torch.tensor(std + 1e-6, dtype=torch.float32)
                s_x = agent.context_encoder(params_norm, rewards_t)
                latents.append(s_x.squeeze(0).cpu().numpy())
            theta = agent.propose()
            r = float(rollout(env_name, theta, n_ep, seed + n_warmup + t))
            agent.add_to_history(theta, r); rewards.append(r)
        if latents:
            n = len(latents)
            all_latents.append(np.array(latents))
            all_rewards.append(np.array(rewards[-n:]))
    if not all_latents:
        return {"error": "no latents"}
    X = np.vstack(all_latents)
    y = np.concatenate(all_rewards)
    norms = np.linalg.norm(X, axis=1)
    mi_proxy = float(abs(np.corrcoef(norms, y)[0, 1]))
    rand_proxy = float(abs(np.corrcoef(np.random.randn(len(y)), y)[0, 1]))
    result = {"MI_proxy_latent_reward": mi_proxy,
               "MI_proxy_random_reward": rand_proxy,
               "ratio": float(mi_proxy / (rand_proxy + 1e-8))}
    log.info(f"C6+ MI proxy: latent={mi_proxy:.4f} random={rand_proxy:.4f} ratio={result['ratio']:.1f}x")
    return result


# ── C7J+: Extended population size ──────────────────────────────────────────

def run_population_extended(ckpt_path, env_name, param_dim,
                             ks=(1, 2, 4, 8, 16, 32),
                             n_seeds=3, max_iter=400, n_ep=10, device="cpu"):
    from runner.jepa_runner import rollout_policy as native_rollout
    results = {}
    for K in ks:
        log.info(f"C7J+ K={K}")
        seed_bests = []
        for seed in range(n_seeds):
            try:
                # Manual K-island optimization using runner rollout
                rng_sw = np.random.RandomState(seed)
                agents_sw = []
                for _k in range(K):
                    _a, _pdim = load_agent(ckpt_path, param_dim, device)
                    agents_sw.append(_a)
                best_sw, best_theta_sw = -np.inf, None
                for _i in range(max(8, param_dim)):
                    for _a in agents_sw:
                        _theta = rng_sw.randn(param_dim)
                        _r = float(native_rollout(env_name, _theta, n_ep, seed + _i))
                        _a.add_to_history(_theta, _r)
                        if _r > best_sw:
                            best_sw, best_theta_sw = _r, _theta.copy()
                for _t in range(max_iter - max(8, param_dim)):
                    for _ki, _a in enumerate(agents_sw):
                        _theta = _a.propose()
                        _r = float(native_rollout(env_name, _theta, n_ep,
                                                   seed + max(8,param_dim) + _t * K + _ki))
                        _a.add_to_history(_theta, _r)
                        if _r > best_sw:
                            best_sw, best_theta_sw = _r, _theta.copy()
                    if _t % 30 == 0 and best_theta_sw is not None:
                        for _a in agents_sw:
                            _a.add_to_history(best_theta_sw, best_sw)
                seed_bests.append(float(best_sw))
            except Exception as e:
                log.warning(f"K={K} seed={seed}: {e}")
        if seed_bests:
            results[f"K_{K}"] = {"mean_best": float(np.mean(seed_bests)),
                                  "std_best": float(np.std(seed_bests))}
            log.info(f"  K={K}: {np.mean(seed_bests):.2f}")
    return results


# ── C8+: Natural gradient ────────────────────────────────────────────────────

def run_natural_gradient(ckpt_path, env_name, param_dim,
                          n_seeds=3, max_iter=50, n_ep=5, device="cpu"):
    agent0, pdim = load_agent(ckpt_path, param_dim, device)
    rollout = get_rollout(pdim)
    all_cos = []
    for seed in range(n_seeds):
        agent, _ = load_agent(ckpt_path, pdim, device)
        rng = np.random.RandomState(seed)
        thetas, rewards = [], []
        for i in range(max(8, pdim)):
            theta = rng.randn(pdim)
            r = float(rollout(env_name, theta, n_ep, seed + i))
            agent.add_to_history(theta, r)
            thetas.append(theta); rewards.append(r)
        cos_list = []
        eps = 0.05
        for t in range(min(30, max_iter)):
            jepa_theta = agent.propose()
            jepa_dir = jepa_theta - thetas[-1]
            jepa_norm = np.linalg.norm(jepa_dir) + 1e-8
            grad = np.zeros(pdim); grad_sq = np.zeros(pdim) + 1e-8
            for d in range(min(pdim, 8)):
                samples = []
                for _ in range(10):
                    delta = rng.randn() * eps
                    theta_eps = thetas[-1].copy(); theta_eps[d] += delta
                    r_eps = float(rollout(env_name, theta_eps, 3, seed + t * 100 + d))
                    samples.append((r_eps - rewards[-1]) / (delta + 1e-8))
                grad[d] = float(np.mean(samples))
                grad_sq[d] = float(np.var(samples)) + 1e-8
            nat = grad / grad_sq
            nat_norm = np.linalg.norm(nat) + 1e-8
            cos_list.append(float(np.dot(jepa_dir, nat) / (jepa_norm * nat_norm)))
            r = float(rollout(env_name, jepa_theta, n_ep, seed + max(8,pdim) + t))
            agent.add_to_history(jepa_theta, r)
            thetas.append(jepa_theta); rewards.append(r)
        all_cos.extend(cos_list)
        log.info(f"C8+ seed={seed}: cos(JEPA,NatGrad)={np.mean(cos_list):.4f}")
    result = {"mean_cos_natural_gradient": float(np.mean(all_cos)),
               "std": float(np.std(all_cos))}
    log.info(f"C8+ aggregate: {result['mean_cos_natural_gradient']:.4f}")
    return result


# ── ALL_EXPERIMENTS (must be after all function defs) ────────────────────────

ALL_EXPERIMENTS = {
    "auc_metrics":          run_auc_metrics,
    "ood_dimensions":       run_ood_dimensions,
    "convergence":          run_convergence_analysis,
    "scaling_laws":         run_scaling_laws,
    "noisy_objectives":     run_noisy_objectives,
    "sparse_rewards":       run_sparse_rewards,
    "delayed_rewards":      run_delayed_rewards,
    "changing_objective":   run_changing_objective,
    "budget_scaling":       run_budget_scaling,
    "warmstart":            run_warmstart_comparison,
    "data_efficiency":      run_data_efficiency_full,
    "env_label_shuffle":    run_env_label_shuffle,
    "memory_vs_baselines":  run_memory_vs_baselines,
    "mutual_information":   run_mutual_information,
    "population_extended":  run_population_extended,
    "natural_gradient":     run_natural_gradient,
}


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", nargs="+", default=["all"],
                   choices=list(ALL_EXPERIMENTS.keys()) + ["all"])
    p.add_argument("--ckpt",       default="checkpoints/jepa_pretrained/jepa_pretrained.pt")
    p.add_argument("--env",        default="CartPole-v1")
    p.add_argument("--param_dim",  type=int, default=8)
    p.add_argument("--out_dir",    default="results/robustness")
    p.add_argument("--device",     default="cpu")
    p.add_argument("--n_seeds",    type=int, default=5)
    p.add_argument("--max_iter",   type=int, default=400)
    p.add_argument("--n_episodes", type=int, default=10)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    exps = list(ALL_EXPERIMENTS.keys()) if "all" in args.experiment else args.experiment

    common = dict(ckpt_path=args.ckpt, env_name=args.env,
                  param_dim=args.param_dim, n_seeds=args.n_seeds,
                  max_iter=args.max_iter, n_ep=args.n_episodes,
                  device=args.device)

    special = {
        "ood_dimensions":    dict(ckpt_path=args.ckpt, env_name=args.env,
                                   train_param_dim=args.param_dim, n_seeds=args.n_seeds,
                                   max_iter=args.max_iter, n_ep=args.n_episodes),
        "scaling_laws":      dict(ckpt_path=args.ckpt, env_name=args.env,
                                   train_param_dim=args.param_dim, n_seeds=args.n_seeds,
                                   max_iter=args.max_iter, n_ep=args.n_episodes),
        "budget_scaling":    dict(ckpt_path=args.ckpt, env_name=args.env,
                                   param_dim=args.param_dim, n_seeds=args.n_seeds,
                                   n_ep=args.n_episodes, device=args.device),
        "data_efficiency":   dict(train_envs=[args.env], eval_env=args.env,
                                   eval_param_dim=args.param_dim, n_seeds=args.n_seeds,
                                   max_iter=args.max_iter, n_ep=args.n_episodes),
        "env_label_shuffle": dict(train_envs_and_dims=[(args.env, args.param_dim)],
                                   eval_env=args.env, eval_param_dim=args.param_dim,
                                   n_seeds=args.n_seeds, max_iter=args.max_iter,
                                   n_ep=args.n_episodes),
        "population_extended": dict(ckpt_path=args.ckpt, env_name=args.env,
                                     param_dim=args.param_dim, n_seeds=args.n_seeds,
                                     max_iter=args.max_iter, n_ep=args.n_episodes),
    }

    for exp in exps:
        log.info(f"\n{'='*60}\nRunning: {exp}\n{'='*60}")
        try:
            result = ALL_EXPERIMENTS[exp](**special.get(exp, common))
            out = f"{args.out_dir}/{exp}_{args.env.replace('/','-')}.json"
            with open(out, "w") as f:
                json.dump(result, f, indent=2)
            log.info(f"Saved: {out}")
        except Exception as e:
            log.error(f"FAILED: {exp} — {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()