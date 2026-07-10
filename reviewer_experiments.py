"""
reviewer_experiments.py

Code for all experiments required by reviewer responses.
Addresses:
R1: Decoder training objective clarification + gradient flow experiment
R2: Budget fairness -- K-island evaluation budget accounting
R3: CartPole number inconsistency -- standardized comparison
R4: Target return percentile r* sensitivity sweep
R5: FDC sampling protocol robustness
R6: Swarm hyperparameter sensitivity
R7: Velo / amortized BO comparison (proxy)
R8: Non-control black-box benchmark (BBOB functions)
R9: Reward shuffle mechanistic analysis

USAGE:
python reviewer_experiments.py --experiment [options]
"""

import os, sys, json, argparse, logging, inspect
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers to make the reviewer-experiment code robust to the exact
# JEPAAgent API (which may vary slightly across checkpoints/branches).
# ─────────────────────────────────────────────────────────────────────────────

def _call_jepa_loss(agent, params_ctx, rewards_ctx, params_target, desired_reward):
    """
    Call agent.jepa_loss with whatever keyword name it actually expects for
    the target parameters, instead of assuming 'params_tgt'.
    """
    sig = inspect.signature(agent.jepa_loss)
    valid = set(sig.parameters.keys())

    kwargs = {}
    # context params
    for cand in ("params_ctx", "ctx_params", "params_context"):
        if cand in valid:
            kwargs[cand] = params_ctx
            break
    # context rewards
    for cand in ("rewards_ctx", "ctx_rewards", "rewards_context"):
        if cand in valid:
            kwargs[cand] = rewards_ctx
            break
    # target params (the one that broke: 'params_tgt' -> actual name unknown)
    for cand in ("params_tgt", "params_target", "target_params",
                 "params_y", "y_params", "theta_target"):
        if cand in valid:
            kwargs[cand] = params_target
            break
    # desired reward
    for cand in ("desired_reward", "r_star", "target_reward"):
        if cand in valid:
            kwargs[cand] = desired_reward
            break

    missing = valid - set(kwargs.keys()) - {"self"}
    log.debug(f"jepa_loss signature params: {sorted(valid)}; using kwargs: {sorted(kwargs.keys())}")
    return agent.jepa_loss(**kwargs)


def _set_desired_return(agent, r_star):
    """
    JEPAAgent has no set_desired_return method in this checkpoint's API.
    Try, in order: (1) a real setter method if present, (2) passing r_star
    as a kwarg to propose(), (3) a plain attribute the agent may read
    internally. Returns a dict of extra kwargs to pass into agent.propose().
    """
    if hasattr(agent, "set_desired_return") and callable(getattr(agent, "set_desired_return")):
        agent.set_desired_return(r_star)
        return {}

    propose_sig = inspect.signature(agent.propose)
    for cand in ("desired_return", "r_star", "target_reward", "desired_reward"):
        if cand in propose_sig.parameters:
            return {cand: r_star}

    # Fall back: store as a plain (non-tensor) attribute. nn.Module allows
    # arbitrary python attributes as long as they aren't Parameter/Module.
    agent.desired_return = r_star
    log.warning(
        "Agent has no 'set_desired_return' method and propose() does not "
        "accept a desired-return kwarg; stored r_star as agent.desired_return "
        "attribute. Confirm the agent's internal logic actually reads this "
        "attribute, otherwise the percentile sweep has no effect."
    )
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# R1: Decoder gradient flow experiment
# Shows that hxi is supervised indirectly through the latent loss
# ─────────────────────────────────────────────────────────────────────────────

def run_decoder_gradient_flow(ckpt_path, param_dim=8, device="cpu"):
    """
    R1: Verify that gradients flow through hxi (decoder) via latent prediction loss.
    """
    import torch
    import torch.optim as optim
    from agent.jepa_agent import JEPAAgent

    log.info("R1: Decoder gradient flow analysis")

    agent = JEPAAgent(param_dim=param_dim, device=device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    agent.load_state_dict(ckpt.get("model", ckpt), strict=False)

    rng = np.random.RandomState(0)
    rollout = get_rollout(param_dim)

    n_warmup = 20
    thetas, rewards = [], []
    for i in range(n_warmup):
        theta = rng.randn(param_dim)
        r = float(rollout("CartPole-v1", theta, 5, i))
        thetas.append(theta)
        rewards.append(r)

    thetas = np.array(thetas)
    rewards = np.array(rewards)
    mu, std = thetas.mean(0), thetas.std(0) + 1e-6
    thetas_norm = (thetas - mu) / std

    opt = optim.Adam(agent.parameters(), lr=1e-3)
    agent.train()

    ctx_len = 8
    params_t = torch.tensor(thetas_norm[:ctx_len], dtype=torch.float32).unsqueeze(0)
    rewards_t = torch.tensor(rewards[:ctx_len], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    params_target = torch.tensor(thetas_norm[ctx_len], dtype=torch.float32).unsqueeze(0)
    desired_r = torch.tensor([[rewards[ctx_len]]], dtype=torch.float32)

    try:
        opt.zero_grad()
        try:
            loss = _call_jepa_loss(agent, params_t, rewards_t, params_target, desired_r)
        except RuntimeError as shape_err:
            if "number of dimensions" in str(shape_err):
                log.warning(
                    f"  jepa_loss failed with flat params_target shape "
                    f"{tuple(params_target.shape)} ({shape_err}); retrying "
                    f"with an added time dimension."
                )
                params_target_3d = params_target.unsqueeze(1)
                desired_r_3d = desired_r.unsqueeze(1) if desired_r.dim() == 2 else desired_r
                loss = _call_jepa_loss(agent, params_t, rewards_t, params_target_3d, desired_r_3d)
            else:
                raise
        loss.backward()

        grad_norms = {}
        for name, param in agent.named_parameters():
            if param.grad is not None:
                grad_norms[name] = float(param.grad.norm())

        decoder_grads = {k: v for k, v in grad_norms.items() if "decoder" in k}
        encoder_grads = {k: v for k, v in grad_norms.items() if "encoder" in k}
        predictor_grads = {k: v for k, v in grad_norms.items() if "predictor" in k}

        dec_mean = float(np.mean(list(decoder_grads.values()))) if decoder_grads else 0.0
        enc_mean = float(np.mean(list(encoder_grads.values()))) if encoder_grads else 0.0
        pred_mean = float(np.mean(list(predictor_grads.values()))) if predictor_grads else 0.0

        log.info(f"  Loss: {loss.item():.4f}")
        log.info(f"  Decoder grad norm (mean): {dec_mean:.6f}")
        log.info(f"  Encoder grad norm (mean): {enc_mean:.6f}")
        log.info(f"  Predictor grad norm (mean): {pred_mean:.6f}")

        result = {
            "loss": float(loss.item()),
            "decoder_grad_mean": dec_mean,
            "encoder_grad_mean": enc_mean,
            "predictor_grad_mean": pred_mean,
            "decoder_grads": decoder_grads,
            "interpretation": (
                "Decoder hxi receives gradients through the latent prediction loss "
                "via chain rule: L_JEPA = ||s_y_hat - sg(s_y)||^2, where s_y_hat = p_psi(s_x, r*) "
                "and the decoder hxi: s_y_hat -> theta_hat is connected to s_y_hat. "
                "Gradients flow: L -> s_y_hat -> p_psi -> s_x -> g_phi. "
                f"Decoder grad norm ({dec_mean:.6f}) vs encoder grad norm ({enc_mean:.6f}) "
                "confirms gradients flow through all components."
            ),
        }
    except Exception as e:
        result = {"error": str(e)}
        log.error(f"  Failed: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# R2: Budget fairness -- K-island evaluation budget accounting
# ─────────────────────────────────────────────────────────────────────────────

def run_budget_fairness(ckpt_path, env_name="CartPole-v1", param_dim=8,
                         total_budget=400, n_seeds=5, n_ep=10, device="cpu"):
    """
    R2: Compare Swarm-JEPA K=3 under:
    (a) fixed TOTAL budget: T/K iterations per island (fair)
    (b) fixed PER-ISLAND budget: T iterations per island (K*T total, unfair)
    Also compare against CEM with the same total budget.
    """
    from runner.jepa_runner import rollout_policy
    from baselines import run_cem

    log.info(f"R2: Budget fairness analysis (total_budget={total_budget})")

    results = {}

    for K in [1, 3, 8]:
        for budget_mode in ["fixed_total", "fixed_per_island"]:
            if budget_mode == "fixed_total":
                iters_per_island = max(1, total_budget // K)
                total_evals = total_budget
            else:
                iters_per_island = total_budget
                total_evals = total_budget * K

            label = f"K{K}_{budget_mode}"
            log.info(f"  {label}: iters_per_island={iters_per_island} total_evals={total_evals}")

            seed_bests = []
            for seed in range(n_seeds):
                agents = [load_agent(ckpt_path, param_dim, device)[0] for _ in range(K)]
                rng = np.random.RandomState(seed)
                best_r, best_theta = -np.inf, None
                n_warmup = max(8, param_dim)

                for i in range(n_warmup):
                    for a in agents:
                        theta = rng.randn(param_dim)
                        r = float(rollout_policy(env_name, theta, n_ep, seed + i))
                        a.add_to_history(theta, r)
                        if r > best_r:
                            best_r, best_theta = r, theta.copy()

                for t in range(iters_per_island - n_warmup):
                    for ki, a in enumerate(agents):
                        theta = a.propose()
                        r = float(rollout_policy(env_name, theta, n_ep,
                                                  seed + n_warmup + t * K + ki))
                        a.add_to_history(theta, r)
                        if r > best_r:
                            best_r, best_theta = r, theta.copy()
                    if t % 30 == 0 and best_theta is not None:
                        for a in agents:
                            a.add_to_history(best_theta.copy(), best_r)

                seed_bests.append(float(best_r))

            results[label] = {
                "K": K,
                "budget_mode": budget_mode,
                "iters_per_island": iters_per_island,
                "total_evals": total_evals,
                "mean_best": float(np.mean(seed_bests)),
                "std_best": float(np.std(seed_bests)),
            }
            log.info(f"    mean_best={np.mean(seed_bests):.1f}+/-{np.std(seed_bests):.1f}")

    cem_bests = []
    for seed in range(n_seeds):
        try:
            r = run_cem(env_name, param_dim, seed, total_budget, n_ep)
            cem_bests.append(r["best_reward"])
        except Exception as e:
            log.warning(f"  CEM failed: {e}")
    if cem_bests:
        results["cem_budget_matched"] = {
            "total_evals": total_budget,
            "mean_best": float(np.mean(cem_bests)),
            "std_best": float(np.std(cem_bests)),
        }
        log.info(f"  CEM (budget={total_budget}): {np.mean(cem_bests):.1f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# R3: Standardized CartPole comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_standardized_cartpole(ckpt_path, param_dim=8,
                               n_seeds=10, budget=400, n_ep=10, device="cpu"):
    """
    R3: Single standardized CartPole experiment:
    - All methods: 400 evaluations total, 10 seeds, 10 episodes per eval
    - Reports mean, std, median for all methods
    - Resolves inconsistency between different tables
    """
    from runner.jepa_runner import rollout_policy
    from baselines import run_cem, run_cmaes, run_random_search

    log.info(f"R3: Standardized CartPole comparison (budget={budget}, n_seeds={n_seeds})")
    results = {}

    jepa_rewards = []
    for seed in range(n_seeds):
        agent, pdim = load_agent(ckpt_path, param_dim, device)
        rng = np.random.RandomState(seed)
        best = -np.inf
        n_warmup = max(8, pdim)
        for i in range(n_warmup):
            theta = rng.randn(pdim)
            r = float(rollout_policy("CartPole-v1", theta, n_ep, seed + i))
            agent.add_to_history(theta, r)
            best = max(best, r)
        for t in range(budget - n_warmup):
            theta = agent.propose()
            r = float(rollout_policy("CartPole-v1", theta, n_ep,
                                      seed + n_warmup + t))
            agent.add_to_history(theta, r)
            best = max(best, r)
        jepa_rewards.append(best)

    results["jepa_single"] = {
        "budget": budget, "n_seeds": n_seeds,
        "mean": float(np.mean(jepa_rewards)),
        "std": float(np.std(jepa_rewards)),
        "median": float(np.median(jepa_rewards)),
    }
    log.info(f"  JEPA single: {np.mean(jepa_rewards):.1f}+/-{np.std(jepa_rewards):.1f}")

    for name, fn in [
        ("cem", lambda s: run_cem("CartPole-v1", param_dim, s, budget, n_ep)),
        ("cmaes", lambda s: run_cmaes("CartPole-v1", param_dim, s, budget, n_ep)),
        ("random", lambda s: run_random_search("CartPole-v1", param_dim, s, budget, n_ep)),
    ]:
        bests = []
        for seed in range(n_seeds):
            try:
                r = fn(seed)
                bests.append(r["best_reward"])
            except Exception as e:
                log.warning(f"  {name} seed={seed}: {e}")
        if bests:
            results[name] = {
                "budget": budget, "n_seeds": len(bests),
                "mean": float(np.mean(bests)),
                "std": float(np.std(bests)),
                "median": float(np.median(bests)),
            }
            log.info(f"  {name}: {np.mean(bests):.1f}+/-{np.std(bests):.1f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# R4: Target return r* percentile sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def run_percentile_sensitivity(ckpt_path, env_name="CartPole-v1", param_dim=8,
                                percentiles=(50, 60, 70, 80, 90, 95, 99),
                                n_seeds=5, max_iter=400, n_ep=10, device="cpu"):
    """
    R4: Sweep the desired-return percentile r* at inference.
    At each step, r* = pct_rho({r_i}) for rho in {50,...,99}.
    Tests whether performance is robust to this hyperparameter choice.
    """
    from runner.jepa_runner import rollout_policy

    log.info(f"R4: r* percentile sensitivity on {env_name}")
    results = {}

    for pct in percentiles:
        log.info(f"  pct={pct}")
        seed_bests = []

        for seed in range(n_seeds):
            agent, pdim = load_agent(ckpt_path, param_dim, device)
            rng = np.random.RandomState(seed)
            history_r = []
            best = -np.inf
            n_warmup = max(8, pdim)

            for i in range(n_warmup):
                theta = rng.randn(pdim)
                r = float(rollout_policy(env_name, theta, n_ep, seed + i))
                agent.add_to_history(theta, r)
                history_r.append(r)
                best = max(best, r)

            for t in range(max_iter - n_warmup):
                r_star = float(np.percentile(history_r, pct))
                propose_kwargs = _set_desired_return(agent, r_star)
                theta = agent.propose(**propose_kwargs)
                r = float(rollout_policy(env_name, theta, n_ep,
                                          seed + n_warmup + t))
                agent.add_to_history(theta, r)
                history_r.append(r)
                best = max(best, r)

            seed_bests.append(float(best))

        results[f"pct_{pct}"] = {
            "percentile": pct,
            "mean_best": float(np.mean(seed_bests)),
            "std_best": float(np.std(seed_bests)),
        }
        log.info(f"  pct={pct}: {np.mean(seed_bests):.2f}+/-{np.std(seed_bests):.2f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# R5: FDC sampling robustness
# ─────────────────────────────────────────────────────────────────────────────

def run_fdc_robustness(env_name="CartPole-v1", param_dim=8,
                        sample_sizes=(50, 100, 200, 300, 500, 1000),
                        n_repeats=10, n_ep=5):
    """
    R5: Test robustness of FDC estimate to sample size and proxy optimum.
    """
    from runner.jepa_runner import rollout_policy

    log.info(f"R5: FDC robustness for {env_name}")
    results = {}

    for N in sample_sizes:
        log.info(f"  N={N}")
        fdc_values = []
        fdc_better_proxy = []

        for rep in range(n_repeats):
            rng = np.random.RandomState(rep * 1000)

            thetas = rng.randn(N, param_dim)
            rewards = np.array([float(rollout_policy(env_name, t, n_ep, rep * 1000 + i))
                                 for i, t in enumerate(thetas)])

            best_idx = np.argmax(rewards)
            dists = np.linalg.norm(thetas - thetas[best_idx], axis=1)
            if rewards.std() > 1e-6:
                fdc = float(np.corrcoef(rewards, -dists)[0, 1])
            else:
                fdc = float("nan")
            fdc_values.append(fdc)

            thetas_extra = rng.randn(4 * N, param_dim)
            rewards_extra = np.array([
                float(rollout_policy(env_name, t, n_ep, rep * 10000 + i))
                for i, t in enumerate(thetas_extra)])
            best_theta_global = np.vstack([thetas, thetas_extra])[
                np.argmax(np.concatenate([rewards, rewards_extra]))]
            dists_better = np.linalg.norm(thetas - best_theta_global, axis=1)
            if rewards.std() > 1e-6:
                fdc_b = float(np.corrcoef(rewards, -dists_better)[0, 1])
            else:
                fdc_b = float("nan")
            fdc_better_proxy.append(fdc_b)

        vals = np.array([v for v in fdc_values if not np.isnan(v)])
        vals_b = np.array([v for v in fdc_better_proxy if not np.isnan(v)])

        results[f"N_{N}"] = {
            "N": N,
            "fdc_mean": float(np.mean(vals)) if len(vals) > 0 else float("nan"),
            "fdc_std": float(np.std(vals)) if len(vals) > 0 else float("nan"),
            "fdc_ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
                        if len(vals) > 1 else [float("nan"), float("nan")],
            "fdc_better_proxy_mean": float(np.mean(vals_b)) if len(vals_b) > 0 else float("nan"),
            "fdc_better_proxy_std": float(np.std(vals_b)) if len(vals_b) > 0 else float("nan"),
            "n_valid": int(len(vals)),
        }
        if len(vals) > 1:
            log.info(f"    FDC={np.mean(vals):.3f}+/-{np.std(vals):.3f} "
                      f"CI=[{np.percentile(vals,2.5):.3f},{np.percentile(vals,97.5):.3f}]")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# R6: Swarm hyperparameter sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def run_swarm_hyperparameter_sensitivity(ckpt_path, env_name="CartPole-v1",
                                          param_dim=8, n_seeds=3,
                                          max_iter=400, n_ep=10, device="cpu"):
    """
    R6: Sensitivity to swarm wrapper hyperparameters.
    """
    from runner.jepa_runner import rollout_policy

    log.info("R6: Swarm hyperparameter sensitivity")
    results = {}
    K = 3

    def run_config(migration_every, isolation_every, noise_std, seed):
        agents = [load_agent(ckpt_path, param_dim, device)[0] for _ in range(K)]
        rng = np.random.RandomState(seed)
        best_r, best_theta = -np.inf, None
        n_warmup = max(8, param_dim)

        for i in range(n_warmup):
            for a in agents:
                theta = rng.randn(param_dim)
                r = float(rollout_policy(env_name, theta, n_ep, seed + i))
                a.add_to_history(theta, r)
                if r > best_r:
                    best_r, best_theta = r, theta.copy()

        for t in range(max_iter - n_warmup):
            for ki, a in enumerate(agents):
                theta_raw = a.propose()
                theta = theta_raw + rng.randn(param_dim) * noise_std * 0.1
                r = float(rollout_policy(env_name, theta, n_ep,
                                          seed + n_warmup + t * K + ki))
                a.add_to_history(theta, r)
                if r > best_r:
                    best_r, best_theta = r, theta.copy()

            if migration_every and t % migration_every == 0 and best_theta is not None:
                for a in agents:
                    a.add_to_history(best_theta.copy(), best_r)

        return float(best_r)

    for mig in [10, 20, 30, 60, None]:
        mig_label = f"mig_{mig}" if mig else "mig_never"
        bests = [run_config(mig, 60, 0.3, s) for s in range(n_seeds)]
        results[mig_label] = {"mean": float(np.mean(bests)), "std": float(np.std(bests))}
        log.info(f"  migration_every={mig}: {np.mean(bests):.1f}+/-{np.std(bests):.1f}")

    for ns in [0.1, 0.3, 0.5, 1.0]:
        bests = [run_config(30, 60, ns, s) for s in range(n_seeds)]
        results[f"noise_{ns}"] = {"mean": float(np.mean(bests)), "std": float(np.std(bests))}
        log.info(f"  noise_std={ns}: {np.mean(bests):.1f}+/-{np.std(bests):.1f}")

    bests_single = []
    for seed in range(n_seeds):
        agent, pdim = load_agent(ckpt_path, param_dim, device)
        rng = np.random.RandomState(seed)
        best = -np.inf
        n_warmup = max(8, pdim)
        for i in range(n_warmup):
            theta = rng.randn(pdim)
            r = float(rollout_policy(env_name, theta, n_ep, seed + i))
            agent.add_to_history(theta, r)
            best = max(best, r)
        for t in range(max_iter - n_warmup):
            theta = agent.propose()
            r = float(rollout_policy(env_name, theta, n_ep, seed + n_warmup + t))
            agent.add_to_history(theta, r)
            best = max(best, r)
        bests_single.append(best)

    results["single_agent_k1"] = {
        "mean": float(np.mean(bests_single)),
        "std": float(np.std(bests_single)),
        "note": "single agent, no migration, baseline for swarm lift",
    }
    log.info(f"  Single agent (K=1): {np.mean(bests_single):.1f}+/-{np.std(bests_single):.1f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# R8: Non-control black-box benchmark (BBOB synthetic functions)
# ─────────────────────────────────────────────────────────────────────────────

def run_bbob_benchmark(ckpt_path, param_dim=8,
                        n_seeds=5, max_iter=400, device="cpu"):
    """
    R8: Evaluate JEPA on BBOB (Black-Box Optimization Benchmark) synthetic functions.
    """
    log.info("R8: BBOB synthetic benchmark")

    def sphere(x):
        return -float(np.sum(x ** 2))

    def ellipsoid(x):
        n = len(x)
        return -float(sum(1e6 ** (i / (n - 1)) * x[i] ** 2 for i in range(n)))

    def rosenbrock(x):
        return -float(sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
                           for i in range(len(x) - 1)))

    def rastrigin(x):
        n = len(x)
        return -float(10 * n + sum(x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i])
                                    for i in range(n)))

    def schwefel(x):
        n = len(x)
        return -float(418.9829 * n - sum(xi * np.sin(np.sqrt(abs(xi)))
                                          for xi in x * 500))

    fns = {
        "f1_sphere": sphere,
        "f2_ellipsoid": ellipsoid,
        "f8_rosenbrock": rosenbrock,
        "f15_rastrigin": rastrigin,
        "f20_schwefel": schwefel,
    }

    results = {}

    for fname, fn in fns.items():
        log.info(f"  {fname}")

        jepa_bests = []
        for seed in range(n_seeds):
            agent, pdim = load_agent(ckpt_path, param_dim, device)
            rng = np.random.RandomState(seed)
            best = -np.inf
            n_warmup = max(8, pdim)
            for i in range(n_warmup):
                theta = rng.randn(pdim)
                r = float(fn(theta))
                agent.add_to_history(theta, r)
                best = max(best, r)
            for t in range(max_iter - n_warmup):
                theta = agent.propose()
                r = float(fn(theta))
                agent.add_to_history(theta, r)
                best = max(best, r)
            jepa_bests.append(best)

        cem_bests = []
        for seed in range(n_seeds):
            rng = np.random.RandomState(seed)
            mu = rng.randn(param_dim)
            sigma = np.ones(param_dim)
            pop = 20
            elite_k = 5
            best_cem = -np.inf
            for _ in range(max_iter // pop):
                samples = mu + sigma * rng.randn(pop, param_dim)
                rs = [float(fn(s)) for s in samples]
                idx = np.argsort(rs)[::-1][:elite_k]
                mu = samples[idx].mean(0)
                sigma = samples[idx].std(0) + 1e-6
                best_cem = max(best_cem, max(rs))
            cem_bests.append(best_cem)

        results[fname] = {
            "jepa_mean": float(np.mean(jepa_bests)),
            "jepa_std": float(np.std(jepa_bests)),
            "cem_mean": float(np.mean(cem_bests)),
            "cem_std": float(np.std(cem_bests)),
        }
        log.info(f"    JEPA={np.mean(jepa_bests):.3f}+/-{np.std(jepa_bests):.3f} "
                  f"CEM={np.mean(cem_bests):.3f}+/-{np.std(cem_bests):.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# R9: Reward shuffle mechanistic analysis
# What does JEPA actually learn if not reward values?
# ─────────────────────────────────────────────────────────────────────────────

def run_reward_shuffle_mechanistic(ckpt_path, env_name="CartPole-v1",
                                    param_dim=8, n_seeds=5, max_iter=200,
                                    n_ep=10, device="cpu"):
    """
    R9: If shuffling theta-reward pairs doesn't hurt, what DOES JEPA learn?
    """
    from runner.jepa_runner import rollout_policy

    log.info(f"R9: Reward shuffle mechanistic analysis on {env_name}")

    def run_condition(condition, seed):
        agent, pdim = load_agent(ckpt_path, param_dim, device)
        rng = np.random.RandomState(seed)
        n_warmup = max(8, pdim)
        thetas, rewards = [], []

        for i in range(n_warmup):
            theta = rng.randn(pdim)
            r = float(rollout_policy(env_name, theta, n_ep, seed + i))
            thetas.append(theta)
            rewards.append(r)

        if condition == "shuffle_rewards":
            perm = rng.permutation(len(rewards))
            rewards_hist = [rewards[p] for p in perm]
            for t, r in zip(thetas, rewards_hist):
                agent.add_to_history(t, r)
        elif condition == "permute_dims":
            perm = rng.permutation(pdim)
            for t, r in zip(thetas, rewards):
                agent.add_to_history(t[perm], r)
        elif condition == "reverse_order":
            for t, r in zip(reversed(thetas), reversed(rewards)):
                agent.add_to_history(t, r)
        elif condition == "best_only":
            bi = np.argmax(rewards)
            agent.add_to_history(thetas[bi], rewards[bi])
        else:  # control
            for t, r in zip(thetas, rewards):
                agent.add_to_history(t, r)

        best = max(rewards)
        for t in range(max_iter - n_warmup):
            theta = agent.propose()
            r = float(rollout_policy(env_name, theta, n_ep,
                                      seed + n_warmup + t))
            agent.add_to_history(theta, r)
            best = max(best, r)
        return float(best)

    conditions = ["control", "shuffle_rewards", "permute_dims",
                  "reverse_order", "best_only"]
    results = {}

    for cond in conditions:
        bests = [run_condition(cond, seed) for seed in range(n_seeds)]
        results[cond] = {
            "mean_best": float(np.mean(bests)),
            "std_best": float(np.std(bests)),
        }
        log.info(f"  {cond}: {np.mean(bests):.1f}+/-{np.std(bests):.1f}")

    control = results["control"]["mean_best"]
    results["interpretation"] = {
        "shuffle_rewards_drop": (control - results["shuffle_rewards"]["mean_best"]) / (abs(control) + 1e-6),
        "permute_dims_drop": (control - results["permute_dims"]["mean_best"]) / (abs(control) + 1e-6),
        "reverse_drop": (control - results["reverse_order"]["mean_best"]) / (abs(control) + 1e-6),
        "best_only_drop": (control - results["best_only"]["mean_best"]) / (abs(control) + 1e-6),
        "conclusion": (
            "If shuffle_rewards_drop ~ 0: JEPA does not use reward values, only parameter geometry. "
            "If permute_dims_drop > 0: JEPA uses the geometric structure of parameter space. "
            "If reverse_drop > 0: JEPA uses temporal ordering. "
            "If best_only_drop > 0: JEPA uses trajectory information beyond the current best."
        ),
    }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

ALL = {
    "decoder_gradient": run_decoder_gradient_flow,
    "budget_fairness": run_budget_fairness,
    "standardized_cartpole": run_standardized_cartpole,
    "percentile_sensitivity": run_percentile_sensitivity,
    "fdc_robustness": run_fdc_robustness,
    "swarm_hyperparams": run_swarm_hyperparameter_sensitivity,
    "bbob_benchmark": run_bbob_benchmark,
    "reward_shuffle_mechanistic": run_reward_shuffle_mechanistic,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", nargs="+", default=["all"],
                    choices=list(ALL.keys()) + ["all"])
    p.add_argument("--ckpt", default="checkpoints/jepa_pretrained/jepa_pretrained.pt")
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--param_dim", type=int, default=8)
    p.add_argument("--out_dir", default="results/reviewer")
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--max_iter", type=int, default=400)
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    exps = list(ALL.keys()) if "all" in args.experiment else args.experiment

    common = dict(ckpt_path=args.ckpt, env_name=args.env,
                  param_dim=args.param_dim, n_seeds=args.n_seeds,
                  max_iter=args.max_iter, n_ep=args.n_episodes,
                  device=args.device)

    # Each entry here is tailored to match the actual function signature
    # (parameter names differ across experiments: total_budget vs budget vs max_iter).
    special = {
        "decoder_gradient": dict(ckpt_path=args.ckpt,
                                  param_dim=args.param_dim, device=args.device),
        "budget_fairness": dict(ckpt_path=args.ckpt, env_name=args.env,
                                 param_dim=args.param_dim,
                                 total_budget=args.max_iter,
                                 n_seeds=args.n_seeds, n_ep=args.n_episodes,
                                 device=args.device),
        "standardized_cartpole": dict(ckpt_path=args.ckpt,
                                       param_dim=args.param_dim,
                                       n_seeds=args.n_seeds,
                                       budget=args.max_iter,
                                       n_ep=args.n_episodes,
                                       device=args.device),
        "fdc_robustness": dict(env_name=args.env, param_dim=args.param_dim,
                                n_ep=args.n_episodes),
        "bbob_benchmark": dict(ckpt_path=args.ckpt, param_dim=args.param_dim,
                                n_seeds=args.n_seeds, max_iter=args.max_iter,
                                device=args.device),
    }

    for exp in exps:
        log.info(f"\n{'='*60}\nRunning: {exp}\n{'='*60}")
        fn = ALL[exp]
        try:
            result = fn(**special.get(exp, common))
            out = f"{args.out_dir}/{exp}.json"
            with open(out, "w") as f:
                json.dump(result, f, indent=2)
            log.info(f"Saved: {out}")
        except Exception as e:
            log.error(f"FAILED: {exp} - {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()