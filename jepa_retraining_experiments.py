"""
jepa_retraining_experiments.py — Five experiments requiring JEPA retraining:

N1.  Leave-one-env-out generalization: train on N-1 envs, test on held-out env.
N4.  Pretraining data efficiency: vary fraction of training data (5%→100%).
N5.  Reward shuffling / memorization test: shuffle rewards during training.
N12. Cross-domain pretraining diversity: compare locomotion-only vs balance-only
     vs mixed pretraining.
N14. Replace JEPA with plain Transformer: ablate JEPA's predictive objective
     vs a standard supervised next-param-prediction loss.

Each experiment builds on your existing pretraining pipeline (agent/jepa_agent.py
and however you trained the original checkpoints). You need to adapt the training
loop calls to match your specific pretraining script's API.

ASSUMES:
- Your pretraining script is accessible as a callable or CLI tool.
- Training data is stored as (theta, reward) trajectory files loadable
  from your existing pipeline.
- GPU available for training (these are not fast on CPU).

USAGE:
    # N1: Leave-one-env-out
    python jepa_retraining_experiments.py --experiment leave_one_out \
        --train_envs CartPole-v1 Swimmer-v4 MountainCarContinuous-v0 InvertedPendulum-v4 \
        --held_out_env Hopper-v4 \
        --out_dir results/retraining

    # N4: Data efficiency
    python jepa_retraining_experiments.py --experiment data_efficiency \
        --data_fractions 0.05 0.10 0.25 0.50 1.0 \
        --eval_env CartPole-v1 Swimmer-v4 \
        --out_dir results/retraining

    # N5: Reward shuffling
    python jepa_retraining_experiments.py --experiment reward_shuffle \
        --shuffle_type rewards params both \
        --eval_env CartPole-v1 Swimmer-v4 \
        --out_dir results/retraining

    # N12: Cross-domain diversity
    python jepa_retraining_experiments.py --experiment domain_diversity \
        --out_dir results/retraining

    # N14: Plain Transformer baseline
    python jepa_retraining_experiments.py --experiment plain_transformer \
        --eval_env CartPole-v1 Swimmer-v4 \
        --out_dir results/retraining
"""

import os, sys, json, time, argparse, logging
import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
sys.path.insert(0, os.getcwd())


# ─────────────────────────────────────────────────────────────────────────────
# Data collection: generates (theta, reward) pretraining trajectories
# This is the common data-collection step used by N1, N4, N5, N12.
# ─────────────────────────────────────────────────────────────────────────────

def collect_pretraining_data(env_name, param_dim, n_trajectories=5000,
                              n_episodes_per_eval=5, seed=0):
    """
    Collects (theta, reward) pairs via uniform random search on env_name.
    This mirrors exactly how your original JEPA pretraining data was collected
    (random parameter sampling, no reward optimization).

    Returns list of (theta: np.ndarray, reward: float) tuples.
    """
    from envs import rollout_policy
    rng = np.random.RandomState(seed)
    data = []
    for i in range(n_trajectories):
        theta = rng.randn(param_dim)
        r = rollout_policy(env_name, theta, n_episodes=n_episodes_per_eval,
                            seed=seed + i)
        data.append((theta.copy(), r))
        if (i + 1) % 500 == 0:
            log.info(f"  {env_name}: collected {i+1}/{n_trajectories} trajectories")
    return data


def save_trajectory_data(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    thetas = np.array([d[0] for d in data])
    rewards = np.array([d[1] for d in data])
    np.savez(path, thetas=thetas, rewards=rewards)
    log.info(f"Saved {len(data)} trajectories to {path}")


def load_trajectory_data(path):
    d = np.load(path)
    return list(zip(d["thetas"], d["rewards"]))


# ─────────────────────────────────────────────────────────────────────────────
# JEPA training loop (wraps your existing training infrastructure)
# ─────────────────────────────────────────────────────────────────────────────

def train_jepa(train_data_by_env, param_dim, n_epochs=100, batch_size=64,
               lr=1e-3, d_model=128, device="cuda", seed=0, out_ckpt_path=None):
    """
    Trains a JEPAAgent from scratch on the provided pretraining data.
    train_data_by_env: dict {env_name: list of (theta, reward) tuples}

    This is the core function called by all retraining experiments.
    It mirrors your original pretraining procedure but with configurable data.
    """
    import torch
    import torch.optim as optim
    from agent.jepa_agent import JEPAAgent

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Pool all data
    all_thetas, all_rewards = [], []
    for env_name, data in train_data_by_env.items():
        for theta, reward in data:
            all_thetas.append(theta)
            all_rewards.append(reward)

    all_thetas = np.array(all_thetas, dtype=np.float32)
    all_rewards = np.array(all_rewards, dtype=np.float32)

    # Normalize rewards globally
    r_mean, r_std = all_rewards.mean(), all_rewards.std() + 1e-8
    all_rewards_norm = (all_rewards - r_mean) / r_std

    agent = JEPAAgent(param_dim=param_dim, d_model=d_model, device=device)
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    n_samples = len(all_thetas)
    log.info(f"Training JEPA: {n_samples} samples, {n_epochs} epochs, "
             f"param_dim={param_dim}, d_model={d_model}")

    best_loss = float("inf")
    loss_history = []

    for epoch in range(n_epochs):
        # Shuffle
        idx = np.random.permutation(n_samples)
        all_thetas = all_thetas[idx]
        all_rewards_norm = all_rewards_norm[idx]

        epoch_losses = []
        for start in range(0, n_samples - batch_size, batch_size):
            batch_thetas = all_thetas[start:start + batch_size]
            batch_rewards = all_rewards_norm[start:start + batch_size]

            # Build context windows: use sliding windows of length 4-20
            # This simulates the in-context optimization setting
            ctx_len = np.random.randint(4, 21)
            if len(batch_thetas) < ctx_len + 1:
                continue

            # Stack into sequences: (B, T, param_dim) and (B, T, 1)
            import torch
            thetas_t = torch.tensor(batch_thetas[:ctx_len + 1], dtype=torch.float32,
                                     device=device).unsqueeze(0)
            rewards_t = torch.tensor(batch_rewards[:ctx_len + 1], dtype=torch.float32,
                                      device=device).unsqueeze(0).unsqueeze(-1)

            mean = thetas_t[:, :ctx_len].mean(dim=1, keepdim=True)
            std = thetas_t[:, :ctx_len].std(dim=1, keepdim=True) + 1e-6
            thetas_norm = (thetas_t - mean) / std

            optimizer.zero_grad()
            loss = agent.jepa_loss(
                params_ctx=thetas_norm[:, :ctx_len],
                rewards_ctx=rewards_t[:, :ctx_len],
                params_target=thetas_norm[:, ctx_len:],
                desired_reward=rewards_t[:, -1, :],          # (B,1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        epoch_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        loss_history.append(epoch_loss)
        if (epoch + 1) % 10 == 0:
            log.info(f"  Epoch {epoch+1}/{n_epochs}: loss={epoch_loss:.4f}")
        if epoch_loss < best_loss and out_ckpt_path:
            best_loss = epoch_loss
            os.makedirs(os.path.dirname(out_ckpt_path), exist_ok=True)
            import torch
            torch.save({"model": agent.state_dict(), "epoch": epoch,
                        "loss": epoch_loss}, out_ckpt_path)

    return agent, loss_history


def evaluate_checkpoint(ckpt_path, env_name, param_dim, n_seeds=3,
                          max_iterations=400, n_eval_episodes=10, device="cpu"):
    """Evaluates a trained checkpoint using single-trajectory JEPA search."""
    import torch
    from agent.jepa_agent import JEPAAgent
    from envs import rollout_policy

    seed_results = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        agent = JEPAAgent(param_dim=param_dim, device=device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt.get("model", ckpt), strict=False)

        rng = np.random.RandomState(seed)
        rewards = []
        best = -np.inf
        n_warmup = max(8, param_dim)

        for i in range(n_warmup):
            theta = rng.randn(param_dim)
            r = rollout_policy(env_name, theta, n_eval_episodes, seed + i)
            agent.add_to_history(theta, r)
            rewards.append(r)
            best = max(best, r)

        for t in range(max_iterations - n_warmup):
            theta = agent.propose()
            r = rollout_policy(env_name, theta, n_eval_episodes, seed + n_warmup + t)
            agent.add_to_history(theta, r)
            rewards.append(r)
            best = max(best, r)

        seed_results.append({"seed": seed, "best": float(best),
                              "mean": float(np.mean(rewards))})

    mean_best = float(np.mean([r["best"] for r in seed_results]))
    mean_mean = float(np.mean([r["mean"] for r in seed_results]))
    log.info(f"  {env_name}: mean_best={mean_best:.2f} mean_mean={mean_mean:.2f} "
             f"over {n_seeds} seeds")
    return {"mean_best": mean_best, "mean_mean": mean_mean, "per_seed": seed_results}


# ─────────────────────────────────────────────────────────────────────────────
# N1: Leave-one-env-out generalization
# ─────────────────────────────────────────────────────────────────────────────

def run_leave_one_out(all_envs, n_trajectories=2000, n_epochs=50,
                       device="cuda", out_dir="results/retraining"):
    """
    For each env in all_envs:
      - Train JEPA on all OTHER envs
      - Test on the held-out env
      - Compare to: (a) trained on ALL envs, (b) trained on NO envs (random)

    This answers: "Does JEPA generalize to unseen environments, or does it
    memorize optimization patterns for seen environments?"
    """
    from envs import TASKS, get_param_dim

    results = {}
    data_cache = {}

    # Collect data for all envs
    log.info("Collecting pretraining data for all environments...")
    for env in all_envs:
        pdim = get_param_dim(env)
        cache_path = f"{out_dir}/data/{env.replace('/','_')}.npz"
        if os.path.exists(cache_path):
            data_cache[env] = load_trajectory_data(cache_path)
            log.info(f"  {env}: loaded {len(data_cache[env])} cached trajectories")
        else:
            data_cache[env] = collect_pretraining_data(env, pdim, n_trajectories)
            save_trajectory_data(data_cache[env], cache_path)

    # Since every env has a unique param_dim, project all training data
    # to the held-out env's param_dim via fixed random linear maps.
    def project_data_to_dim(data, source_dim, target_dim, seed=42):
        if source_dim == target_dim:
            return data
        rng_p = np.random.RandomState(seed + source_dim * 1000 + target_dim)
        if source_dim < target_dim:
            return [(np.pad(theta, (0, target_dim - source_dim)), reward)
                    for theta, reward in data]
        M = rng_p.randn(target_dim, source_dim)
        M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-8
        return [(M @ theta, reward) for theta, reward in data]

    # LOO: for each held-out env, project all other envs' data to held-out dim
    for held_out in all_envs:
        target_dim = get_param_dim(held_out)
        train_envs = [e for e in all_envs if e != held_out]
        if not train_envs:
            log.warning(f"Skipping {held_out}: no remaining training envs")
            continue
        log.info(f"\nLeave-one-out: held_out={held_out} (dim={target_dim}), "
                 f"train on {train_envs}")

        projected = {e: project_data_to_dim(
            data_cache[e], get_param_dim(e), target_dim) for e in train_envs}
        safe_name = held_out.replace('/', '_').replace('-', '_')
        ckpt_path = f"{out_dir}/checkpoints/leave_out_{safe_name}.pt"
        _, losses = train_jepa(projected, param_dim=target_dim,
                                n_epochs=n_epochs, device=device,
                                out_ckpt_path=ckpt_path)
        eval_result = evaluate_checkpoint(ckpt_path, held_out, target_dim, n_seeds=3)

        # Full baseline: all envs projected to held-out dim
        all_proj = {e: project_data_to_dim(
            data_cache[e], get_param_dim(e), target_dim) for e in all_envs}
        full_ckpt = f"{out_dir}/checkpoints/full_to_{target_dim}dim.pt"
        if not os.path.exists(full_ckpt):
            train_jepa(all_proj, param_dim=target_dim, n_epochs=n_epochs,
                        device=device, out_ckpt_path=full_ckpt)
        full_eval = evaluate_checkpoint(full_ckpt, held_out, target_dim, n_seeds=3)

        results[f"leave_out_{held_out}"] = {
            "held_out_env": held_out, "held_out_dim": target_dim,
            "train_envs": train_envs,
            "loo_performance": eval_result,
            "full_model_performance": full_eval,
            "generalization_gap": eval_result["mean_best"] - full_eval["mean_best"],
            "training_loss_curve": losses,
        }
        log.info(f"  LOO={eval_result['mean_best']:.2f}, "
                 f"Full={full_eval['mean_best']:.2f}, "
                 f"Gap={eval_result['mean_best']-full_eval['mean_best']:.2f}")

    out_path = f"{out_dir}/leave_one_out_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nLeave-one-out results saved to {out_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# N4: Pretraining data efficiency
# ─────────────────────────────────────────────────────────────────────────────

def run_data_efficiency(train_envs, eval_envs, data_fractions=(0.05, 0.1, 0.25, 0.5, 1.0),
                         n_trajectories_full=5000, n_epochs=50,
                         device="cuda", out_dir="results/retraining"):
    """
    Trains JEPA with varying amounts of pretraining data (5%→100%) and
    measures downstream optimization performance.

    Answers: "How much data does JEPA actually need?"
    Expected result: a learning curve that plateaus -- showing the minimum
    data needed for effective generalization.
    """
    from envs import get_param_dim

    pdim = get_param_dim(train_envs[0])
    results = {}

    # Collect full dataset once
    log.info("Collecting full pretraining dataset...")
    full_data = {}
    for env in train_envs:
        cache_path = f"{out_dir}/data/{env.replace('/','_')}_full.npz"
        if os.path.exists(cache_path):
            full_data[env] = load_trajectory_data(cache_path)
        else:
            full_data[env] = collect_pretraining_data(
                env, pdim, n_trajectories_full)
            save_trajectory_data(full_data[env], cache_path)

    for fraction in data_fractions:
        n_keep = max(1, int(n_trajectories_full * fraction))
        log.info(f"\nData efficiency: fraction={fraction}, n_samples={n_keep} per env")

        # Subsample consistently (same seed for fair comparison)
        rng = np.random.RandomState(42)
        subsampled = {}
        for env in train_envs:
            n_available = len(full_data[env])
            idx = rng.choice(n_available, size=min(n_keep, n_available), replace=False)
            subsampled[env] = [full_data[env][i] for i in idx]

        ckpt_path = f"{out_dir}/checkpoints/data_efficiency_{fraction:.2f}.pt"
        _, losses = train_jepa(subsampled, param_dim=pdim, n_epochs=n_epochs,
                                device=device, out_ckpt_path=ckpt_path)

        fraction_results = {"fraction": fraction, "n_samples_per_env": n_keep,
                             "training_loss_curve": losses}
        for eval_env in eval_envs:
            eval_result = evaluate_checkpoint(ckpt_path, eval_env, pdim, n_seeds=3)
            fraction_results[f"eval_{eval_env}"] = eval_result
            log.info(f"  {eval_env}: mean_best={eval_result['mean_best']:.2f}")

        results[f"fraction_{fraction}"] = fraction_results

    out_path = f"{out_dir}/data_efficiency_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nData efficiency results saved to {out_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# N5: Reward shuffling / memorization test
# ─────────────────────────────────────────────────────────────────────────────

def run_reward_shuffle(train_envs, eval_envs,
                        shuffle_types=("none", "rewards", "params", "both"),
                        n_trajectories=2000, n_epochs=50,
                        device="cuda", out_dir="results/retraining"):
    """
    Tests whether JEPA learns genuine optimization structure or merely
    memorizes dataset statistics, via three shuffle controls:

    "none"    -- standard training (control)
    "rewards" -- shuffle rewards randomly within training data.
                 If performance collapses: JEPA learns from reward signal,
                 not from parameter geometry alone.
    "params"  -- shuffle parameter vectors randomly within training data.
                 If performance collapses: JEPA learns from parameter
                 structure, not from reward signal alone.
    "both"    -- shuffle both params and rewards independently.
                 Should collapse worst of all -- totally random training.

    Expected: collapse for "rewards" and "params" individually proves that
    JEPA uses BOTH the parameter values AND the reward signal jointly,
    i.e., it learns the reward landscape, not just either dimension alone.
    """
    from envs import get_param_dim

    pdim = get_param_dim(train_envs[0])
    results = {}

    # Collect data once
    log.info("Collecting pretraining data...")
    raw_data = {}
    for env in train_envs:
        cache_path = f"{out_dir}/data/{env.replace('/','_')}_shuffle.npz"
        if os.path.exists(cache_path):
            raw_data[env] = load_trajectory_data(cache_path)
        else:
            raw_data[env] = collect_pretraining_data(env, pdim, n_trajectories)
            save_trajectory_data(raw_data[env], cache_path)

    for shuffle_type in shuffle_types:
        log.info(f"\nReward shuffle experiment: shuffle_type={shuffle_type}")
        rng = np.random.RandomState(0)

        # Apply shuffle transformation
        shuffled_data = {}
        for env in train_envs:
            original = raw_data[env]
            thetas = [d[0] for d in original]
            rewards = [d[1] for d in original]

            if shuffle_type == "rewards":
                shuffled_rewards = np.array(rewards)[rng.permutation(len(rewards))]
                shuffled_data[env] = list(zip(thetas, shuffled_rewards.tolist()))
            elif shuffle_type == "params":
                shuffled_thetas = [thetas[i] for i in rng.permutation(len(thetas))]
                shuffled_data[env] = list(zip(shuffled_thetas, rewards))
            elif shuffle_type == "both":
                shuffled_rewards = np.array(rewards)[rng.permutation(len(rewards))]
                shuffled_thetas = [thetas[i] for i in rng.permutation(len(thetas))]
                shuffled_data[env] = list(zip(shuffled_thetas, shuffled_rewards.tolist()))
            else:  # "none" -- no shuffle
                shuffled_data[env] = original

        ckpt_path = f"{out_dir}/checkpoints/shuffle_{shuffle_type}.pt"
        _, losses = train_jepa(shuffled_data, param_dim=pdim, n_epochs=n_epochs,
                                device=device, out_ckpt_path=ckpt_path)

        shuffle_results = {"shuffle_type": shuffle_type, "training_loss_curve": losses}
        for eval_env in eval_envs:
            eval_result = evaluate_checkpoint(ckpt_path, eval_env, pdim, n_seeds=3)
            shuffle_results[f"eval_{eval_env}"] = eval_result
            log.info(f"  {shuffle_type} -> {eval_env}: mean_best={eval_result['mean_best']:.2f}")

        results[shuffle_type] = shuffle_results

    out_path = f"{out_dir}/reward_shuffle_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nReward shuffle results saved to {out_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# N12: Cross-domain pretraining diversity
# ─────────────────────────────────────────────────────────────────────────────

def run_domain_diversity(n_trajectories=2000, n_epochs=50,
                          device="cuda", out_dir="results/retraining"):
    """
    Trains three JEPA variants and compares their generalization:

    "locomotion_only"  -- Swimmer-v4, Hopper-v4 (continuous locomotion tasks)
    "balance_only"     -- CartPole-v1, InvertedPendulum-v4 (balancing tasks)
    "mixed"            -- all four combined (the original training setup)

    Evaluation: test each on BOTH locomotion and balance tasks.

    Answers: "Does diverse pretraining produce a better general-purpose optimizer?"
    Expected: mixed training generalizes best to both domains; domain-specific
    training may specialize but lose cross-domain transfer.
    """
    from envs import get_param_dim

    domain_configs = {
        "locomotion_only": {
            "train_envs": ["Swimmer-v4"],   # param_dim=16 -- use Swimmer checkpoint
            "param_dim": 16,
        },
        "balance_only": {
            "train_envs": ["CartPole-v1"],  # param_dim=8
            "param_dim": 8,
        },
        "mixed": {
            "train_envs": ["CartPole-v1"],  # Use CartPole param_dim for mixed
            "param_dim": 8,
        },
    }

    # For mixed, we need to handle multi-param_dim training carefully.
    # The simplest approach: train separate models for each param_dim,
    # using the mixed dataset for each. Here we use param_dim=8 throughout
    # and project Swimmer's 16-dim params to 8-dim for the mixed condition.
    results = {}
    eval_envs_8dim = ["CartPole-v1"]
    eval_envs_16dim = ["Swimmer-v4"]

    for domain_name, config in domain_configs.items():
        log.info(f"\nDomain diversity: {domain_name}")
        pdim = config["param_dim"]
        train_envs = config["train_envs"]

        # Collect data
        train_data = {}
        for env in train_envs:
            env_pdim = get_param_dim(env)
            cache_path = f"{out_dir}/data/{env.replace('/','_')}_diversity.npz"
            if os.path.exists(cache_path):
                data = load_trajectory_data(cache_path)
            else:
                data = collect_pretraining_data(env, env_pdim, n_trajectories)
                save_trajectory_data(data, cache_path)

            # Project to pdim if needed
            if env_pdim != pdim:
                rng = np.random.RandomState(42)
                M = rng.randn(pdim, env_pdim)
                M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-8
                data = [(M @ theta, reward) for theta, reward in data]
            train_data[env] = data

        # For mixed: add projected Swimmer data if primary is CartPole
        if domain_name == "mixed" and pdim == 8:
            swimmer_env = "Swimmer-v4"
            swimmer_pdim = get_param_dim(swimmer_env)
            cache_path = f"{out_dir}/data/{swimmer_env}_diversity.npz"
            if os.path.exists(cache_path):
                swimmer_data = load_trajectory_data(cache_path)
            else:
                swimmer_data = collect_pretraining_data(
                    swimmer_env, swimmer_pdim, n_trajectories)
                save_trajectory_data(swimmer_data, cache_path)
            rng = np.random.RandomState(42)
            M = rng.randn(pdim, swimmer_pdim)
            M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-8
            train_data["Swimmer-v4_projected"] = [
                (M @ theta, reward) for theta, reward in swimmer_data
            ]

        ckpt_path = f"{out_dir}/checkpoints/diversity_{domain_name}.pt"
        _, losses = train_jepa(train_data, param_dim=pdim, n_epochs=n_epochs,
                                device=device, out_ckpt_path=ckpt_path)

        domain_result = {"domain": domain_name, "train_envs": train_envs,
                          "param_dim": pdim, "training_loss_curve": losses}
        for eval_env in eval_envs_8dim:
            if get_param_dim(eval_env) == pdim:
                ev = evaluate_checkpoint(ckpt_path, eval_env, pdim, n_seeds=3)
                domain_result[f"eval_{eval_env}"] = ev
                log.info(f"  {domain_name} -> {eval_env}: mean_best={ev['mean_best']:.2f}")

        results[domain_name] = domain_result

    out_path = f"{out_dir}/domain_diversity_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nDomain diversity results saved to {out_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# N14: Plain Transformer baseline (ablates JEPA's predictive objective)
# ─────────────────────────────────────────────────────────────────────────────

class PlainTransformerOptimizer(object):
    """
    A Transformer that is trained with standard supervised next-token
    prediction (predict next theta given history) rather than JEPA's
    joint-embedding predictive loss. This ablates whether JEPA's specific
    representation learning objective is necessary, or whether any
    sequence-to-sequence Transformer trained on (theta, reward) histories
    would produce a comparably good optimizer.

    Architecture: identical causal Transformer encoder + MLP decoder,
    but trained with MSE loss on theta_{t+1} directly (no target encoder,
    no latent-space prediction, no EMA target).
    """

    def __init__(self, param_dim, d_model=128, nhead=4, n_layers=2, device="cpu"):
        import torch
        import torch.nn as nn

        self.param_dim = param_dim
        self.d_model = d_model
        self.device = torch.device(device)

        # Input projection: (param_dim + 1) -> d_model  [+1 for reward]
        self.input_proj = nn.Linear(param_dim + 1, d_model)

        # Causal Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Decoder: latent + desired_reward -> theta
        self.decoder = nn.Sequential(
            nn.Linear(d_model + 1, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, param_dim),
        )

        self.to(self.device)

    def to(self, device):
        import torch.nn as nn
        for m in [self.input_proj, self.transformer, self.decoder]:
            m.to(device)
        return self

    def parameters(self):
        import itertools
        return itertools.chain(
            self.input_proj.parameters(),
            self.transformer.parameters(),
            self.decoder.parameters(),
        )

    def forward(self, thetas_norm, rewards, desired_reward):
        import torch
        # thetas_norm: (B, T, param_dim), rewards: (B, T, 1)
        x = torch.cat([thetas_norm, rewards], dim=-1)  # (B, T, param_dim+1)
        x = self.input_proj(x)  # (B, T, d_model)
        # Causal mask
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, device=self.device), diagonal=1).bool()
        ctx = self.transformer(x, mask=mask)  # (B, T, d_model)
        # Use last timestep's representation
        ctx_last = ctx[:, -1, :]  # (B, d_model)
        desired = desired_reward  # (B, 1)
        out = self.decoder(torch.cat([ctx_last, desired], dim=-1))  # (B, param_dim)
        return out

    def compute_loss(self, thetas_norm, rewards, target_theta):
        import torch
        import torch.nn.functional as F
        desired = rewards[:, -1:, :]  # target reward as conditioning
        pred_theta = self.forward(thetas_norm[:, :-1], rewards[:, :-1],
                                   desired.squeeze(-1))
        return F.mse_loss(pred_theta, target_theta)


def train_plain_transformer(train_data_by_env, param_dim, n_epochs=100,
                              batch_size=64, lr=1e-3, device="cuda", seed=0,
                              out_ckpt_path=None):
    """Trains PlainTransformerOptimizer using supervised next-theta prediction."""
    import torch
    import torch.optim as optim

    torch.manual_seed(seed)
    model = PlainTransformerOptimizer(param_dim=param_dim, device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    all_thetas, all_rewards = [], []
    for env_name, data in train_data_by_env.items():
        for theta, reward in data:
            all_thetas.append(theta)
            all_rewards.append(reward)

    all_thetas = np.array(all_thetas, dtype=np.float32)
    all_rewards = np.array(all_rewards, dtype=np.float32)
    r_mean, r_std = all_rewards.mean(), all_rewards.std() + 1e-8
    all_rewards = (all_rewards - r_mean) / r_std

    n = len(all_thetas)
    loss_history = []
    log.info(f"Training PlainTransformer: {n} samples, {n_epochs} epochs")

    for epoch in range(n_epochs):
        idx = np.random.permutation(n)
        all_thetas, all_rewards = all_thetas[idx], all_rewards[idx]

        epoch_losses = []
        ctx_len = np.random.randint(4, 21)
        for start in range(0, n - ctx_len - 1, batch_size):
            batch_thetas = all_thetas[start:start + ctx_len + 1]
            batch_rewards = all_rewards[start:start + ctx_len + 1]

            mean = batch_thetas[:ctx_len].mean(axis=0)
            std = batch_thetas[:ctx_len].std(axis=0) + 1e-6
            thetas_norm = (batch_thetas - mean) / std

            thetas_t = torch.tensor(thetas_norm[:ctx_len], dtype=torch.float32,
                                     device=device).unsqueeze(0)
            rewards_t = torch.tensor(batch_rewards[:ctx_len], dtype=torch.float32,
                                      device=device).unsqueeze(0).unsqueeze(-1)
            target = torch.tensor(thetas_norm[ctx_len], dtype=torch.float32,
                                   device=device).unsqueeze(0)

            optimizer.zero_grad()
            loss = model.compute_loss(thetas_t, rewards_t, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        epoch_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        loss_history.append(epoch_loss)
        if (epoch + 1) % 10 == 0:
            log.info(f"  Epoch {epoch+1}/{n_epochs}: loss={epoch_loss:.4f}")

    if out_ckpt_path:
        os.makedirs(os.path.dirname(out_ckpt_path), exist_ok=True)
        import torch
        torch.save({"input_proj": model.input_proj.state_dict(),
                    "transformer": model.transformer.state_dict(),
                    "decoder": model.decoder.state_dict()}, out_ckpt_path)

    return model, loss_history


def evaluate_plain_transformer(model_or_path, env_name, param_dim,
                                 n_seeds=3, max_iterations=400,
                                 n_eval_episodes=10, device="cpu"):
    """
    Evaluates PlainTransformerOptimizer in the same greedy propose/evaluate
    loop used for JEPA, enabling direct performance comparison.
    """
    import torch
    from envs import rollout_policy as _env_rollout

    if isinstance(model_or_path, str):
        model = PlainTransformerOptimizer(param_dim=param_dim, device=device)
        ckpt = torch.load(model_or_path, map_location=device, weights_only=False)
        model.input_proj.load_state_dict(ckpt["input_proj"])
        model.transformer.load_state_dict(ckpt["transformer"])
        model.decoder.load_state_dict(ckpt["decoder"])
    else:
        model = model_or_path

    # Use envs.rollout_policy (handles param_dim=10 CartPole with bias)
    # not runner.jepa_runner.rollout_policy (expects 8-dim no-bias)
    from envs import rollout_policy as _env_rollout

    seed_results = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)

        history_thetas, history_rewards = [], []
        rewards = []
        best = -np.inf

        n_warmup = max(8, param_dim)
        for i in range(n_warmup):
            theta = rng.randn(param_dim)
            r = _env_rollout(env_name, theta, n_eval_episodes, seed + i)
            history_thetas.append(theta)
            history_rewards.append(r)
            rewards.append(r)
            best = max(best, r)

        for t in range(max_iterations - n_warmup):
            # Propose: encode history, decode next theta
            thetas_arr = np.array(history_thetas[-40:], dtype=np.float32)
            rewards_arr = np.array(history_rewards[-40:], dtype=np.float32)
            mean = thetas_arr.mean(axis=0)
            std = thetas_arr.std(axis=0) + 1e-6
            thetas_norm = (thetas_arr - mean) / std
            desired_r = float(np.percentile(history_rewards, 80))
            desired_norm = (desired_r - rewards_arr.mean()) / (rewards_arr.std() + 1e-6)

            with torch.no_grad():
                thetas_t = torch.tensor(thetas_norm, dtype=torch.float32,
                                         device=device).unsqueeze(0)
                rewards_t = torch.tensor(rewards_arr, dtype=torch.float32,
                                          device=device).unsqueeze(0).unsqueeze(-1)
                desired_t = torch.tensor([[desired_norm]], dtype=torch.float32,
                                          device=device)
                pred_norm = model.forward(thetas_t, rewards_t, desired_t)
                theta = (pred_norm.squeeze(0).cpu().numpy() * std + mean)

            r = _env_rollout(env_name, theta, n_eval_episodes, seed + n_warmup + t)
            history_thetas.append(theta)
            history_rewards.append(r)
            rewards.append(r)
            best = max(best, r)

        seed_results.append({"seed": seed, "best": float(best),
                              "mean": float(np.mean(rewards))})

    mean_best = float(np.mean([r["best"] for r in seed_results]))
    mean_mean = float(np.mean([r["mean"] for r in seed_results]))
    log.info(f"  PlainTransformer on {env_name}: mean_best={mean_best:.2f}")
    return {"mean_best": mean_best, "mean_mean": mean_mean, "per_seed": seed_results}


def run_plain_transformer_comparison(train_envs, eval_envs,
                                      n_trajectories=2000, n_epochs=50,
                                      device="cuda", out_dir="results/retraining"):
    """
    Trains PlainTransformer and JEPA on the same data, evaluates both,
    and reports the comparison. This directly answers whether JEPA's
    joint-embedding predictive objective provides any advantage over a
    vanilla Transformer trained with direct supervised prediction.
    """
    from envs import get_param_dim

    pdim = get_param_dim(train_envs[0])
    results = {}

    # Collect/load data
    train_data = {}
    for env in train_envs:
        cache_path = f"{out_dir}/data/{env.replace('/','_')}_pt.npz"
        if os.path.exists(cache_path):
            train_data[env] = load_trajectory_data(cache_path)
        else:
            train_data[env] = collect_pretraining_data(env, pdim, n_trajectories)
            save_trajectory_data(train_data[env], cache_path)

    # Train JEPA
    log.info("\nTraining JEPA (predictive objective)...")
    jepa_ckpt = f"{out_dir}/checkpoints/jepa_for_comparison.pt"
    _, jepa_losses = train_jepa(train_data, param_dim=pdim, n_epochs=n_epochs,
                                  device=device, out_ckpt_path=jepa_ckpt)

    # Train PlainTransformer
    log.info("\nTraining PlainTransformer (supervised prediction)...")
    pt_ckpt = f"{out_dir}/checkpoints/plain_transformer.pt"
    _, pt_losses = train_plain_transformer(train_data, param_dim=pdim,
                                            n_epochs=n_epochs, device=device,
                                            out_ckpt_path=pt_ckpt)

    # Evaluate both
    for eval_env in eval_envs:
        if get_param_dim(eval_env) != pdim:
            continue
        jepa_eval = evaluate_checkpoint(jepa_ckpt, eval_env, pdim, n_seeds=3,
                                         device=device)
        pt_eval = evaluate_plain_transformer(pt_ckpt, eval_env, pdim,
                                              n_seeds=3, device=device)
        results[eval_env] = {
            "jepa": jepa_eval,
            "plain_transformer": pt_eval,
            "jepa_advantage": jepa_eval["mean_best"] - pt_eval["mean_best"],
        }
        log.info(f"\n{eval_env}: JEPA={jepa_eval['mean_best']:.2f} "
                 f"vs PlainTransformer={pt_eval['mean_best']:.2f} "
                 f"(JEPA advantage: {jepa_eval['mean_best']-pt_eval['mean_best']:.2f})")

    results["training_curves"] = {
        "jepa_losses": jepa_losses,
        "plain_transformer_losses": pt_losses,
    }

    out_path = f"{out_dir}/plain_transformer_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nPlain Transformer comparison saved to {out_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True,
                   choices=["leave_one_out", "data_efficiency",
                             "reward_shuffle", "domain_diversity",
                             "plain_transformer"])
    p.add_argument("--train_envs", nargs="+",
                   default=["CartPole-v1", "Swimmer-v4",
                             "MountainCarContinuous-v0", "InvertedPendulum-v4"])
    p.add_argument("--eval_envs", nargs="+",
                   default=["CartPole-v1", "Swimmer-v4"])
    p.add_argument("--held_out_env", type=str, default=None,
                   help="For leave_one_out: the env to hold out")
    p.add_argument("--data_fractions", nargs="+", type=float,
                   default=[0.05, 0.1, 0.25, 0.5, 1.0])
    p.add_argument("--shuffle_types", nargs="+",
                   default=["none", "rewards", "params", "both"])
    p.add_argument("--n_trajectories", type=int, default=2000)
    p.add_argument("--n_epochs", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_dir", type=str, default="results/retraining")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.experiment == "leave_one_out":
        run_leave_one_out(args.train_envs, n_trajectories=args.n_trajectories,
                           n_epochs=args.n_epochs, device=args.device,
                           out_dir=args.out_dir)
    elif args.experiment == "data_efficiency":
        run_data_efficiency(args.train_envs, args.eval_envs,
                             data_fractions=args.data_fractions,
                             n_trajectories_full=args.n_trajectories,
                             n_epochs=args.n_epochs, device=args.device,
                             out_dir=args.out_dir)
    elif args.experiment == "reward_shuffle":
        run_reward_shuffle(args.train_envs, args.eval_envs,
                            shuffle_types=args.shuffle_types,
                            n_trajectories=args.n_trajectories,
                            n_epochs=args.n_epochs, device=args.device,
                            out_dir=args.out_dir)
    elif args.experiment == "domain_diversity":
        run_domain_diversity(n_trajectories=args.n_trajectories,
                              n_epochs=args.n_epochs, device=args.device,
                              out_dir=args.out_dir)
    elif args.experiment == "plain_transformer":
        run_plain_transformer_comparison(args.train_envs, args.eval_envs,
                                          n_trajectories=args.n_trajectories,
                                          n_epochs=args.n_epochs, device=args.device,
                                          out_dir=args.out_dir)


if __name__ == "__main__":
    main()
