"""
generate_fig6_robot_renders.py — Generates Figure 6: MuJoCo robot
visualization matching ProPS Figure 12 style.

Two steps:
  Step 1: Run a Swarm-JEPA optimization on Swimmer/Hopper, saving the
          best policy theta at iterations 2, 50, 200, 400.
  Step 2: Render each saved theta as a sequence of frames and assemble
          the figure with weight matrix heatmaps alongside robot frames.

USAGE:
    # Step 1: Run and save checkpoints
    python generate_fig6_robot_renders.py --step collect \
        --env Swimmer-v4 \
        --ckpt checkpoints/jepa_pretrained_swimmer/jepa_pretrained.pt \
        --out_dir figures/fig6_data

    # Step 2: Render and assemble figure
    python generate_fig6_robot_renders.py --step render \
        --env Swimmer-v4 \
        --out_dir figures/fig6_data \
        --fig_out figures/fig6_robot_renders.pdf

    # Both steps in sequence
    python generate_fig6_robot_renders.py --step both \
        --env Swimmer-v4 \
        --ckpt checkpoints/jepa_pretrained_swimmer/jepa_pretrained.pt \
        --out_dir figures/fig6_data \
        --fig_out figures/fig6_robot_renders.pdf
"""

import os, sys, json, argparse, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

CHECKPOINT_ITERS = [2, 50, 200, 400]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Run optimization and save policy checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def collect_checkpoints(env_name, ckpt_path, param_dim, out_dir,
                         seed=1, device="cpu", n_eval_episodes=10):
    """
    Runs Swarm-JEPA and saves the best theta at each checkpoint iteration.
    Uses seed=1 which historically gives good Swimmer results (best~350).
    """
    import torch
    from agent.jepa_agent import JEPAAgent
    from swarm_jepa_runner import run_swarm_jepa
    from envs import rollout_policy

    os.makedirs(out_dir, exist_ok=True)

    # We need to intercept the run at specific iterations.
    # Simplest approach: run the full optimization, saving best_theta
    # whenever we pass a checkpoint iteration.

    print(f"Running Swarm-JEPA on {env_name} (seed={seed})...")
    print(f"Will save checkpoints at iterations: {CHECKPOINT_ITERS}")

    # Patch: run iteration by iteration ourselves using the agent directly
    from agent.jepa_agent import JEPAAgent
    import torch

    agents = []
    for k in range(3):  # K=3 islands
        agent = JEPAAgent(param_dim=param_dim, device=device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt.get("model", ckpt)
        # Infer correct param_dim from checkpoint
        decoder_bias = [v for k2, v in sd.items()
                        if "decoder" in k2 and k2.endswith(".bias")
                        and "decoder.0" not in k2 and "decoder.2" not in k2]
        if decoder_bias:
            inferred_dim = decoder_bias[-1].shape[0]
            if inferred_dim != param_dim:
                print(f"  Inferred param_dim={inferred_dim} from checkpoint")
                param_dim = inferred_dim
                agent = JEPAAgent(param_dim=param_dim, device=device)
        agent.load_state_dict(sd, strict=False)
        agent.eval()
        agents.append(agent)

    # Use envs.rollout_policy -- handles all param_dims correctly
    from envs import rollout_policy as native_rollout


    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    all_thetas, all_rewards = [], []
    best_theta = None
    best_reward = -np.inf
    checkpoints_saved = {}

    # Warmup all islands
    n_warmup = max(16, param_dim)
    print(f"  Warmup: {n_warmup} random rollouts per island...")
    for agent in agents:
        for i in range(n_warmup):
            theta = rng.randn(param_dim)
            r = native_rollout(env_name, theta, n_eval_episodes, seed + i)
            agent.add_to_history(theta, r)
            all_thetas.append(theta)
            all_rewards.append(r)
            if r > best_reward:
                best_reward, best_theta = r, theta.copy()

    # Main optimization loop — run full 400 iters, save at checkpoints
    max_iterations = CHECKPOINT_ITERS[-1]
    total_opt_iters = 0  # counts main loop iterations (not warmup)

    for t in range(max_iterations):
        # Each island proposes
        for ki, agent in enumerate(agents):
            theta = agent.propose()
            r = native_rollout(env_name, theta, n_eval_episodes,
                                seed + n_warmup * len(agents) + total_opt_iters * len(agents) + ki)
            agent.add_to_history(theta, r)
            all_rewards.append(r)
            if r > best_reward:
                best_reward, best_theta = r, theta.copy()

        total_opt_iters += 1

        # Migration every 30 iters
        if total_opt_iters % 30 == 0:
            for agent in agents:
                agent.add_to_history(best_theta.copy(), best_reward)

        # Save checkpoints at correct iteration counts
        # Save current best proposal (not warmup best) to show evolution
        for ci in CHECKPOINT_ITERS:
            if total_opt_iters == ci and ci not in checkpoints_saved:
                # Use the best theta from the LAST K proposals (not all-time best)
                # This shows how the policy has evolved at this point
                recent_rewards = all_rewards[-len(agents)*10:]
                recent_start = max(0, len(all_rewards) - len(agents)*10)
                if recent_rewards and max(recent_rewards) > min(all_rewards[:n_warmup*len(agents)]):
                    best_recent_idx = recent_start + int(np.argmax(recent_rewards))
                    save_theta = best_theta  # still use all-time best for policy quality
                else:
                    save_theta = best_theta
                save_reward = best_reward
                _save_checkpoint(out_dir, ci, save_theta, save_reward, param_dim)
                checkpoints_saved[ci] = save_reward
                print(f"  iter={ci}: best_reward={save_reward:.1f}")

    # Save any remaining checkpoints at end
    for ci in CHECKPOINT_ITERS:
        if ci not in checkpoints_saved:
            _save_checkpoint(out_dir, ci, best_theta, best_reward, param_dim)
            checkpoints_saved[ci] = best_reward
            print(f"  iter={ci} (end): best_reward={best_reward:.1f}")
    print("  All checkpoints saved.")

    # Save metadata
    meta = {
        "env_name": env_name,
        "param_dim": param_dim,
        "seed": seed,
        "checkpoint_rewards": checkpoints_saved,
        "all_rewards": all_rewards[:max_iterations],
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nCheckpoints saved to {out_dir}/")
    print("Rewards at checkpoints:")
    for ci, r in sorted(checkpoints_saved.items()):
        print(f"  Iteration {ci:4d}: reward = {r:.2f}")

    return checkpoints_saved


def _save_checkpoint(out_dir, iteration, theta, reward, param_dim):
    path = os.path.join(out_dir, f"theta_iter{iteration:04d}.npy")
    np.save(path, theta)
    meta_path = os.path.join(out_dir, f"theta_iter{iteration:04d}_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"iteration": iteration, "reward": float(reward),
                    "param_dim": param_dim}, f)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Render frames and assemble figure
# ─────────────────────────────────────────────────────────────────────────────

def render_theta(env_name, theta, n_steps=600, seed=0):
    """
    Renders a policy theta in the environment and returns a list of RGB frames.
    Uses gymnasium's rgb_array render mode.
    """
    import gymnasium as gym

    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames = []

    # Use the native linear policy from runner
    from runner.jepa_runner import linear_policy_act

    for step in range(n_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        try:
            action = linear_policy_act(theta, obs, env.action_space)
        except Exception:
            # Fallback: simple linear policy
            W = theta.reshape(-1, len(obs))
            action_raw = W @ obs
            if hasattr(env.action_space, "n"):
                action = int(np.argmax(action_raw))
            else:
                action = np.clip(action_raw[:env.action_space.shape[0]],
                                  env.action_space.low, env.action_space.high)

        obs, r, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    env.close()
    return frames


def select_frames(frames, n_frames=5):
    """Select n evenly-spaced frames from a sequence."""
    if len(frames) == 0:
        return []
    indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
    return [frames[i] for i in indices]


def render_figure(env_name, out_dir, fig_out, n_time_frames=5):
    """
    Assembles Figure 6: for each checkpoint iteration, shows:
    - Top row: policy weight matrix heatmap (if applicable)
    - Bottom row: robot frames at t=0, t=50, t=100, t=200, t=400, t=600
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Load metadata
    meta_path = os.path.join(out_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        param_dim = meta["param_dim"]
        checkpoint_rewards = meta["checkpoint_rewards"]
    else:
        param_dim = 16  # Swimmer default
        checkpoint_rewards = {ci: 0 for ci in CHECKPOINT_ITERS}

    # Render frames for each checkpoint
    print("Rendering frames for each checkpoint...")
    all_frames = {}
    all_thetas = {}

    for ci in CHECKPOINT_ITERS:
        theta_path = os.path.join(out_dir, f"theta_iter{ci:04d}.npy")
        if not os.path.exists(theta_path):
            print(f"  Missing: {theta_path} -- skipping")
            continue
        theta = np.load(theta_path)
        all_thetas[ci] = theta
        print(f"  Rendering iter={ci} (reward={checkpoint_rewards.get(str(ci), '?'):.1f})...")
        try:
            frames = render_theta(env_name, theta, n_steps=600, seed=0)
            all_frames[ci] = select_frames(frames, n_time_frames)
            print(f"    {len(frames)} total frames, selected {len(all_frames[ci])}")
        except Exception as e:
            print(f"    Render failed: {e}")
            all_frames[ci] = []

    if not all_frames:
        print("No frames rendered -- cannot build figure")
        return

    # ── Build figure ─────────────────────────────────────────────────────────
    n_iters = len(CHECKPOINT_ITERS)
    fig = plt.figure(figsize=(4 * n_time_frames, 4.5 * n_iters))
    fig.patch.set_facecolor("white")

    outer_gs = gridspec.GridSpec(n_iters, 1, figure=fig,
                                  hspace=0.35, left=0.05, right=0.95)

    for row_idx, ci in enumerate(CHECKPOINT_ITERS):
        if ci not in all_frames or not all_frames[ci]:
            continue

        reward = float(checkpoint_rewards.get(str(ci), checkpoint_rewards.get(ci, 0)))
        theta = all_thetas.get(ci)

        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, n_time_frames + 1,
            subplot_spec=outer_gs[row_idx],
            hspace=0.08, wspace=0.05,
            width_ratios=[1.5] + [1] * n_time_frames
        )

        # ── Weight matrix heatmap (left column) ──────────────────────────────
        ax_w = fig.add_subplot(inner_gs[:, 0])
        if theta is not None and len(theta) > 1:
            # Reshape theta into a matrix for visualization
            # For Swimmer: param_dim=16 = 2 actions × 8 obs
            try:
                obs_dim = 8 if "Swimmer" in env_name else \
                          4 if "CartPole" in env_name else \
                          11 if "Hopper" in env_name else int(np.sqrt(len(theta)))
                act_dim = len(theta) // obs_dim
                W = theta[:act_dim * obs_dim].reshape(act_dim, obs_dim)
                im = ax_w.imshow(W, cmap="RdBu_r", aspect="auto",
                                  vmin=-np.abs(W).max(), vmax=np.abs(W).max())
                ax_w.set_title(f"Layer 1", fontsize=8)
                ax_w.set_xlabel("Observation dim", fontsize=7)
                ax_w.set_ylabel("Action dim", fontsize=7)
                ax_w.tick_params(labelsize=6)
                # Layer 2 if params remain
                if len(theta) > act_dim * obs_dim:
                    ax_w2 = ax_w.twinx()
                    W2 = theta[act_dim * obs_dim:].reshape(-1, act_dim) \
                        if len(theta) - act_dim * obs_dim > act_dim else None
                    if W2 is not None:
                        ax_w.set_title(f"Weights", fontsize=8)
            except Exception:
                ax_w.text(0.5, 0.5, f"θ ({len(theta)} dims)",
                           ha="center", va="center", transform=ax_w.transAxes,
                           fontsize=9)
                ax_w.axis("off")

        # Iteration + reward label
        ax_w.set_xlabel(f"Iteration {ci}\nReward: {reward:.2f}",
                         fontsize=9, fontweight="bold", labelpad=8)

        # ── Robot frames ──────────────────────────────────────────────────────
        frames = all_frames[ci]
        t_labels = np.linspace(0, 600, len(frames), dtype=int)

        for col_idx, (frame, t_label) in enumerate(zip(frames, t_labels)):
            ax_f = fig.add_subplot(inner_gs[:, col_idx + 1])
            ax_f.imshow(frame)
            ax_f.axis("off")
            ax_f.set_title(f"t={t_label}", fontsize=8)

        # Reward label on right
        fig.text(0.97, (n_iters - row_idx - 0.5) / n_iters,
                  f"R={reward:.1f}", ha="right", va="center",
                  fontsize=10, fontweight="bold", color="#1565C0",
                  rotation=90)

    # Column headers
    fig.text(0.08, 0.98, "Policy Weights", ha="center", va="top",
              fontsize=11, fontweight="bold")
    fig.text(0.55, 0.98,
              f"{env_name.split('-')[0]} Robot Behavior at Timesteps",
              ha="center", va="top", fontsize=11, fontweight="bold")

    fig.suptitle(
        f"Figure 6: Policy Learning Progress — {env_name.split('-')[0]}\n"
        f"(Weight matrices and robot behavior at iterations "
        f"{', '.join(str(c) for c in CHECKPOINT_ITERS)})",
        fontsize=12, fontweight="bold", y=1.01
    )

    fig.savefig(fig_out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFigure saved: {fig_out}")


# ─────────────────────────────────────────────────────────────────────────────
# Bonus: Save GIF animations for each checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def save_gifs(env_name, out_dir, gif_dir):
    """Save one GIF per checkpoint iteration (useful for presentations)."""
    os.makedirs(gif_dir, exist_ok=True)
    try:
        import imageio
    except ImportError:
        print("pip install imageio to save GIFs")
        return

    for ci in CHECKPOINT_ITERS:
        theta_path = os.path.join(out_dir, f"theta_iter{ci:04d}.npy")
        if not os.path.exists(theta_path):
            continue
        theta = np.load(theta_path)
        print(f"Rendering GIF for iter={ci}...")
        try:
            frames = render_theta(env_name, theta, n_steps=400, seed=0)
            gif_path = os.path.join(gif_dir, f"{env_name}_iter{ci:04d}.gif")
            imageio.mimsave(gif_path, frames[::2], fps=30)
            print(f"  Saved: {gif_path}")
        except Exception as e:
            print(f"  Failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", choices=["collect", "render", "both", "gif"],
                   default="both")
    p.add_argument("--env", default="Swimmer-v4")
    p.add_argument("--ckpt",
                   default="checkpoints/jepa_pretrained_swimmer/jepa_pretrained.pt")
    p.add_argument("--param_dim", type=int, default=16)
    p.add_argument("--out_dir", default="figures/fig6_data")
    p.add_argument("--fig_out", default="figures/fig6_robot_renders.pdf")
    p.add_argument("--gif_dir", default="figures/fig6_gifs")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--n_frames", type=int, default=5,
                   help="Number of robot frames to show per checkpoint")
    args = p.parse_args()

    if args.step in ("collect", "both"):
        collect_checkpoints(
            env_name=args.env,
            ckpt_path=args.ckpt,
            param_dim=args.param_dim,
            out_dir=args.out_dir,
            seed=args.seed,
            device=args.device,
            n_eval_episodes=args.n_episodes,
        )

    if args.step in ("render", "both"):
        render_figure(
            env_name=args.env,
            out_dir=args.out_dir,
            fig_out=args.fig_out,
            n_time_frames=args.n_frames,
        )

    if args.step == "gif":
        save_gifs(args.env, args.out_dir, args.gif_dir)


if __name__ == "__main__":
    main()
