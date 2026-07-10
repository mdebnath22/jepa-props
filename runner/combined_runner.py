"""
Combined JEPA-ProPS + Layered Validator
=========================================
The full combined system:

  JEPA proposes (θ_control, θ_morph) jointly
        ↓
  Morphology parameters → LLM generates MJCF XML
        ↓
  Layered Validator (with JEPA-SP2) repairs XML
        ↓
  Valid simulator → MuJoCo rollout of control policy
        ↓
  reward → history buffer → next JEPA proposal

This enables JEPA to do co-design: jointly optimize morphology and
control policy, with the validator ensuring every proposed morphology
actually compiles before we spend RL budget on it.

The validator here uses the JEPA repair model (JEPA-SP2-predict mode)
as its SP2 operator, creating a second JEPA loop inside the validator.

Key ablations this enables (from combined system):
  A1: Pure JEPA policy search (fixed morphology, validator unused)
  A2: JEPA policy search + validator (editdist SP2)
  A3: JEPA policy search + validator (JEPA-SP2-match)
  A4: JEPA policy search + validator (JEPA-SP2-predict)  ← full system
  A5: Co-design JEPA (θ_morph+θ_control) + validator JEPA-SP2-predict
"""

import os
import time
import logging
import numpy as np
import torch
from typing import Dict, Optional, Tuple, List
import xml.etree.ElementTree as ET

from agent.jepa_agent import JEPAAgent
from validator.layered_validator import LayeredValidator
from validator.jepa_repair_model import JEPARepairMatcher
from runner.jepa_runner import rollout_policy, _online_finetune

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Morphology parameter space
# ─────────────────────────────────────────────────────────────────────────────

def morph_params_to_xml(
    morph_params: np.ndarray,
    task: str = "locomotion",
    n_joints: int = 6,
) -> str:
    """
    Convert continuous morphology parameters to a MuJoCo MJCF string.

    morph_params encodes:
      - body segment lengths  (n_joints values, normalized to [0.05, 0.3])
      - body segment widths   (n_joints values, normalized to [0.02, 0.1])
      - gear ratios           (n_joints values, normalized to [20, 150])

    Total param_dim = 3 * n_joints

    This is a simplified morphology space. In the full co-design setup,
    you'd use a richer parameterization (or a grammar-constrained one).
    """
    n = n_joints
    lengths = 0.05 + 0.25 * (morph_params[:n] - morph_params[:n].min()) / \
              (morph_params[:n].ptp() + 1e-6)
    widths = 0.02 + 0.08 * (morph_params[n:2*n] - morph_params[n:2*n].min()) / \
             (morph_params[n:2*n].ptp() + 1e-6)
    gears = 20 + 130 * (morph_params[2*n:3*n] - morph_params[2*n:3*n].min()) / \
            (morph_params[2*n:3*n].ptp() + 1e-6)

    sides = ["FR", "FL", "RR", "RL", "MR", "ML"]
    bodies = []
    motors = []
    for i in range(n):
        jname = f"hip_{sides[i % len(sides)]}_{i}"
        bodies.append(f"""
    <body name="body_{i}" pos="0 {0.15*i:.3f} 0">
      <joint name="{jname}" type="hinge" axis="0 1 0" range="-90 90"/>
      <geom type="capsule" size="{widths[i]:.4f} {lengths[i]:.4f}"/>
    </body>""")
        motors.append(f'    <motor joint="{jname}" gear="{gears[i]:.1f}"/>')

    floor_line = "" if task == "swimming" else \
        '<geom name="floor" type="plane" size="20 20 0.1"/>'
    option_line = '<option viscosity="0.8" density="1000" gravity="0 0 0"/>' \
        if task == "swimming" else ""

    bodies_str = "".join(bodies)
    motors_str = "\n".join(motors)
    xml = f"""<mujoco model="jepa_morph_{task}">
  {option_line}
  <worldbody>
    {floor_line}
    <body name="torso" pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.2 0.1 0.1"/>
      {bodies_str}
    </body>
  </worldbody>
  <actuator>
{motors_str}
  </actuator>
</mujoco>"""
    return xml


# ─────────────────────────────────────────────────────────────────────────────
# Combined runner
# ─────────────────────────────────────────────────────────────────────────────

def run_combined_training_loop(
    # Environment
    env_name: str,
    control_param_dim: int,         # policy control parameters
    morph_param_dim: int = 0,       # 0 = fixed morphology (standard ProPS mode)
    n_joints: int = 6,
    task: str = "locomotion",
    # JEPA architecture (shared for control + morphology)
    d_model: int = 128,
    nhead: int = 4,
    num_enc_layers: int = 3,
    num_pred_layers: int = 3,
    dim_ff: int = 256,
    ema_decay: float = 0.996,
    desired_pct: float = 80.0,
    noise_std: float = 0.05,
    n_candidates: int = 5,
    # Validator
    sp2_mode: str = "jepa-predict",  # "editdist" | "jepa-match" | "jepa-predict"
    jepa_repair_ckpt: str = None,
    jepa_conf_threshold: float = 0.5,
    edit_dist_delta: float = 0.6,
    # Training
    max_iterations: int = 400,
    n_eval_episodes: int = 5,
    seed: int = 0,
    # Pretraining
    policy_ckpt: str = None,
    online_finetune: bool = True,
    online_lr: float = 1e-4,
    online_finetune_every: int = 10,
    online_finetune_steps: int = 5,
    # Logging
    log_dir: str = "results",
    run_name: str = None,
    device: str = "cpu",
    **kwargs,
) -> Dict:
    """
    Combined JEPA policy search with validator in the reward loop.

    If morph_param_dim > 0: co-design mode — JEPA proposes both morphology
    and control parameters jointly.

    If morph_param_dim == 0: standard mode — fixed morphology, only control
    parameters are optimized (equivalent to the original JEPA-ProPS runner
    but with validator-verified morphology).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(log_dir, exist_ok=True)
    run_name = run_name or f"combined_{env_name}_{sp2_mode}_seed{seed}"
    log_path = os.path.join(log_dir, f"{run_name}.csv")

    total_param_dim = control_param_dim + morph_param_dim

    log.info(
        f"[Combined] env={env_name} control_dim={control_param_dim} "
        f"morph_dim={morph_param_dim} sp2={sp2_mode} task={task}"
    )

    # ── Build JEPA policy agent ──────────────────────────────────────────
    agent = JEPAAgent(
        param_dim=total_param_dim,
        d_model=d_model,
        nhead=nhead,
        num_enc_layers=num_enc_layers,
        num_pred_layers=num_pred_layers,
        dim_ff=dim_ff,
        ema_decay=ema_decay,
        desired_pct=desired_pct,
        noise_std=noise_std,
        device=device,
    )

    if policy_ckpt and os.path.exists(policy_ckpt):
        log.info(f"Loading policy JEPA from {policy_ckpt}")
        agent.load_state_dict(torch.load(policy_ckpt, map_location=device), strict=False)

    # ── Build validator ──────────────────────────────────────────────────
    jepa_matcher = None
    if "jepa" in sp2_mode:
        jepa_matcher = JEPARepairMatcher(d_model=64, nhead=4)
        if jepa_repair_ckpt and os.path.exists(jepa_repair_ckpt):
            log.info(f"Loading JEPA repair model from {jepa_repair_ckpt}")
            jepa_matcher.load_state_dict(
                torch.load(jepa_repair_ckpt, map_location=device), strict=False
            )
        else:
            log.warning("No JEPA repair checkpoint — using random weights for SP2.")
        jepa_matcher.eval()

    validator = LayeredValidator(
        sp2_mode=sp2_mode,
        jepa_matcher=jepa_matcher,
        edit_dist_delta=edit_dist_delta,
        jepa_confidence_threshold=jepa_conf_threshold,
        task=task,
        device=device,
    )

    online_params = [p for n, p in agent.named_parameters()
                     if "ema_head" not in n and p.requires_grad]
    online_optimizer = torch.optim.AdamW(online_params, lr=online_lr)

    # ── Tracking ─────────────────────────────────────────────────────────
    all_rewards, all_params = [], []
    best_reward, best_params = -np.inf, None
    validator_stats = {"total": 0, "repaired": 0, "failed": 0, "sp2_calls": 0}
    t_start = time.time()

    with open(log_path, "w") as f:
        f.write("iteration,reward,best_reward,valid_before,valid_after,elapsed\n")

    # ── Warm-start ───────────────────────────────────────────────────────
    for i in range(5):
        theta = np.random.randn(total_param_dim)
        r, valid_pre, valid_post = _evaluate_with_validator(
            theta, env_name, control_param_dim, morph_param_dim,
            n_joints, task, validator, n_eval_episodes, seed + i,
            validator_stats,
        )
        agent.add_to_history(theta, r)
        log.info(f"  Warm-start {i}: r={r:.2f} valid_pre={valid_pre} valid_post={valid_post}")

    # ── Main loop ────────────────────────────────────────────────────────
    for t in range(max_iterations):
        theta = agent.propose(n_candidates=n_candidates)

        reward, valid_pre, valid_post = _evaluate_with_validator(
            theta, env_name, control_param_dim, morph_param_dim,
            n_joints, task, validator, n_eval_episodes, seed + 100 + t,
            validator_stats,
        )

        agent.add_to_history(theta, reward)
        all_rewards.append(reward)
        all_params.append(theta)

        if reward > best_reward:
            best_reward = reward
            best_params = theta.copy()

        elapsed = time.time() - t_start
        log.info(
            f"[{t+1}/{max_iterations}] r={reward:.2f} best={best_reward:.2f} "
            f"pre={valid_pre} post={valid_post} elapsed={elapsed:.1f}s"
        )

        with open(log_path, "a") as f:
            f.write(f"{t+1},{reward:.4f},{best_reward:.4f},"
                    f"{int(valid_pre)},{int(valid_post)},{elapsed:.2f}\n")

        if online_finetune and len(agent.history) >= 10 and \
                (t + 1) % online_finetune_every == 0:
            _online_finetune(agent, online_optimizer, online_finetune_steps, device)

    log.info(f"Validator stats: {validator_stats}")
    return {
        "rewards": all_rewards,
        "params": all_params,
        "best_reward": best_reward,
        "best_params": best_params,
        "validator_stats": validator_stats,
        "log_path": log_path,
    }


def _evaluate_with_validator(
    theta: np.ndarray,
    env_name: str,
    control_param_dim: int,
    morph_param_dim: int,
    n_joints: int,
    task: str,
    validator: LayeredValidator,
    n_eval_episodes: int,
    seed: int,
    stats: Dict,
) -> Tuple[float, bool, bool]:
    """
    Split theta into control + morph params, validate/repair morphology XML,
    then evaluate control policy. Returns (reward, valid_before, valid_after).
    """
    control_params = theta[:control_param_dim]

    if morph_param_dim > 0:
        # Co-design mode: validate and repair the proposed morphology
        morph_params = theta[control_param_dim:]
        xml_str = morph_params_to_xml(morph_params, task=task, n_joints=n_joints)

        is_valid_pre, _ = validator.validate(xml_str)
        stats["total"] += 1

        if not is_valid_pre:
            xml_str, success, repair_log = validator(xml_str)
            if success:
                stats["repaired"] += 1
            else:
                stats["failed"] += 1
                # Penalty for unrecoverable morphology
                return -100.0, False, False
        else:
            success = True

        valid_post = True  # if we got here, it compiled
        # In a real setup: load xml_str into MuJoCo and run PPO
        # For now: evaluate control policy in the fixed env
        reward = rollout_policy(
            env_name, control_params, n_eval_episodes, seed
        )
        return reward, is_valid_pre, valid_post

    else:
        # Standard mode: fixed morphology, just evaluate control params
        reward = rollout_policy(env_name, theta, n_eval_episodes, seed)
        return reward, True, True
