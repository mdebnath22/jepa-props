"""
KarmaJEPA v2 — Corrected Integration
======================================
v1 BUG (diagnosed from CartPole/Swimmer regression):
  The anisotropic "K" perturbation was added ON TOP OF an already-complete
  PSO proposal from original_propose(), which internally already samples
  noise at self.noise_std and picks the best candidate. This meant every
  proposal received PSO's tuned noise PLUS an uncoordinated extra random
  walk -- doubling effective noise rather than shaping it. Result: Swimmer
  mean dropped 263.43 -> 155.24, CartPole variance nearly doubled.

  Root design flaw: K's anisotropic noise was computed in PARAMETER space
  (per parameter dimension) but PSO's actual noise injection happens in
  JEPA's LATENT space (s_x perturbation), inside original_propose(), which
  this module cannot see or shape from the outside without rewriting
  SwarmJEPAAgent.propose_for_island() internals directly.

v2 FIX:
  K2 ONLY: adaptive scalar noise_std (this part was already correctly
  REPLACING self.noise_std before calling original_propose(), not adding
  to its output -- so K2 alone should be tested as it was the part NOT
  implicated in the regression).

  K (anisotropic) REMOVED: the param-space perturbation-on-output is
  deleted entirely. If anisotropic shaping is wanted later, it needs to
  be implemented as a patch to propose_for_island's internal noise
  generation, not a post-hoc addition.

  M (consensus migration) is now INDEPENDENTLY TOGGLEABLE via a flag,
  so it can be ablated separately from K2 to isolate which mechanism
  helps or hurts.

USAGE:
    from swarm_jepa_agent import SwarmJEPAAgent
    from karma_jepa_v2 import patch_swarm_jepa_agent

    # Test K2 alone:
    patch_swarm_jepa_agent(SwarmJEPAAgent, enable_k2=True, enable_m=False)

    # Test M alone:
    patch_swarm_jepa_agent(SwarmJEPAAgent, enable_k2=False, enable_m=True)

    # Test both:
    patch_swarm_jepa_agent(SwarmJEPAAgent, enable_k2=True, enable_m=True)

    # Test neither (sanity check -- should match plain Swarm-JEPA exactly):
    patch_swarm_jepa_agent(SwarmJEPAAgent, enable_k2=False, enable_m=False)
"""

import numpy as np
import math
import logging
from typing import Optional, List, Tuple

log = logging.getLogger(__name__)


class KarmicLineageTracker:
    """K2 only: tracks reward-weighted signal strength to adapt noise_std.
    (K's anisotropic per-dimension shaping removed in v2 -- see module docstring.)
    """

    def __init__(self, param_dim: int, decay: float = 0.3, window: int = 100):
        self.param_dim = param_dim
        self.decay = decay
        self.window = window
        self._history: List[Tuple[np.ndarray, float, int]] = []
        self._current_t = 0
        self._tendency_mean_history: List[float] = []

    def add(self, theta: np.ndarray, reward: float):
        self._history.append((theta.copy(), reward, self._current_t))
        if len(self._history) > self.window:
            self._history.pop(0)
        self._current_t += 1

        # Compute and record this step's tendency strength (scalar signal quality)
        if len(self._history) >= 3:
            tau_mean = self._compute_tendency_strength()
            self._tendency_mean_history.append(tau_mean)

    def _compute_tendency_strength(self) -> float:
        """
        Scalar signal strength: how consistently does reward correlate with
        parameter magnitude across history? High = clear signal (low noise
        needed). Low = noisy/contradictory signal (high noise needed).
        """
        T = self._current_t
        r_arr = np.array([r for _, r, _ in self._history])
        r_min, r_max = r_arr.min(), r_arr.max()
        r_range = r_max - r_min + 1e-8

        scores = np.zeros(self.param_dim)
        for theta, reward, t in self._history:
            r_norm = (reward - r_min) / r_range
            decay_w = math.exp(-self.decay * (T - t))
            scores += r_norm * np.abs(theta) * decay_w

        s_max = scores.max()
        if s_max > 1e-8:
            scores = scores / s_max
        return float(scores.mean())

    def karma_noise_std(
        self,
        base_noise_std: float,
        noise_std_min: float = 0.1,
        noise_std_max: float = 1.0,
    ) -> float:
        """
        K2: adaptive noise_std. This REPLACES base_noise_std entirely
        (caller must use the returned value, not add to it).
        """
        if len(self._tendency_mean_history) < 3:
            return base_noise_std

        recent = self._tendency_mean_history[-5:]
        tau_bar = float(np.mean(recent))

        all_taus = self._tendency_mean_history
        tau_min = float(min(all_taus))
        tau_max = float(max(all_taus)) + 1e-9

        normed = (tau_bar - tau_min) / (tau_max - tau_min)
        karma_std = noise_std_max - normed * (noise_std_max - noise_std_min)
        return float(np.clip(karma_std, noise_std_min, noise_std_max))


class ConsensusTracker:
    """M: cross-island dimension consensus for migration noise shaping.
    Unchanged from v1 -- this mechanism was not implicated in the regression
    (it only affects migration events, which fire every 30 iters, not every
    proposal like K did).
    """

    def __init__(self, param_dim: int, n_islands: int, consensus_threshold: float = 0.6):
        self.param_dim = param_dim
        self.n_islands = n_islands
        self.consensus_threshold = consensus_threshold

    def consensus_dimensions(
        self,
        island_best_thetas: List[Optional[np.ndarray]],
        island_best_rewards: List[float],
    ) -> np.ndarray:
        valid = [(t, r) for t, r in zip(island_best_thetas, island_best_rewards)
                 if t is not None and r > -np.inf]
        if len(valid) < 2:
            return np.ones(self.param_dim)

        thetas = np.array([t for t, r in valid])
        rewards = np.array([r for t, r in valid])
        r_norm = (rewards - rewards.min()) / (np.ptp(rewards) + 1e-8)
        weights = r_norm + 0.1

        signs = np.sign(thetas)
        weighted_sign_mean = np.average(signs, axis=0, weights=weights)
        sign_consensus = np.abs(weighted_sign_mean)

        cv = np.std(thetas, axis=0) / (np.abs(np.mean(thetas, axis=0)) + 1e-8)
        cv_consensus = 1.0 / (1.0 + cv)

        consensus = 0.5 * sign_consensus + 0.5 * cv_consensus
        return np.clip(consensus, 0.0, 1.0)

    def migration_noise_weights(
        self,
        island_best_thetas: List[Optional[np.ndarray]],
        island_best_rewards: List[float],
    ) -> np.ndarray:
        cons = self.consensus_dimensions(island_best_thetas, island_best_rewards)
        return 2.0 - cons


def patch_swarm_jepa_agent(
    SwarmJEPAAgent_cls,
    enable_k2: bool = True,
    enable_m: bool = True,
):
    """
    Patches SwarmJEPAAgent with K2 and/or M, independently toggleable.
    Idempotent w.r.t. repeated calls with the SAME flags (re-patching with
    different flags on an already-patched class will re-wrap on top of the
    previous patch -- to test a different flag combination cleanly, restart
    the Python process / re-import the class fresh).
    """
    tag = f"k2={enable_k2}_m={enable_m}"
    patched_attr = f"_karma_patched_{tag}"
    if getattr(SwarmJEPAAgent_cls, "_karma_patched_any", False):
        log.warning(
            "SwarmJEPAAgent already patched with a different KarmaJEPA "
            "configuration in this process. Restart Python to test a "
            "different enable_k2/enable_m combination cleanly."
        )
        return SwarmJEPAAgent_cls

    original_init = SwarmJEPAAgent_cls.__init__
    original_add_to_island = SwarmJEPAAgent_cls.add_to_island
    original_propose = SwarmJEPAAgent_cls.propose_for_island
    original_migrate = SwarmJEPAAgent_cls.maybe_migrate

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if enable_k2:
            self._karma_trackers = [
                KarmicLineageTracker(self.param_dim) for _ in range(self.K)
            ]
        if enable_m:
            self._consensus_tracker = ConsensusTracker(self.param_dim, self.K)
        log.info(f"KarmaJEPA v2 activated: K2={enable_k2}, M={enable_m}")

    def new_add_to_island(self, island_id: int, theta, reward: float):
        original_add_to_island(self, island_id, theta, reward)
        if enable_k2:
            self._karma_trackers[island_id].add(theta, reward)

    def new_propose(self, island_id: int, t: int, total_T: int):
        if not enable_k2:
            return original_propose(self, island_id, t, total_T)

        tracker = self._karma_trackers[island_id]
        karma_std = tracker.karma_noise_std(
            self.noise_std,
            noise_std_min=self.noise_std * 0.3,
            noise_std_max=self.noise_std * 2.0,
        )

        # REPLACE noise_std for this single proposal call, do not add
        # anything to the output afterward.
        original_noise_std = self.noise_std
        self.noise_std = karma_std
        theta = original_propose(self, island_id, t, total_T)
        self.noise_std = original_noise_std

        return theta

    def new_migrate(self, t: int) -> bool:
        if not enable_m:
            return original_migrate(self, t)

        if t == 0 or t % self.migration_every != 0:
            return False
        if self._global_best_theta is None:
            return False

        island_thetas = [isl.personal_best_theta for isl in self.islands]
        island_rewards = [isl.personal_best_reward for isl in self.islands]
        noise_weights = self._consensus_tracker.migration_noise_weights(
            island_thetas, island_rewards
        )

        log.info(f"[t={t}] Consensus-Migration: broadcasting global best "
                 f"(r={self._global_best_reward:.1f}) consensus_weight_mean="
                 f"{noise_weights.mean():.2f}")

        for k, island in enumerate(self.islands):
            if not island.in_isolation:
                consensus_noise = np.random.randn(self.param_dim) * (
                    self.noise_std * 0.1 * noise_weights
                )
                migrated_theta = self._global_best_theta + consensus_noise
                self.add_to_island(k, migrated_theta, self._global_best_reward)
        return True

    SwarmJEPAAgent_cls.__init__ = new_init
    SwarmJEPAAgent_cls.add_to_island = new_add_to_island
    SwarmJEPAAgent_cls.propose_for_island = new_propose
    SwarmJEPAAgent_cls.maybe_migrate = new_migrate
    SwarmJEPAAgent_cls._karma_patched_any = True

    log.info(f"SwarmJEPAAgent patched: KarmaJEPA v2 (K2={enable_k2}, M={enable_m})")
    return SwarmJEPAAgent_cls
