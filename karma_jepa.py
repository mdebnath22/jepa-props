"""
KarmaJEPA Extensions for SwarmJEPAAgent
=========================================
Transplants three mechanisms from the Hindu-philosophy paper (Conditions K, K2, M)
into JEPA's continuous latent-space search.

WHAT EACH MECHANISM DOES IN JEPA CONTEXT:

K  — Karmic Lineage Tracking:
     Computes per-parameter-dimension tendency scores τ(d) from the reward-weighted
     history with exponential recency decay. High-tendency dimensions get higher
     exploitation weight in PSO noise; low-tendency dimensions stay exploratory.
     Replaces uniform isotropic Gaussian noise with learned anisotropic noise.

K2 — Karma Temperature Scaling:
     Uses the mean karmic tendency strength to adaptively scale noise_std.
     Strong karmic signal → lower noise (commit to good region).
     Weak/contradictory signal → higher noise (escape / explore).
     Directly addresses Swimmer's stagnation pathology.

M  — Consensus Migration (sharpened):
     Instead of broadcasting the full global-best theta vector, identifies which
     parameter DIMENSIONS are consistently high-reward across all islands and
     biases migration noise toward those specific dimensions.
     More robust than whole-vector broadcast when different islands found
     different good regions.

These are DROP-IN additions to SwarmJEPAAgent — patch the existing class
rather than replacing it.
"""

import numpy as np
import math
from typing import Optional, List, Tuple


class KarmicLineageTracker:
    """
    Tracks per-parameter-dimension tendency scores across the search history.
    Equivalent to the paper's Eq. (3) but in continuous parameter space
    rather than discrete topological feature space.

    τ_t(d) = mean over recent history of [ r_t * |θ_t[d]| * exp(-λ*(T-t)) ]
             normalized by max absolute value to keep scores in [0,1]

    High τ(d) means dimension d has been strongly correlated with reward.
    Low τ(d) means dimension d is irrelevant or noisy relative to reward.
    """

    def __init__(self, param_dim: int, decay: float = 0.3, window: int = 100):
        self.param_dim = param_dim
        self.decay = decay
        self.window = window

        self._history: List[Tuple[np.ndarray, float, int]] = []  # (theta, reward, t)
        self._current_t = 0
        self._tendency: Optional[np.ndarray] = None
        self._tendency_mean_history: List[float] = []

    def add(self, theta: np.ndarray, reward: float):
        self._history.append((theta.copy(), reward, self._current_t))
        if len(self._history) > self.window:
            self._history.pop(0)
        self._current_t += 1
        self._tendency = None  # invalidate cache

    def tendency(self) -> np.ndarray:
        """
        Returns per-dimension tendency scores in [0, 1].
        Cached until a new observation is added.
        """
        if self._tendency is not None:
            return self._tendency

        if len(self._history) < 3:
            self._tendency = np.ones(self.param_dim) * 0.5
            return self._tendency

        T = self._current_t
        scores = np.zeros(self.param_dim)

        r_arr = np.array([r for _, r, _ in self._history])
        r_min, r_max = r_arr.min(), r_arr.max()
        r_range = r_max - r_min + 1e-8

        for theta, reward, t in self._history:
            # Normalize reward to [0, 1] so high-reward steps contribute more
            r_norm = (reward - r_min) / r_range
            decay_w = math.exp(-self.decay * (T - t))
            # Contribution: reward-weighted absolute parameter value
            # High |θ[d]| when reward is high → dimension d matters
            scores += r_norm * np.abs(theta) * decay_w

        # Normalize to [0, 1]
        s_max = scores.max()
        if s_max > 1e-8:
            scores = scores / s_max
        else:
            scores = np.ones(self.param_dim) * 0.5

        self._tendency = scores
        tau_mean = float(scores.mean())
        self._tendency_mean_history.append(tau_mean)
        return self._tendency

    def karma_noise_std(
        self,
        base_noise_std: float,
        noise_std_min: float = 0.1,
        noise_std_max: float = 1.0,
    ) -> float:
        """
        K2: Adaptive noise_std based on karmic signal strength.

        Strong, consistent karmic signal (high mean tendency) → lower noise
        (exploit the good region the karmic history points to).

        Weak/contradictory signal (low mean tendency) → higher noise
        (explore — the history doesn't point clearly anywhere).
        """
        if len(self._tendency_mean_history) < 3:
            return base_noise_std

        recent = self._tendency_mean_history[-5:]
        tau_bar = float(np.mean(recent))

        all_taus = self._tendency_mean_history
        tau_min = float(min(all_taus))
        tau_max = float(max(all_taus)) + 1e-9

        normed = (tau_bar - tau_min) / (tau_max - tau_min)

        # High tendency → low noise (commit); low tendency → high noise (explore)
        karma_std = noise_std_max - normed * (noise_std_max - noise_std_min)
        return float(np.clip(karma_std, noise_std_min, noise_std_max))

    def karma_noise_covariance(self, noise_std: float) -> np.ndarray:
        """
        K: Anisotropic noise covariance from tendency scores.

        Returns per-dimension noise std array:
        - High-tendency dims: exploit harder (noise_std * explore_scale)
          Actually LOWER noise on high-tendency dims to commit to them,
          but we keep them in the proposal so JEPA can refine there.
        - Low-tendency dims: explore more (noise_std * explore_scale)

        This replaces uniform isotropic noise in propose_for_island().
        """
        tau = self.tendency()

        # High-tendency dims: lower noise (exploit good region)
        # Low-tendency dims: higher noise (explore)
        # Scale: [0.5 * noise_std, 1.5 * noise_std] across dimensions
        dim_std = noise_std * (1.5 - tau)  # tau=1 → 0.5*std, tau=0 → 1.5*std
        return np.clip(dim_std, noise_std * 0.3, noise_std * 2.0)


class ConsensusTracker:
    """
    M: Tracks which parameter dimensions are consistently high-reward
    ACROSS multiple islands simultaneously (spatial consensus vs K's temporal).

    Instead of broadcasting the global-best full theta vector, identifies
    which DIMENSIONS of that vector are robustly good across islands.

    Migration injects noise that is stronger along consensus dimensions
    and weaker on island-specific dimensions.
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
        """
        Returns per-dimension consensus weight in [0, 1].
        A dimension has high consensus if it has a consistent sign and
        magnitude across all islands' best thetas.

        High consensus weight → this dimension is robustly good across islands.
        Low consensus weight → this dimension is island-specific.
        """
        valid = [(t, r) for t, r in zip(island_best_thetas, island_best_rewards)
                 if t is not None and r > -np.inf]

        if len(valid) < 2:
            return np.ones(self.param_dim)

        thetas = np.array([t for t, r in valid])
        rewards = np.array([r for t, r in valid])

        # Reward-weight the thetas
        r_norm = (rewards - rewards.min()) / (np.ptp(rewards) + 1e-8)
        weights = r_norm + 0.1  # avoid zero-weight

        # Per-dimension: measure agreement in sign across islands
        # High sign agreement → consensus; low → divergent
        signs = np.sign(thetas)
        weighted_sign_mean = np.average(signs, axis=0, weights=weights)
        # |weighted_sign_mean| ≈ 1 → all islands agree on direction
        # |weighted_sign_mean| ≈ 0 → islands disagree
        sign_consensus = np.abs(weighted_sign_mean)

        # Also measure coefficient of variation across islands (low CV = consensus)
        cv = np.std(thetas, axis=0) / (np.abs(np.mean(thetas, axis=0)) + 1e-8)
        cv_consensus = 1.0 / (1.0 + cv)  # high CV → low consensus, sigmoid-like

        consensus = 0.5 * sign_consensus + 0.5 * cv_consensus
        return np.clip(consensus, 0.0, 1.0)

    def migration_noise_weights(
        self,
        island_best_thetas: List[Optional[np.ndarray]],
        island_best_rewards: List[float],
    ) -> np.ndarray:
        """
        Returns per-dimension noise weight for migration injection.
        High consensus dimensions: low noise (inject global best's value reliably).
        Low consensus dimensions: high noise (don't force island-specific features).
        """
        cons = self.consensus_dimensions(island_best_thetas, island_best_rewards)
        # Invert: low consensus → high noise during migration
        return 2.0 - cons  # range [1.0, 2.0]


def patch_swarm_jepa_agent(SwarmJEPAAgent_cls):
    """
    Monkey-patches SwarmJEPAAgent with K, K2, M mechanisms.
    Call once after importing SwarmJEPAAgent. Idempotent: calling multiple
    times (e.g. once per seed) is safe and will not double-patch or recurse.

    Usage:
        from swarm_jepa_agent import SwarmJEPAAgent
        from karma_jepa import patch_swarm_jepa_agent
        patch_swarm_jepa_agent(SwarmJEPAAgent)
    """
    import logging
    log = logging.getLogger(__name__)

    if getattr(SwarmJEPAAgent_cls, "_karma_patched", False):
        log.debug("KarmaJEPA already patched on this class, skipping")
        return SwarmJEPAAgent_cls

    original_init = SwarmJEPAAgent_cls.__init__
    original_add_to_island = SwarmJEPAAgent_cls.add_to_island
    original_propose = SwarmJEPAAgent_cls.propose_for_island
    original_migrate = SwarmJEPAAgent_cls.maybe_migrate

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Add karmic trackers per island
        self._karma_trackers = [
            KarmicLineageTracker(self.param_dim)
            for _ in range(self.K)
        ]
        self._consensus_tracker = ConsensusTracker(self.param_dim, self.K)
        log.info(f"KarmaJEPA: K/K2/M mechanisms activated on {self.K} islands")

    def new_add_to_island(self, island_id: int, theta, reward: float):
        # Call the TRUE original (captured before patching), not the class
        # attribute, which would now point to this very function and recurse
        original_add_to_island(self, island_id, theta, reward)
        # Update karmic tracker for this island
        self._karma_trackers[island_id].add(theta, reward)

    def new_propose(self, island_id: int, t: int, total_T: int):
        """K + K2: use anisotropic karmic noise with adaptive std."""
        tracker = self._karma_trackers[island_id]

        # K2: adaptive noise_std from karmic signal strength
        karma_std = tracker.karma_noise_std(
            self.noise_std,
            noise_std_min=self.noise_std * 0.3,
            noise_std_max=self.noise_std * 2.0,
        )

        # Temporarily override noise_std for this proposal
        original_noise_std = self.noise_std
        self.noise_std = karma_std

        # Call original propose (which uses self.noise_std)
        theta = original_propose(self, island_id, t, total_T)

        # Restore
        self.noise_std = original_noise_std

        # K: apply anisotropic perturbation along high-tendency dimensions
        # This refines the proposal by adding dimension-specific noise
        if len(tracker._history) >= 5:
            dim_stds = tracker.karma_noise_covariance(karma_std * 0.3)
            karma_perturbation = np.random.randn(self.param_dim) * dim_stds
            theta = theta + karma_perturbation

        return theta

    def new_migrate(self, t: int) -> bool:
        """M: consensus-aware migration — noise weights based on cross-island agreement."""
        if t == 0 or t % self.migration_every != 0:
            return False
        if self._global_best_theta is None:
            return False

        # Get per-dimension consensus weights
        island_thetas = [isl.personal_best_theta for isl in self.islands]
        island_rewards = [isl.personal_best_reward for isl in self.islands]
        noise_weights = self._consensus_tracker.migration_noise_weights(
            island_thetas, island_rewards
        )

        log.info(f"[t={t}] Karma-Migration: broadcasting global best "
                 f"(r={self._global_best_reward:.1f}) with consensus weights "
                 f"mean={noise_weights.mean():.2f}")

        for k, island in enumerate(self.islands):
            if not island.in_isolation:
                # Consensus-aware injection: low noise on consensus dims,
                # high noise on island-specific dims
                consensus_noise = np.random.randn(self.param_dim) * (
                    self.noise_std * 0.1 * noise_weights
                )
                migrated_theta = self._global_best_theta + consensus_noise
                self.add_to_island(k, migrated_theta, self._global_best_reward)
        return True

    # Apply patches
    SwarmJEPAAgent_cls.__init__ = new_init
    SwarmJEPAAgent_cls.add_to_island = new_add_to_island
    SwarmJEPAAgent_cls.propose_for_island = new_propose
    SwarmJEPAAgent_cls.maybe_migrate = new_migrate
    SwarmJEPAAgent_cls._karma_patched = True

    log.info("SwarmJEPAAgent patched with KarmaJEPA (K + K2 + M)")
    return SwarmJEPAAgent_cls
