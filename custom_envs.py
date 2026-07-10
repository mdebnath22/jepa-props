"""
custom_envs.py — Original implementations of Maze, Nim, and Navigation,
functionally matching the task structure ProPS describes (grid maze with
walls/holes, combinatorial Nim game, continuous 2D navigation with
obstacles), since the original repos for these are not public.

These are NOT attempts to reproduce ProPS's exact reward numbers (the
original environments' precise layouts, reward shaping, and difficulty are
unpublished) — they're original environments built to the same task
DESCRIPTION (grid maze with optimal-path solving as shown in ProPS Fig. 5a;
combinatorial Nim with win/lose terminal reward; continuous navigation with
obstacle avoidance, matching the reward scale seen in ProPS Table 1,
roughly O(1000-4000)).

Each environment exposes a gymnasium.Env-compatible interface (reset, step)
so they slot directly into envs.py's rollout_policy without special-casing.
"""

import numpy as np


class MazeEnv:
    """
    Grid maze with walls, holes (terminal failure), and a goal flag.
    Matches the visual structure in ProPS Figure 5a: a 4x4 grid with
    holes (failure states) and a flag (goal state), agent moves in
    4 cardinal directions, deterministic transitions.

    Observation: integer cell index (0..15), flattened row-major.
    Action: 0=up, 1=right, 2=down, 3=left.
    Reward: +1.0 on reaching goal, -1.0 on falling in a hole, -0.01 per
            step otherwise (small step penalty encourages shortest path).
    Episode ends on goal, hole, or max_steps.
    """

    metadata = {}

    def __init__(self, size=4, hole_positions=None, seed=None):
        self.size = size
        self.n_states = size * size
        self.observation_space = _DiscreteSpace(self.n_states)
        self.action_space = _DiscreteSpace(4)

        # Default hole layout roughly matching ProPS Fig 5a's visual density
        if hole_positions is None:
            hole_positions = {5, 6, 9, 10, 13}
        self.hole_positions = set(hole_positions)
        self.goal_position = self.n_states - 1
        self.start_position = 0
        self.max_steps = 100

        self._pos = self.start_position
        self._steps = 0
        self._rng = np.random.RandomState(seed)

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._pos = self.start_position
        self._steps = 0
        return self._pos, {}

    def step(self, action):
        row, col = divmod(self._pos, self.size)
        if action == 0 and row > 0:
            row -= 1
        elif action == 1 and col < self.size - 1:
            col += 1
        elif action == 2 and row < self.size - 1:
            row += 1
        elif action == 3 and col > 0:
            col -= 1
        self._pos = row * self.size + col
        self._steps += 1

        terminated = False
        reward = -0.01
        if self._pos in self.hole_positions:
            reward = -1.0
            terminated = True
        elif self._pos == self.goal_position:
            reward = 1.0
            terminated = True

        truncated = self._steps >= self.max_steps
        return self._pos, reward, terminated, truncated, {}

    def close(self):
        pass


class NimEnv:
    """
    Combinatorial Nim game vs. a fixed heuristic opponent. Matches the
    reward scale in ProPS Table 1 (roughly [-1, 1] range): +1 for winning
    the game (taking the last object), -1 for losing.

    State: tuple of pile sizes, flattened to a fixed-size observation
    vector (padded/truncated to n_piles).
    Action: encodes (pile_index, n_to_remove) as a single integer:
            action = pile_index * max_remove + (n_to_remove - 1)
    Opponent: plays optimally with probability opponent_skill, else
    randomly — gives a tunable difficulty knob.
    """

    metadata = {}

    def __init__(self, n_piles=3, max_pile_size=7, opponent_skill=0.7, seed=None):
        self.n_piles = n_piles
        self.max_pile_size = max_pile_size
        self.max_remove = max_pile_size
        self.opponent_skill = opponent_skill
        self.observation_space = _BoxSpace(low=0, high=max_pile_size, shape=(n_piles,))
        self.action_space = _DiscreteSpace(n_piles * self.max_remove)
        self.max_steps = 50

        self._piles = None
        self._steps = 0
        self._rng = np.random.RandomState(seed)

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._piles = self._rng.randint(1, self.max_pile_size + 1, size=self.n_piles).astype(float)
        self._steps = 0
        return self._piles.copy(), {}

    def _decode_action(self, action):
        action = int(action) % (self.n_piles * self.max_remove)
        pile_idx = action // self.max_remove
        n_remove = (action % self.max_remove) + 1
        return pile_idx, n_remove

    def _opponent_move(self):
        """Nim-sum optimal play with probability opponent_skill, else random."""
        if self._rng.rand() < self.opponent_skill:
            nim_sum = 0
            for p in self._piles:
                nim_sum ^= int(p)
            for i, p in enumerate(self._piles):
                target = nim_sum ^ int(p)
                if target < int(p):
                    self._piles[i] = target
                    return
        # fallback: random legal move
        nonzero = [i for i, p in enumerate(self._piles) if p > 0]
        if nonzero:
            i = self._rng.choice(nonzero)
            self._piles[i] -= self._rng.randint(1, int(self._piles[i]) + 1)

    def step(self, action):
        pile_idx, n_remove = self._decode_action(action)
        self._steps += 1

        # Clip to a legal move if the proposed action is illegal
        pile_idx = pile_idx % self.n_piles
        n_remove = min(n_remove, max(1, int(self._piles[pile_idx])))
        self._piles[pile_idx] = max(0, self._piles[pile_idx] - n_remove)

        if self._piles.sum() <= 0:
            return self._piles.copy(), 1.0, True, False, {}  # agent took last object: WIN

        self._opponent_move()
        if self._piles.sum() <= 0:
            return self._piles.copy(), -1.0, True, False, {}  # opponent took last: LOSE

        truncated = self._steps >= self.max_steps
        return self._piles.copy(), 0.0, False, truncated, {}

    def close(self):
        pass


class NavigationEnv:
    """
    Continuous 2D navigation with circular obstacles, matching ProPS
    Table 1's reward scale (roughly 1000-4000, suggesting a dense
    per-step shaping reward plus a large terminal bonus).

    Observation: [agent_x, agent_y, agent_vx, agent_vy, goal_x, goal_y]
    Action: continuous [ax, ay] acceleration, clipped to [-1, 1]^2.
    Reward: dense shaping = -distance_to_goal - obstacle_penalty,
            +500 terminal bonus on reaching goal.
    """

    metadata = {}

    def __init__(self, n_obstacles=4, world_size=10.0, seed=None):
        self.world_size = world_size
        self.n_obstacles = n_obstacles
        self.observation_space = _BoxSpace(low=-world_size, high=world_size, shape=(6,))
        self.action_space = _BoxSpace(low=-1.0, high=1.0, shape=(2,))
        self.max_steps = 200
        self.dt = 0.1
        self.goal_radius = 0.5
        self.obstacle_radius = 0.8

        self._rng = np.random.RandomState(seed)
        self._pos = None
        self._vel = None
        self._goal = None
        self._obstacles = None
        self._steps = 0

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._pos = np.array([0.0, 0.0])
        self._vel = np.array([0.0, 0.0])
        self._goal = self._rng.uniform(-self.world_size * 0.8, self.world_size * 0.8, size=2)
        self._obstacles = [
            self._rng.uniform(-self.world_size * 0.6, self.world_size * 0.6, size=2)
            for _ in range(self.n_obstacles)
        ]
        self._steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self._pos, self._vel, self._goal])

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self._vel = np.clip(self._vel + action * self.dt, -2.0, 2.0)
        self._pos = np.clip(self._pos + self._vel * self.dt,
                            -self.world_size, self.world_size)
        self._steps += 1

        dist_to_goal = np.linalg.norm(self._pos - self._goal)
        reward = -dist_to_goal * 0.1

        obstacle_penalty = 0.0
        for obs in self._obstacles:
            d = np.linalg.norm(self._pos - obs)
            if d < self.obstacle_radius:
                obstacle_penalty += (self.obstacle_radius - d) * 5.0
        reward -= obstacle_penalty

        terminated = False
        if dist_to_goal < self.goal_radius:
            reward += 500.0
            terminated = True

        truncated = self._steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, {}

    def close(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Minimal gymnasium-compatible space stand-ins (avoid a hard gymnasium
# dependency for these custom envs specifically)
# ─────────────────────────────────────────────────────────────────────────────

class _DiscreteSpace:
    def __init__(self, n):
        self.n = n


class _BoxSpace:
    def __init__(self, low, high, shape):
        self.low = low
        self.high = high
        self.shape = shape


CUSTOM_ENV_REGISTRY = {
    "Maze": MazeEnv,
    "Nim": NimEnv,
    "Navigation": NavigationEnv,
}


class BallLandingEnv:
    """
    Ball landing task (ProPS Table 10): a projectile is launched with an
    initial velocity determined by the policy parameters, and must land
    as close as possible to a fixed target on the ground.

    This matches the task description from ProPS Table 10: "distance from
    goal (m)" as the metric, with ProPS achieving 0.07m vs OpenAI-ES 0.40m.

    Physics: 2D ballistic motion with air resistance. The policy parameters
    control [launch_angle, launch_speed, lateral_offset] (param_dim=3).
    Reward: -distance_from_goal (negative, so maximizing = minimizing distance).

    The goal is fixed at a random position each reset (seeded), and the
    agent launches from the origin.
    """

    metadata = {}

    def __init__(self, g=9.81, drag=0.01, seed=None):
        self.g = g
        self.drag = drag
        self.observation_space = _BoxSpace(low=-100, high=100, shape=(3,))
        self.action_space = _BoxSpace(low=-1.0, high=1.0, shape=(3,))
        self.max_episode_steps = 1
        self._rng = np.random.RandomState(seed)
        self._goal = None

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        # Goal: random position 10-50m away in x, at ground level (z=0)
        dist = self._rng.uniform(10, 50)
        angle = self._rng.uniform(-np.pi / 4, np.pi / 4)
        self._goal = np.array([dist * np.cos(angle), dist * np.sin(angle)])
        return self._goal.copy(), {}

    def step(self, action):
        """
        action: [launch_angle (rad), launch_speed (m/s), lateral_angle (rad)]
        Simulates ballistic trajectory and returns negative landing distance.
        """
        launch_angle = float(np.clip(action[0], -1, 1)) * np.pi / 2
        launch_speed = float(np.clip(action[1], -1, 1)) * 30 + 30  # 0 to 60 m/s
        lateral_angle = float(np.clip(action[2], -1, 1)) * np.pi / 4

        # 3D velocity components
        vx = launch_speed * np.cos(launch_angle) * np.cos(lateral_angle)
        vy = launch_speed * np.cos(launch_angle) * np.sin(lateral_angle)
        vz = launch_speed * np.sin(launch_angle)

        # Simulate until ball hits ground (z <= 0)
        x, y, z = 0.0, 0.0, 0.0
        dt = 0.02
        for _ in range(5000):
            speed = np.sqrt(vx**2 + vy**2 + vz**2)
            ax = -self.drag * vx * speed
            ay = -self.drag * vy * speed
            az = -self.g - self.drag * vz * speed
            vx += ax * dt
            vy += ay * dt
            vz += az * dt
            x += vx * dt
            y += vy * dt
            z += vz * dt
            if z <= 0:
                break

        landing = np.array([x, y])
        dist = float(np.linalg.norm(landing - self._goal))
        reward = -dist  # maximize => minimize distance

        return self._goal.copy(), reward, True, False, {"landing": landing, "dist": dist}

    def close(self):
        pass


# Register the ball landing env
CUSTOM_ENV_REGISTRY["BallLanding"] = BallLandingEnv
