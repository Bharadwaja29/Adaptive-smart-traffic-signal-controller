import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SmartTrafficEnv(gym.Env):
    """
    Custom 4-way Intersection Environment (2 competing axes: NS vs EW).
    State  : Discrete index encoding (min(q_ns, 10), min(q_ew, 10), phase)
    Action : 0 = Keep current green phase, 1 = Switch green phase
    Reward : Negative total waiting cars minus switching penalty
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        lambda_ns: float = 1.6,
        lambda_ew: float = 0.4,
        max_steps: int = 60,
        max_clear_rate: int = 3,
        switch_penalty: float = 1.5,
    ):
        super().__init__()
        self.lambda_ns = lambda_ns
        self.lambda_ew = lambda_ew
        self.max_steps = max_steps
        self.max_clear_rate = max_clear_rate
        self.switch_penalty = switch_penalty

        self.max_state_q = 10   # Queue cap for compact Q-table (0..10)
        self.max_road_cap = 25  # Physical road capacity per axis

        # 11 (NS) * 11 (EW) * 2 (Phase) = 242 discrete states
        self.observation_space = spaces.Discrete(
            (self.max_state_q + 1) * (self.max_state_q + 1) * 2
        )
        # 0 = Keep current phase, 1 = Switch phase
        self.action_space = spaces.Discrete(2)

        self.q_ns = 0
        self.q_ew = 0
        self.phase = 0  # 0 = NS Green, 1 = EW Green
        self.current_step = 0

    def _encode_state(self) -> int:
        ns_bin = min(self.q_ns, self.max_state_q)
        ew_bin = min(self.q_ew, self.max_state_q)
        return (ns_bin * (self.max_state_q + 1) + ew_bin) * 2 + self.phase

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and options.get("random_start", False):
            self.q_ns = int(self.np_random.integers(0, self.max_state_q + 1))
            self.q_ew = int(self.np_random.integers(0, self.max_state_q + 1))
            self.phase = int(self.np_random.integers(0, 2))
        else:
            self.q_ns = 2
            self.q_ew = 2
            self.phase = 0
        self.current_step = 0
        return self._encode_state(), self._get_info()

    def _get_info(self) -> dict:
        return {
            "q_ns": int(self.q_ns),
            "q_ew": int(self.q_ew),
            "phase": int(self.phase),
            "total_queue": int(self.q_ns + self.q_ew),
            "step": int(self.current_step),
        }

    def step(self, action: int):
        switched = (action == 1)
        if switched:
            self.phase = 1 - self.phase

        # Punish wasting a green light on an empty lane while the other lane waits
        wasted_green = (self.phase == 0 and self.q_ns == 0 and self.q_ew > 0) or (
            self.phase == 1 and self.q_ew == 0 and self.q_ns > 0
        )
        idle_penalty = 8.0 if wasted_green else 0.0

        # Yellow-light clearance delay reduces throughput by 1 car on switch
        effective_clear = self.max_clear_rate - (1 if switched else 0)

        # Clear vehicles from whichever axis currently has the green light
        if self.phase == 0:
            cleared = min(self.q_ns, effective_clear)
            self.q_ns -= cleared
        else:
            cleared = min(self.q_ew, effective_clear)
            self.q_ew -= cleared

        # Stochastic vehicle arrivals using Poisson process
        arrivals_ns = int(self.np_random.poisson(self.lambda_ns))
        arrivals_ew = int(self.np_random.poisson(self.lambda_ew))

        self.q_ns = min(self.max_road_cap, self.q_ns + arrivals_ns)
        self.q_ew = min(self.max_road_cap, self.q_ew + arrivals_ew)

        # Linear queue reward + switching cost + empty-green penalty
        reward = (
            -float(self.q_ns + self.q_ew)
            - (self.switch_penalty if switched else 0.0)
            - idle_penalty
        )

        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps

        return self._encode_state(), reward, terminated, truncated, self._get_info()