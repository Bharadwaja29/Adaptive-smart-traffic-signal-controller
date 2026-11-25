import numpy as np
from traffic_env import SmartTrafficEnv


class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.30,
        epsilon_decay: float = 0.996,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        init_q = -10.0 / (1.0 - gamma)
        self.q_table = np.full((n_states, n_actions), init_q, dtype=np.float64)
        self.visits = np.zeros((n_states, n_actions), dtype=np.int64)

    def choose_action(self, state: int, explore: bool = True) -> int:
        if explore and np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int):
        self.visits[state, action] += 1
        # Fast initial updates erase 0-init bias, then decay to smooth out noise
        step_alpha = max(
            self.alpha,
            1.0 / (1.0 + 0.25 * self.visits[state, action]),
        )
        best_next_q = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += step_alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class FixedTimerAgent:
    """Traditional traffic light that switches every N steps regardless of traffic."""

    def __init__(self, switch_every: int = 3):
        self.switch_every = switch_every

    def choose_action(self, step_idx: int) -> int:
        return 1 if (step_idx > 0 and step_idx % self.switch_every == 0) else 0


def train_q_agent(
    lambda_ns: float,
    lambda_ew: float,
    episodes: int = 8000,
    alpha: float = 0.1,
    gamma: float = 0.95,
    switch_penalty: float = 1.5,
):
    np.random.seed(0)  # Locks training so heatmap stays consistent every run
    env = SmartTrafficEnv(
        lambda_ns=lambda_ns,
        lambda_ew=lambda_ew,
        max_steps=20,
        switch_penalty=switch_penalty,
    )
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        epsilon_min=0.05,
    )

    rewards_history = []
    for ep in range(episodes):
        state, _ = env.reset(seed=ep, options={"random_start": True})
        total_reward = 0.0
        done = False
        step_num = 0

        while not done:
            # True Exploring Starts: test both actions equally from the random start state
            if step_num == 0:
                action = int(np.random.randint(agent.n_actions))
            else:
                action = agent.choose_action(state, explore=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            step_num += 1
            done = terminated or truncated

        agent.epsilon = max(agent.epsilon_min, 1.0 - (ep / (0.8 * episodes)))
        rewards_history.append(total_reward)

    return agent, rewards_history


def evaluate_controllers(
    q_agent: QLearningAgent,
    lambda_ns: float,
    lambda_ew: float,
    switch_penalty: float = 1.5,
    fixed_cycle: int = 3,
    eval_seed: int = 999,
):
    """Runs both controllers on the EXACT same seeded Poisson arrival stream."""
    env_fixed = SmartTrafficEnv(
        lambda_ns=lambda_ns, lambda_ew=lambda_ew, max_steps=60, switch_penalty=switch_penalty
    )
    env_rl = SmartTrafficEnv(
        lambda_ns=lambda_ns, lambda_ew=lambda_ew, max_steps=60, switch_penalty=switch_penalty
    )

    fixed_agent = FixedTimerAgent(switch_every=fixed_cycle)

    # Run Fixed-Timer Baseline
    _, info_f = env_fixed.reset(seed=eval_seed)
    fixed_trace = [info_f]
    done = False
    step_idx = 0
    while not done:
        act = fixed_agent.choose_action(step_idx)
        _, _, term, trunc, info_f = env_fixed.step(act)
        fixed_trace.append(info_f)
        step_idx += 1
        done = term or trunc

    # Run Trained Q-Learning Agent (greedy evaluation, explore=False)
    state, info_r = env_rl.reset(seed=eval_seed)
    rl_trace = [info_r]
    done = False
    while not done:
        act = q_agent.choose_action(state, explore=False)
        state, _, term, trunc, info_r = env_rl.step(act)
        rl_trace.append(info_r)
        done = term or trunc

    return fixed_trace, rl_trace