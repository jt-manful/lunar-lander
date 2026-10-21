
from collections import defaultdict

import gymnasium as gym
import numpy as np


class LunarLanderAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.99,
        n_bins: int = 8,
    ):
        self.env = env

        # Number of discrete bins for each observation dimension.
        self.n_bins = n_bins

        # Approximate bounds for LunarLander observations.
        # Clip extreme values before discretizing.
        self.low = np.array([
            -1.5, -1.5, -5.0, -5.0,
            -np.pi, -5.0, 0.0, 0.0
        ])

        self.high = np.array([
            1.5, 1.5, 5.0, 5.0,
            np.pi, 5.0, 1.0, 1.0
        ])

        # One bin per interval, for each observation dimension.
        self.q_values = defaultdict(
            lambda: np.zeros(env.action_space.n)
        )

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def discretize_state(self, obs):
        """Convert a continuous observation into a hashable discrete state."""
        obs = np.asarray(obs, dtype=np.float64)

        # Keep values inside the selected bounds.
        obs = np.clip(obs, self.low, self.high)

        # Normalize each dimension to [0, 1].
        normalized = (obs - self.low) / (self.high - self.low)

        # Convert to integer bin indices.
        state = np.floor(normalized * self.n_bins).astype(int)

        # The upper boundary must not create an extra bin.
        state = np.clip(state, 0, self.n_bins - 1)

        return tuple(state)

    def get_action(self, obs) -> int:
        state = self.discretize_state(obs)

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        return int(np.argmax(self.q_values[state]))

    def update(self, obs, action, reward, terminated, next_obs):
        state = self.discretize_state(obs)
        next_state = self.discretize_state(next_obs)

        # No future reward after a true terminal state.
        if terminated:
            future_q_value = 0.0
        else:
            future_q_value = np.max(self.q_values[next_state])

        target = reward + self.discount_factor * future_q_value

        temporal_difference = target - self.q_values[state][action]

        self.q_values[state][action] += self.lr * temporal_difference

        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(
            self.final_epsilon,
            self.epsilon - self.epsilon_decay,
        )
