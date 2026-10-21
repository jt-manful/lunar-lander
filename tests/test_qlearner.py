from types import SimpleNamespace

import numpy as np

from lunar_lander.agents.qlearner import LunarLanderAgent


def make_agent() -> LunarLanderAgent:
    env = SimpleNamespace(action_space=SimpleNamespace(n=4))
    return LunarLanderAgent(
        env=env,
        learning_rate=0.1,
        initial_epsilon=1.0,
        epsilon_decay=0.1,
        final_epsilon=0.1,
    )


def test_discretize_state_returns_one_valid_bin_per_observation() -> None:
    agent = make_agent()

    state = agent.discretize_state(np.zeros(8))

    assert len(state) == 8
    assert all(0 <= index < agent.n_bins for index in state)


def test_discretize_state_clips_observations_to_bounds() -> None:
    agent = make_agent()

    state = agent.discretize_state(agent.high * 10)

    assert state == (agent.n_bins - 1,) * 8
