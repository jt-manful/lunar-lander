
import csv
import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from lunar_lander.agents.qlearner import LunarLanderAgent


OUTPUT_DIR = Path("outputs/q_learning")

MODEL_PATH = OUTPUT_DIR / "q_table.pkl"
STATS_PATH = OUTPUT_DIR / "training_stats.csv"

N_EPISODES = 1000
LEARNING_RATE = 0.01
START_EPSILON = 1.0
FINAL_EPSILON = 0.1
DISCOUNT_FACTOR = 0.99


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Train without rendering to improve speed.
    env = gym.make("LunarLander-v3")

    agent = LunarLanderAgent(
        env=env,
        learning_rate=LEARNING_RATE,
        initial_epsilon=START_EPSILON,
        epsilon_decay=(START_EPSILON - FINAL_EPSILON)
        / (N_EPISODES * 0.8),
        final_epsilon=FINAL_EPSILON,
        discount_factor=DISCOUNT_FACTOR,
        n_bins=8,
    )

    episode_stats = []

    try:
        for episode in tqdm(range(N_EPISODES), desc="Training"):
            obs, _ = env.reset()
            total_reward = 0.0
            steps = 0
            done = False

            # Keep only this episode's TD errors.
            error_start = len(agent.training_error)

            while not done:
                action = agent.get_action(obs)

                next_obs, reward, terminated, truncated, _ = (
                    env.step(action)
                )

                agent.update(
                    obs,
                    action,
                    reward,
                    terminated,
                    next_obs,
                )

                obs = next_obs
                total_reward += reward
                steps += 1
                done = terminated or truncated

            errors = agent.training_error[error_start:]
            mean_abs_td_error = (
                float(np.mean(np.abs(errors))) if errors else 0.0
            )

            # Prevent the per-step error list from growing indefinitely.
            del agent.training_error[error_start:]

            agent.decay_epsilon()

            episode_stats.append({
                "episode": episode + 1,
                "reward": total_reward,
                "length": steps,
                "epsilon": agent.epsilon,
                "mean_abs_td_error": mean_abs_td_error,
            })

            if (episode + 1) % 1000 == 0:
                recent = [
                    row["reward"] for row in episode_stats[-1000:]
                ]
                tqdm.write(
                    f"Episode {episode + 1:,} | "
                    f"Mean reward (last 1000): "
                    f"{np.mean(recent):.2f} | "
                    f"Epsilon: {agent.epsilon:.3f}"
                )

        # Save the Q-table and settings needed to recreate the agent.
        with MODEL_PATH.open("wb") as file:
            pickle.dump(
                {
                    "q_values": dict(agent.q_values),
                    "n_bins": agent.n_bins,
                    "learning_rate": LEARNING_RATE,
                    "discount_factor": DISCOUNT_FACTOR,
                    "final_epsilon": FINAL_EPSILON,
                },
                file,
            )

        # Save training metrics.
        with STATS_PATH.open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=episode_stats[0].keys(),
            )
            writer.writeheader()
            writer.writerows(episode_stats)

        print(f"\nModel saved to: {MODEL_PATH}")
        print(f"Training statistics saved to: {STATS_PATH}")
        print(
            "Final 100-episode mean reward:",
            round(
                np.mean([row["reward"] for row in episode_stats[-100:]]),
                2,
            ),
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
