
import argparse
import csv
import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np

from lunar_lander.agents.qlearner import LunarLanderAgent


OUTPUT_DIR = Path("outputs/q_learning")
MODEL_PATH = OUTPUT_DIR / "q_table.pkl"


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="lunar-lander evaluate-q-learning",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Show the lander visually during testing.",
    )
    args = parser.parse_args(argv)

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            "Run the training script first."
        )

    with MODEL_PATH.open("rb") as file:
        checkpoint = pickle.load(file)

    env = gym.make(
        "LunarLander-v3",
        render_mode="human" if args.render else None,
    )

    agent = LunarLanderAgent(
        env=env,
        learning_rate=checkpoint["learning_rate"],
        initial_epsilon=0.0,
        epsilon_decay=0.0,
        final_epsilon=0.0,
        discount_factor=checkpoint["discount_factor"],
        n_bins=checkpoint["n_bins"],
    )

    # Restore learned Q-values while preserving the defaultdict.
    agent.q_values.update(checkpoint["q_values"])
    agent.epsilon = 0.0

    episode_results = []

    try:
        for episode in range(args.episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = agent.get_action(obs)

                obs, reward, terminated, truncated, _ = (
                    env.step(action)
                )

                total_reward += reward
                steps += 1

            episode_results.append({
                "episode": episode + 1,
                "reward": total_reward,
                "length": steps,
                "terminated": terminated,
                "truncated": truncated,
                "success": total_reward > 0,
            })

    finally:
        env.close()

    rewards = np.array(
        [row["reward"] for row in episode_results]
    )
    lengths = np.array(
        [row["length"] for row in episode_results]
    )
    successes = np.array(
        [row["success"] for row in episode_results]
    )

    summary = {
        "episodes": args.episodes,
        "win_rate": float(np.mean(successes)),
        "average_reward": float(np.mean(rewards)),
        "median_reward": float(np.median(rewards)),
        "reward_std": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "average_length": float(np.mean(lengths)),
    }

    episodes_path = OUTPUT_DIR / "test_episodes.csv"
    summary_path = OUTPUT_DIR / "test_stats.csv"

    with episodes_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=episode_results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(episode_results)

    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summary.keys(),
        )
        writer.writeheader()
        writer.writerow(summary)

    print(f"\nTest results over {args.episodes} episodes")
    print(f"Win rate: {summary['win_rate']:.1%}")
    print(f"Average reward: {summary['average_reward']:.3f}")
    print(f"Median reward: {summary['median_reward']:.3f}")
    print(f"Reward standard deviation: {summary['reward_std']:.3f}")
    print(f"Average episode length: {summary['average_length']:.1f}")
    print(f"Best episode reward: {summary['max_reward']:.3f}")
    print(f"Worst episode reward: {summary['min_reward']:.3f}")
    print(f"\nSummary saved to: {summary_path}")
    print(f"Episode results saved to: {episodes_path}")


if __name__ == "__main__":
    main()
