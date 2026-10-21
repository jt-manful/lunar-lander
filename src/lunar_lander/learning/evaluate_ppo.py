
import argparse
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

OUTPUT_DIR = Path("outputs/ppo")
MODEL_PATH = OUTPUT_DIR / "ppo_lunarlander.zip"

N_EPISODES = 100


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="lunar-lander evaluate-ppo")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Display the LunarLander window",
    )
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "LunarLander-v3",
        render_mode="human" if args.render else None,
    )

    model = PPO.load(str(MODEL_PATH))

    results = []

    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=1000 + episode)
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                obs, reward, terminated, truncated, _ = (
                    env.step(action)
                )

                total_reward += reward
                steps += 1

            results.append({
                "episode": episode + 1,
                "reward": total_reward,
                "length": steps,
                "terminated": terminated,
                "truncated": truncated,
                "success": total_reward > 0,
            })

    finally:
        env.close()

    rewards = np.array([r["reward"] for r in results])
    successes = np.array([r["success"] for r in results])

    print(f"Evaluation episodes: {args.episodes}")
    print(f"Mean reward: {rewards.mean():.2f}")
    print(f"Median reward: {np.median(rewards):.2f}")
    print(f"Reward standard deviation: {rewards.std():.2f}")
    print(f"Positive-reward rate: {successes.mean():.1%}")

    output_path = OUTPUT_DIR / "evaluation_episodes.csv"

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Episode results saved to {output_path}")


if __name__ == "__main__":
    main()
