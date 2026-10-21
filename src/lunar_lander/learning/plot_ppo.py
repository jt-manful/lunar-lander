
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("outputs/ppo")
EVAL_PATH = OUTPUT_DIR / "logs" / "evaluations.npz"
OUTPUT_PATH = OUTPUT_DIR / "ppo_learning_curve.png"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not EVAL_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation data not found: {EVAL_PATH}\n"
            "Run ppo_train.py first."
        )

    data = np.load(EVAL_PATH)

    timesteps = data["timesteps"]
    rewards = data["results"]

    # One row per evaluation checkpoint.
    mean_rewards = rewards.mean(axis=1)
    std_rewards = rewards.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        timesteps,
        mean_rewards,
        color="royalblue",
        label="Mean evaluation reward",
    )

    ax.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        color="royalblue",
        alpha=0.2,
        label="±1 standard deviation",
    )

    ax.set_title("PPO LunarLander Learning Curve")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean episode reward")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)

    print(f"Plot saved to: {OUTPUT_PATH}")

    # Display the graph interactively as well.
    plt.show()


if __name__ == "__main__":
    main()
