
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("outputs/q_learning")
STATS_PATH = OUTPUT_DIR / "training_stats.csv"
PLOT_PATH = OUTPUT_DIR / "training_plots.png"

WINDOW = 500


def moving_average(values, window):
    values = np.asarray(values, dtype=float)

    if len(values) < window:
        return values

    return np.convolve(
        values,
        np.ones(window) / window,
        mode="valid",
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with STATS_PATH.open(newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("Training statistics file is empty.")

    episodes = np.array(
        [int(row["episode"]) for row in rows]
    )
    rewards = np.array(
        [float(row["reward"]) for row in rows]
    )
    lengths = np.array(
        [int(row["length"]) for row in rows]
    )
    errors = np.array(
        [float(row["mean_abs_td_error"]) for row in rows]
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(
        episodes,
        rewards,
        alpha=0.2,
        label="Episode reward",
    )
    axes[0].plot(
        episodes[WINDOW - 1:],
        moving_average(rewards, WINDOW),
        color="red",
        label=f"{WINDOW}-episode average",
    )
    axes[0].set_title("Rewards during training")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend()

    axes[1].plot(
        episodes[WINDOW - 1:],
        moving_average(lengths, WINDOW),
    )
    axes[1].set_title("Episode length")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")

    axes[2].plot(
        episodes,
        errors,
        alpha=0.7,
    )
    axes[2].set_title("Mean absolute TD error")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Absolute TD error")

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)

    print(f"Training plots saved to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
