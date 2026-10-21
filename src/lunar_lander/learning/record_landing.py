
import argparse
from pathlib import Path
import re

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VIDEO_DIR = ROOT / "outputs" / "ppo" / "videos"


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Record video of a trained  LunarLander model."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the saved  model (.zip extension optional).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Video filename prefix (defaults to the model name).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to record (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Initial environment seed (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help="Directory for recordings.",
    )

    args = parser.parse_args(argv)

    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = ROOT / model_path

    # SB3 normally appends .zip when loading a model.
    if not model_path.exists() and model_path.suffix != ".zip":
        model_path = model_path.with_suffix(".zip")

    if not model_path.is_file():
        parser.error(f"Model file not found: {model_path}")

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    video_dir = args.output.expanduser()
    if not video_dir.is_absolute():
        video_dir = ROOT / video_dir

    video_dir.mkdir(parents=True, exist_ok=True)

    name = args.name or model_path.stem
    # Avoid invalid filename characters.
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")
    if not name:
        parser.error("--name must contain a valid filename character")

    model = PPO.load(str(model_path))

    env = gym.make(
        "LunarLander-v3",
        render_mode="rgb_array",
    )

    env = RecordVideo(
        env,
        video_folder=str(video_dir),
        episode_trigger=lambda episode_id: True,
        name_prefix=name,
    )

    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                obs, reward, terminated, truncated, info = (
                    env.step(action)
                )

                total_reward += reward
                steps += 1

            print(
                f"Episode {episode + 1}/{args.episodes} | "
                f"Reward: {total_reward:.2f} | "
                f"Steps: {steps}"
            )

    finally:
        env.close()

    print(f"\nRecordings saved to: {video_dir.resolve()}")


if __name__ == "__main__":
    main()
