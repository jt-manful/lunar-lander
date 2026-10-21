from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env


OUTPUT_DIR = Path("outputs/ppo")
LOG_DIR = OUTPUT_DIR / "logs"
BEST_DIR = OUTPUT_DIR / "checkpoints" / "best_ppo_model"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    BEST_DIR.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_env(
        "LunarLander-v3",
        n_envs=1,
        seed=42,
    )
    eval_env = gym.make("LunarLander-v3")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(BEST_DIR),
        log_path=str(LOG_DIR),
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        seed=42,
        device="cpu",
        tensorboard_log=str(LOG_DIR),
    )

    try:
        model.learn(
            total_timesteps=500_000,
            callback=eval_callback,
            progress_bar=True,
        )
        model.save(str(OUTPUT_DIR / "ppo_lunarlander"))
        print("Training complete. Final model saved.")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
