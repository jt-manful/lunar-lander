import gymnasium as gym


def main() -> None:
    env = gym.make("LunarLander-v3")
    try:
        env.reset(seed=42)
        env.step(env.action_space.sample())
    finally:
        env.close()
    print("LunarLander-v3 initialized and stepped successfully.")


if __name__ == "__main__":
    main()