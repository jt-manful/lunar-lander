# Commands

Run these commands from the repository root.

```sh
uv sync
uv run lunar-lander check-env
```

## Q-learning

```sh
uv run lunar-lander train-q-learning
uv run lunar-lander plot-q-learning
uv run lunar-lander evaluate-q-learning --episodes 1000
uv run lunar-lander evaluate-q-learning --episodes 5 --render
```

## PPO

```sh
uv run lunar-lander train-ppo
uv run lunar-lander plot-ppo
uv run lunar-lander evaluate-ppo --episodes 100
uv run lunar-lander evaluate-ppo --episodes 5 --render
uv run lunar-lander record-landing --model outputs/ppo/ppo_lunarlander.zip
```

Training models, metrics, and logs are written beneath `outputs/`. Recorded
videos are saved to `outputs/ppo/videos/` by default; use `--output` to choose
a different directory.
