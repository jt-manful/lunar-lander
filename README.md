# Lunar Lander: Reward Engineering for Reinforcement Learning

![2D Lunar Lander](assets/lander.png)

## Overview

This project explores reinforcement learning for Gymnasium's `LunarLander-v3`, with a focus on how reward design affects sample efficiency, stability, and landing performance. It currently includes a tabular Q-learning baseline and a PPO implementation. Reward-shaping comparisons and multi-seed experiments are planned.

## Setup

Requirements: macOS or Linux, Git, Python 3.12, and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/jt-manful/lunar-lander.git
cd lunar-lander
uv sync
uv run lunar-lander check-env
```

If Box2D needs to be compiled on macOS, install SWIG and sync again:

```sh
brew install swig
uv sync
```

## Run

Train, evaluate, and plot Q-learning:

```sh
uv run lunar-lander train-q-learning
uv run lunar-lander evaluate-q-learning --episodes 1000
uv run lunar-lander plot-q-learning
```

Train, evaluate, and plot PPO:

```sh
uv run lunar-lander train-ppo
uv run lunar-lander evaluate-ppo --episodes 100
uv run lunar-lander plot-ppo
```

Add `--render` to either evaluation command to display the environment. Use `uv run lunar-lander --help` or `uv run lunar-lander <command> --help` to see available options.

## Project layout

```text
docs/                 Project notes and command reference
assets/               Images used by the documentation
src/lunar_lander/     Installable package and experiment code
tests/                Automated tests
outputs/              Generated models, metrics, plots, and logs (ignored by Git)
```

Generated artifacts are grouped under `outputs/q_learning/` and `outputs/ppo/`. These are ignored by Git; retain any results you want to publish separately or commit only intentionally selected artifacts.

## Evaluation focus

The planned comparisons include learning curves, landing success rate, crash frequency, touchdown conditions, and control efficiency. Results should eventually be evaluated across multiple random seeds and consistent training settings.

For the project learning notes, see [the learning log](docs/learning-log.md). For all runnable commands, see [the command reference](docs/commands.md).

## Resources

- [Gymnasium LunarLander documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [Stable-Baselines3 documentation](https://stable-baselines3.readthedocs.io/)
- [uv documentation](https://docs.astral.sh/uv/)
