import argparse
import sys


COMMANDS = (
    "check-env",
    "train-q-learning",
    "evaluate-q-learning",
    "plot-q-learning",
    "train-ppo",
    "evaluate-ppo",
    "plot-ppo",
    "record-landing",
)


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="lunar-lander",
        description="Train, evaluate, and record LunarLander agents.",
    )
    parser.add_argument("command", choices=COMMANDS)

    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return

    command, *remaining = arguments
    if command not in COMMANDS:
        parser.error(f"invalid choice: '{command}'")

    if command == "check-env":
        from lunar_lander.check_env import main as command_main
    elif command == "train-q-learning":
        from lunar_lander.learning.train_q_learning import main as command_main
    elif command == "evaluate-q-learning":
        from lunar_lander.learning.evaluate_q_learning import (
            main as command_main,
        )
        command_main(remaining)
        return
    elif command == "plot-q-learning":
        from lunar_lander.learning.plot_q_learning import main as command_main
    elif command == "train-ppo":
        from lunar_lander.learning.ppo_train import main as command_main
    elif command == "evaluate-ppo":
        from lunar_lander.learning.evaluate_ppo import (
            main as command_main,
        )
        command_main(remaining)
        return
    elif command == "record-landing":
        from lunar_lander.learning.record_landing import (
            main as command_main,
        )
        command_main(remaining)
        return
    else:
        from lunar_lander.learning.plot_ppo import main as command_main

    if remaining in (["-h"], ["--help"]):
        command_parser = argparse.ArgumentParser(
            prog=f"{parser.prog} {command}",
        )
        command_parser.print_help()
        return
    if remaining:
        parser.error(
            f"unrecognized arguments for {command}: {' '.join(remaining)}"
        )

    command_main()
