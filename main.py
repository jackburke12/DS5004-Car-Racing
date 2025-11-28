# main.py
"""
Master script to train multiple CarRacing RL agents sequentially.

This script simply calls train.py three times using the YAML configs:
    - configs/dqn.yaml
    - configs/double_dqn.yaml
    - configs/dueling_dqn.yaml

You can easily expand the MODEL_CONFIGS list to train additional
agents (C51, PPO, Rainbow, etc.) without modifying any other script.
"""

import subprocess
import sys
import os
from pathlib import Path

# All models to train (in order)
MODEL_CONFIGS = [
    "configs/dqn.yaml",
    "configs/double_dqn.yaml",
    "configs/dueling_dqn.yaml",
]

def run_training(config_path):
    """Calls train.py with a given config file."""
    print("\n=====================================================")
    print(f"Starting training with config: {config_path}")
    print("=====================================================\n")

    # Call train.py as a subprocess so each model runs cleanly
    result = subprocess.run(
        [sys.executable, "train.py", "--config", config_path],
        capture_output=True,
        text=True
    )

    # Print output from train.py
    print(result.stdout)
    if result.stderr:
        print("Warnings/Errors:\n", result.stderr)

    print("\n=====================================================")
    print(f"Finished training: {config_path}")
    print("=====================================================\n")


def main():
    # Ensure configs exist
    for cfg in MODEL_CONFIGS:
        if not Path(cfg).exists():
            print(f"ERROR: Missing config file: {cfg}")
            return

    print("\n==============================================")
    print("  Starting Training of All RL Models")
    print("==============================================\n")

    # Train models sequentially
    for cfg in MODEL_CONFIGS:
        run_training(cfg)

    print("\n==============================================")
    print("  All Training Jobs Complete!")
    print("==============================================\n")


if __name__ == "__main__":
    main()
