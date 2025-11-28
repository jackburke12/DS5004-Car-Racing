# plot_results.py
"""
Plot training results for CarRacing RL agents.

Usage:
    python plot_results.py --files results/dqn_results.csv \
                                   results/double_dqn_results.csv \
                                   results/dueling_dqn_results.csv

Plots:
    - Episode reward curves
    - Rolling average (default window=50)

Outputs:
    - figures/reward_curves.png
    - figures/rolling_average.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os


# -------------------------------------------------------------------
# Plot episode reward for each model
# -------------------------------------------------------------------
def plot_episode_rewards(csv_paths, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))

    for csv in csv_paths:
        df = pd.read_csv(csv)
        label = Path(csv).stem.replace("_results", "")
        plt.plot(df["EpisodeReward"], label=label, linewidth=1.5)

    plt.title("Episode Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(alpha=0.3)

    out_path = os.path.join(save_dir, "reward_curves.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


# -------------------------------------------------------------------
# Plot rolling average reward (default window=50)
# -------------------------------------------------------------------
def plot_rolling_average(csv_paths, window=50, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))

    for csv in csv_paths:
        df = pd.read_csv(csv)
        label = Path(csv).stem.replace("_results", "")

        roll = df["EpisodeReward"].rolling(window).mean()
        plt.plot(roll, label=f"{label} (avg{window})", linewidth=2)

    plt.title(f"Rolling Average Reward (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.legend()
    plt.grid(alpha=0.3)

    out_path = os.path.join(save_dir, f"rolling_average_{window}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


# -------------------------------------------------------------------
# Combined plotting
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="List of result CSV files to plot"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Rolling average window size"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="figures",
        help="Directory to save output plots"
    )
    args = parser.parse_args()

    csv_paths = args.files
    window = args.window
    outdir = args.outdir

    plot_episode_rewards(csv_paths, save_dir=outdir)
    plot_rolling_average(csv_paths, window=window, save_dir=outdir)


if __name__ == "__main__":
    main()
