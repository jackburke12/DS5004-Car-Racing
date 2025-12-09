# plot_results.py
"""
Plot training results for CarRacing RL agents.

Usage:
    python plot_results.py --files results/sac_results.csv

Outputs:
    - figures/combined_rewards.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os


# -------------------------------------------------------------------
# Plot both episode reward + rolling average in ONE figure
# -------------------------------------------------------------------
def plot_combined(csv_paths, window=50, save_dir="figures"):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))  # 2 subplots side-by-side
    ax1, ax2 = axes

    # ---------------- Left subplot: raw rewards ----------------
    for csv in csv_paths:
        df = pd.read_csv(csv)
        ax1.plot(df["RawReward"], label="Soft Actor-Critic", linewidth=1.5)

    ax1.set_title("Episode Rewards in Continuous Action Space")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # ---------------- Right subplot: rolling average ----------------
    for csv in csv_paths:
        df = pd.read_csv(csv)
        roll = df["RawReward"].rolling(window).mean()
        ax2.plot(roll, label="Soft Actor-Critic", linewidth=2)

    ax2.set_title(f"Average Reward Over {window} Episodes")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Avg Reward")
    ax2.grid(alpha=0.3)
    ax2.legend()

    # Tight layout so titles don’t overlap
    plt.tight_layout()

    out_path = os.path.join(save_dir, "combined_rewards.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


# -------------------------------------------------------------------
# Combined runner
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

    plot_combined(
        csv_paths=args.files,
        window=args.window,
        save_dir=args.outdir
    )


if __name__ == "__main__":
    main()
