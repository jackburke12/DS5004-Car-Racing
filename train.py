# train.py
"""
Unified training script for CarRacing RL agents.

Usage:
    python train.py --config configs/dqn.yaml
    python train.py --config configs/double_dqn.yaml
    python train.py --config configs/dueling_dqn.yaml

This script:
    - Loads YAML configuration
    - Instantiates the correct agent
    - Handles environment, frame stacking, training loop
    - Saves model checkpoints + training metrics
"""

import argparse
import yaml
import os
import time
import gymnasium as gym
import numpy as np
import torch
import pandas as pd

from utils.frame_stack import FrameStack
from utils.action_mapping import ACTIONS
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent
from agents.dueling_dqn_agent import DuelingDQNAgent
from agents.ddpg_agent import DDPGAgent


# --------------------------------------------------------------------------
# Agent registry (maps YAML "agent_name" to class)
# --------------------------------------------------------------------------
AGENT_REGISTRY = {
    "dqn": DQNAgent,
    "double_dqn": DoubleDQNAgent,
    "dueling_dqn": DuelingDQNAgent,
    "ddpg": DDPGAgent
}


# --------------------------------------------------------------------------
# Training function
# --------------------------------------------------------------------------
def train(config_path):
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    agent_name = cfg["agent_name"]
    env_id = cfg["env"]["id"]
    render_mode = cfg["env"].get("render_mode", None)
    num_stack = cfg["env"]["num_stack"]

    episodes = cfg["training"]["episodes"]
    max_steps = cfg["training"]["max_steps_per_episode"]
    warmup_steps = cfg["training"]["warmup_steps"]
    train_every = cfg["training"]["train_every"]
    updates_per_step = cfg["training"]["updates_per_step"]
    target_update_every = cfg["training"]["target_update_every"]

    hp = cfg["hyperparameters"]
    exp = cfg["exploration"]
    out = cfg["output"]

    # Instantiate environment
    env = gym.make(env_id, render_mode=render_mode)
    env.reset(seed=42)

    # Instantiate agent
    AgentClass = AGENT_REGISTRY[agent_name]
    agent = AgentClass(
        replay_size=hp["replay_size"],
        batch_size=hp["batch_size"],
        gamma=hp["gamma"],
        lr=hp["lr"],
        eps_start=exp["eps_start"],
        eps_end=exp["eps_end"],
        eps_decay_steps=exp["eps_decay_steps"],
        in_channels=num_stack,
        img_h=84,
        img_w=84,
    )

    # Frame stack
    fs = FrameStack(num_stack)

    # Output directories
    os.makedirs(out["checkpoint_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(out["results_csv"]), exist_ok=True)

    # Tracking
    episode_rewards = []
    avg50_rewards = []
    global_step = 0

    print(f"\n===== Training {agent_name} =====")
    print(f"Config: {config_path}")
    print(f"Episodes: {episodes}")
    print(f"Output: {out['checkpoint_dir']}")
    print("=================================\n")

    # ----------------------------------------------------------------------
    # MAIN TRAINING LOOP
    # ----------------------------------------------------------------------
    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        state = fs.reset(obs)

        ep_reward = 0.0

        for step in range(max_steps):
            # Select action
            if agent_name == "ddpg":
                # DDPG already outputs continuous env action [steer, gas, brake]
                cont_action = agent.select_action(state)
            else:
                action_idx = agent.select_action(state)
                cont_action = ACTIONS[action_idx]


            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(cont_action)
            done = terminated or truncated

            next_state = fs.append(next_obs)

            # Store transition
            if agent_name == "ddpg":
              agent.store_transition(state, cont_action, reward, next_state, done)
            else:
              agent.store_transition(state, action_idx, reward, next_state, done)


            state = next_state
            ep_reward += reward
            global_step += 1
            agent.total_steps = global_step

            # Training updates
            if global_step > warmup_steps and global_step % train_every == 0:
                for _ in range(updates_per_step):
                    agent.update()

            # Target network update
            if global_step % target_update_every == 0:
                agent.hard_update_target()

            if done:
                break

        # Record metrics
        episode_rewards.append(ep_reward)
        avg50 = np.mean(episode_rewards[-50:])
        avg50_rewards.append(avg50)

        print(
            f"Episode {ep:4d} | "
            f"Reward: {ep_reward:7.1f} | "
            f"Avg50: {avg50:7.2f} | "
            f"Epsilon: {agent.eps:.3f}"
        )

        # Save checkpoint every 50 episodes
        if ep % 50 == 0:
            ckpt_path = os.path.join(out["checkpoint_dir"], f"{agent_name}_ep{ep}.pth")
            agent.save(ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Final results to CSV
    results_df = pd.DataFrame({
        "EpisodeReward": episode_rewards,
        "Avg50EpReward": avg50_rewards
    })
    results_df.to_csv(out["results_csv"], index=False)

    print(f"\nTraining complete! Results saved to: {out['results_csv']}\n")


# --------------------------------------------------------------------------
# Script entrypoint
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    train(args.config)
