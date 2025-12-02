# train.py
"""
Unified training script for CarRacing RL agents.

Adds:
    - Random warmup for continuous agents (DDPG/TD3)
    - TD3 uses ONLY soft target updates (no hard copy)
    - Clean agent registry and safe action handling
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
from agents.td3_agent import TD3Agent
from agents.sac_agent import SACAgent

# --------------------------------------------------------------
# Agent registry
# --------------------------------------------------------------
AGENT_REGISTRY = {
    "dqn": DQNAgent,
    "double_dqn": DoubleDQNAgent,
    "dueling_dqn": DuelingDQNAgent,
    "ddpg": DDPGAgent,
    "td3": TD3Agent,
    "sac": SACAgent
}

# --------------------------------------------------------------
# Training function
# --------------------------------------------------------------
def train(config_path):
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

    # --------------------------------------------------------------
    # --- DRIVE INTEGRATION (only activates inside Colab)
    # --------------------------------------------------------------
    IN_COLAB = False
    try:
        import google.colab  # noqa
        IN_COLAB = True
    except:
        pass

    if IN_COLAB:
        from google.colab import drive
        drive.mount('/content/drive')

        # Override output paths to live inside Drive
        base_dir = "/content/drive/MyDrive/car_racing_output"
        os.makedirs(base_dir, exist_ok=True)

        out["checkpoint_dir"] = os.path.join(base_dir, "checkpoints")
        out["results_csv"] = os.path.join(base_dir, f"{agent_name}_results.csv")

        print("\n[Google Drive] Output redirected to:")
        print(f"  Checkpoints → {out['checkpoint_dir']}")
        print(f"  Results CSV → {out['results_csv']}\n")

    # Create environment
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
        actor_lr = hp.get("actor_lr", hp["lr"]),
        critic_lr = hp.get("critic_lr", hp["lr"]),
        alpha = hp.get("alpha", 0.01),
        automatic_entropy_tuning = hp.get("automatic_entropy_tuning", True),
    )

    is_continuous = getattr(agent, "is_continuous", False)
    agent_uses_soft_update = getattr(agent, "uses_soft_update", False)

    # Frame stack
    fs = FrameStack(num_stack)

    # Output dirs
    os.makedirs(out["checkpoint_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(out["results_csv"]), exist_ok=True)

    episode_rewards = []
    avg50_rewards = []
    global_step = 0

    print(f"\n===== TRAINING {agent_name.upper()} =====")
    print(f"Episodes: {episodes}")
    print(f"Output directory: {out['checkpoint_dir']}")
    print("========================================\n")

    # --------------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------------
    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        state = fs.reset(obs)
        ep_reward = 0.0

        for step in range(max_steps):

            # ACTION SELECTION
            if is_continuous:
                if global_step < warmup_steps:
                    steer = np.random.uniform(-1.0, 1.0)
                    gas   = np.random.uniform(0.0, 1.0)
                    brake = np.random.uniform(0.0, 1.0)
                    cont_action = np.array([steer, gas, brake], dtype=np.float32)
                    action_for_store = cont_action
                else:
                    cont_action = agent.select_action(state)
                    action_for_store = cont_action
            else:
                action_idx = agent.select_action(state)
                cont_action = ACTIONS[action_idx]
                action_for_store = action_idx

            # ENV STEP
            next_obs, reward, terminated, truncated, info = env.step(cont_action)
            done = terminated or truncated
            next_state = fs.append(next_obs)

            agent.store_transition(state, action_for_store, reward, next_state, done)

            state = next_state
            ep_reward += reward
            global_step += 1
            agent.total_steps = global_step

            if global_step > warmup_steps and global_step % train_every == 0:
                for _ in range(updates_per_step):
                    agent.update()

            if (global_step % target_update_every == 0) and (not agent_uses_soft_update):
                agent.hard_update_target()

            if done:
                break

        # LOGGING
        episode_rewards.append(ep_reward)
        avg50 = np.mean(episode_rewards[-50:])
        avg50_rewards.append(avg50)

        print(
            f"Episode {ep:4d} | "
            f"Reward: {ep_reward:7.1f} | "
            f"Avg50: {avg50:7.2f} | "
            f"Eps: {agent.eps:.3f}"
        )

        # Save checkpoint
        if ep % 50 == 0:
            ckpt_path = os.path.join(out["checkpoint_dir"], f"{agent_name}_ep{ep}.pth")
            agent.save(ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # Save results
    df = pd.DataFrame({
        "EpisodeReward": episode_rewards,
        "Avg50Reward": avg50_rewards,
    })
    df.to_csv(out["results_csv"], index=False)

    print(f"\nTraining complete. Results saved to: {out['results_csv']}\n")

    # --------------------------------------------------------------
    # Optional auto-download at end (safe fallback)
    # --------------------------------------------------------------
    if IN_COLAB:
        try:
            from google.colab import files
            files.download(out["results_csv"])
        except:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)
