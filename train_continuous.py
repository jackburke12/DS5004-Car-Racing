# train.py
"""
Unified training script for CarRacing RL agents.

FIXED: Better reward shaping that encourages actual progress, not just survival.
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
# BETTER reward shaping for CarRacing
# --------------------------------------------------------------
def shape_reward(reward, done, step_count, info=None):
    """
    Key insight: Don't reward survival, reward PROGRESS.
    
    CarRacing gives positive rewards for visiting new track tiles.
    We should amplify positive rewards and heavily penalize crashes.
    """
    shaped_reward = reward
    
    # Amplify positive rewards (driving on new tiles)
    if reward > 0:
        shaped_reward = reward * 2.0  # Make progress more rewarding
    
    # Heavy penalty for early termination (crash/off-track)
    if done and step_count < 200:
        shaped_reward -= 100
    
    # NO survival bonus - this was the mistake!
    # The agent should only get rewarded for actual progress
    
    return shaped_reward

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

    # Create environment
    env = gym.make(env_id, render_mode=render_mode)
    
    # Apply frame skip wrapper if specified
    frame_skip = cfg["env"].get("frame_skip", 0)
    if frame_skip > 0:
        from utils.frame_skip_wrapper import FrameSkipWrapper
        env = FrameSkipWrapper(env, skip=frame_skip)
        print(f"Frame skip enabled: {frame_skip}x")
    
    env.reset(seed=42)

    AgentClass = AGENT_REGISTRY[agent_name]

    # Build kwargs common to all agents
    agent_kwargs = dict(
        replay_size=hp["replay_size"],
        batch_size=hp["batch_size"],
        gamma=hp["gamma"],
        eps_start=exp["eps_start"],
        eps_end=exp["eps_end"],
        eps_decay_steps=exp["eps_decay_steps"],
        in_channels=num_stack,
        img_h=84,
        img_w=84,
    )

    if agent_name == "sac":
        agent_kwargs["actor_lr"] = hp["actor_lr"]
        agent_kwargs["critic_lr"] = hp["critic_lr"]
        agent_kwargs["alpha"] = hp["alpha"]
        agent_kwargs["automatic_entropy_tuning"] = hp["automatic_entropy_tuning"]
        if "target_entropy" in hp:
            agent_kwargs["target_entropy"] = hp["target_entropy"]
    else:
        agent_kwargs["lr"] = hp["lr"]

    # Instantiate agent
    agent = AgentClass(**agent_kwargs)

    is_continuous = getattr(agent, "is_continuous", False)
    agent_uses_soft_update = getattr(agent, "uses_soft_update", False)

    # Frame stack
    fs = FrameStack(num_stack)

    # Output dirs
    os.makedirs(out["checkpoint_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(out["results_csv"]), exist_ok=True)

    episode_rewards = []
    episode_raw_rewards = []
    avg50_rewards = []
    global_step = 0

    print(f"\n===== TRAINING {agent_name.upper()} =====")
    print(f"Episodes: {episodes}")
    print(f"Reward Shaping: Progress-based (v2)")
    print(f"Output directory: {out['checkpoint_dir']}")
    print("========================================\n")

    # --------------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------------
    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        state = fs.reset(obs)
        ep_reward = 0.0
        ep_raw_reward = 0.0

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

            # REWARD SHAPING
            ep_raw_reward += reward
            shaped_reward = shape_reward(reward, done, step, info)

            agent.store_transition(state, action_for_store, shaped_reward, next_state, done)

            state = next_state
            ep_reward += shaped_reward
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
        episode_raw_rewards.append(ep_raw_reward)
        avg50 = np.mean(episode_rewards[-50:])
        avg50_raw = np.mean(episode_raw_rewards[-50:])
        avg50_rewards.append(avg50)

        print(
            f"Episode {ep:4d} | "
            f"Shaped: {ep_reward:7.1f} | "
            f"Raw: {ep_raw_reward:7.1f} | "
            f"Avg50: {avg50:7.2f} | "
            f"RawAvg50: {avg50_raw:7.2f} | "
            f"Steps: {step:4d} | "
            f"Alpha: {agent.log_alpha.exp().item():.3f}"
        )

        # Save checkpoint
        if ep % 100 == 0:
            ckpt_path = os.path.join(out["checkpoint_dir"], f"{agent_name}_ep{ep}.pth")
            agent.save(ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # Save results
    df = pd.DataFrame({
        "EpisodeReward": episode_rewards,
        "RawReward": episode_raw_rewards,
        "Avg50Reward": avg50_rewards,
    })
    df.to_csv(out["results_csv"], index=False)

    print(f"\nTraining complete. Results saved to: {out['results_csv']}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)