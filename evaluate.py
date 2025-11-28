# evaluate.py
"""
Unified evaluation script for CarRacing RL agents.

Usage:
    python evaluate.py --config configs/dqn.yaml --checkpoint checkpoints/dqn/dqn_ep1500.pth
    python evaluate.py --config configs/double_dqn.yaml --checkpoint checkpoints/double_dqn/double_dqn_ep1500.pth
    python evaluate.py --config configs/dueling_dqn.yaml --checkpoint checkpoints/dueling_dqn/dueling_dqn_ep1500.pth
"""

import argparse
import yaml
import gymnasium as gym
import torch
import time
import numpy as np

from utils.frame_stack import FrameStack
from utils.action_mapping import ACTIONS
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent
from agents.dueling_dqn_agent import DuelingDQNAgent
from agents.ddpg_agent import DDPGAgent
from agents.td3_agent import TD3Agent


# ------------------------------------------------------------------------
# Agent registry
# ------------------------------------------------------------------------
AGENT_REGISTRY = {
    "dqn": DQNAgent,
    "double_dqn": DoubleDQNAgent,
    "dueling_dqn": DuelingDQNAgent,
    "ddpg": DDPGAgent,
    "td3": TD3Agent
}


# ------------------------------------------------------------------------
# Evaluate function
# ------------------------------------------------------------------------
def evaluate(config_path, checkpoint_path, episodes=5, render=True):
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    agent_name = cfg["agent_name"]
    env_id = cfg["env"]["id"]
    num_stack = cfg["env"]["num_stack"]

    max_steps = cfg["training"]["max_steps_per_episode"]

    hp = cfg["hyperparameters"]
    exp = cfg["exploration"]

    # Select render mode
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)
    env.reset(seed=42)

    # Build agent
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

    is_continuous = getattr(agent, "is_continuous", False)

    # Load weights
    print(f"Loading checkpoint: {checkpoint_path}")
    agent.load(checkpoint_path)

    # Frame stack
    fs = FrameStack(num_stack)

    print(f"\n===== Evaluating {agent_name} =====")
    total_rewards = []

    # ------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------
    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        state = fs.reset(obs)

        ep_reward = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps:
            
            # Greedy action selection
            if is_continuous:
                cont_action = agent.select_action(state, eval_mode=True)
            else:
                action_idx = agent.select_action(state, eval_mode=True)
                cont_action = ACTIONS[action_idx]

            # Step environment
            next_obs, reward, terminated, truncated, _ = env.step(cont_action)
            done = terminated or truncated

            state = fs.append(next_obs)
            ep_reward += reward
            steps += 1

            # Slow rendering for visibility
            if render_mode == "human":
                time.sleep(0.01)

        total_rewards.append(ep_reward)
        print(f"Eval Episode {ep}: reward = {ep_reward:.1f}")

    env.close()

    print("\n===== Evaluation Complete =====")
    print(f"Average Reward Over {episodes} Episodes: {np.mean(total_rewards):.2f}")
    return total_rewards


# ------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (*.pth) file")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable visual rendering")

    args = parser.parse_args()

    evaluate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        render=not args.no_render
    )
