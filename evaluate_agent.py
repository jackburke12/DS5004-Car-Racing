"""
Evaluate a trained SAC agent and optionally record videos.

Usage:
    python evaluate_agent.py --checkpoint checkpoints/sac_ep700.pth --episodes 10 --record
"""

import argparse
import gymnasium as gym
import numpy as np
import torch
from pathlib import Path

from utils.frame_stack import FrameStack
from utils.frame_skip_wrapper import FrameSkipWrapper
from agents.sac_agent import SACAgent


def evaluate_agent(checkpoint_path, num_episodes=10, render=True, record=False, output_dir="videos"):
    """
    Load a trained agent and evaluate its performance.
    
    Args:
        checkpoint_path: Path to saved checkpoint
        num_episodes: Number of episodes to evaluate
        render: Whether to render during evaluation (human mode)
        record: Whether to save video recordings
        output_dir: Directory to save videos
    """
    
    # Create environment
    if record:
        # RecordVideo wrapper for saving videos
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env, 
            video_folder=output_dir,
            episode_trigger=lambda x: True,  # Record every episode
            name_prefix="sac_agent"
        )
    elif render:
        env = gym.make("CarRacing-v3", render_mode="human")
    else:
        env = gym.make("CarRacing-v3", render_mode=None)
    
    # Apply frame skip (must match training!)
    env = FrameSkipWrapper(env, skip=4)
    
    # Create frame stack (k=4 frames, 84x84 images)
    fs = FrameStack(k=4, img_h=84, img_w=84)
    
    # Initialize agent (same config as training)
    agent = SACAgent(
        replay_size=50000,  # Doesn't matter for evaluation
        batch_size=64,
        gamma=0.99,
        actor_lr=0.0003,
        critic_lr=0.0003,
        alpha=0.1,
        automatic_entropy_tuning=False,
        in_channels=4,
        img_h=84,
        img_w=84,
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    agent.load(checkpoint_path)
    agent.actor.eval()  # Set to evaluation mode
    
    # Evaluate
    episode_rewards = []
    episode_lengths = []
    
    print(f"\nEvaluating for {num_episodes} episodes...")
    print("=" * 60)
    
    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        state = fs.reset(obs)
        
        episode_reward = 0.0
        steps = 0
        done = False
        
        while not done:
            # Select action (deterministic for evaluation)
            action = agent.select_action(state, eval_mode=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = fs.append(obs)
            
            episode_reward += reward
            steps += 1
            
            if steps >= 1000:  # Max steps with frame_skip=4
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        print(f"Episode {ep:3d} | Reward: {episode_reward:7.1f} | Steps: {steps:3d}")
    
    # Summary statistics
    print("=" * 60)
    print(f"\nEvaluation Summary:")
    print(f"  Episodes:      {num_episodes}")
    print(f"  Mean Reward:   {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
    print(f"  Min Reward:    {np.min(episode_rewards):.1f}")
    print(f"  Max Reward:    {np.max(episode_rewards):.1f}")
    print(f"  Mean Steps:    {np.mean(episode_lengths):.1f}")
    
    if record:
        print(f"\nVideos saved to: {output_dir}/")
    
    env.close()
    
    return episode_rewards, episode_lengths


def compare_checkpoints(checkpoint_dir, episodes_per_checkpoint=5):
    """
    Evaluate multiple checkpoints to see training progression.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        episodes_per_checkpoint: How many episodes to evaluate each checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("sac_ep*.pth"))
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoints)} checkpoints")
    print("=" * 60)
    
    results = []
    
    for ckpt in checkpoints:
        # Extract episode number from filename
        ep_num = int(ckpt.stem.split("ep")[1])
        
        print(f"\nEvaluating {ckpt.name}...")
        rewards, _ = evaluate_agent(
            str(ckpt), 
            num_episodes=episodes_per_checkpoint,
            render=False,
            record=False
        )
        
        mean_reward = np.mean(rewards)
        results.append((ep_num, mean_reward))
        print(f"  Episode {ep_num}: Mean Reward = {mean_reward:.1f}")
    
    print("\n" + "=" * 60)
    print("Training Progression:")
    print("=" * 60)
    for ep_num, mean_reward in results:
        print(f"Episode {ep_num:4d}: {mean_reward:7.1f}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained SAC agent")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to checkpoint file")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true",
                       help="Disable rendering (faster)")
    parser.add_argument("--record", action="store_true",
                       help="Record videos of episodes")
    parser.add_argument("--output-dir", type=str, default="videos",
                       help="Directory to save videos")
    parser.add_argument("--compare", action="store_true",
                       help="Compare all checkpoints in directory")
    
    args = parser.parse_args()
    
    if args.compare:
        # Extract directory from checkpoint path
        checkpoint_dir = Path(args.checkpoint).parent
        compare_checkpoints(checkpoint_dir)
    else:
        evaluate_agent(
            args.checkpoint,
            num_episodes=args.episodes,
            render=not args.no_render,
            record=args.record,
            output_dir=args.output_dir
        )