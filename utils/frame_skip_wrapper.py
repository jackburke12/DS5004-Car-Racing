"""
Frame skip wrapper for gymnasium environments.
Used by SAC to improve learning on continuous action space.

"""

import gymnasium as gym
import numpy as np


class FrameSkipWrapper(gym.Wrapper):
    """
    Repeat the agent's action for `skip` frames and accumulate rewards.
    
    Args:
        env: Gymnasium environment
        skip: Number of frames to repeat each action (default: 4)
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
    
    def step(self, action):
        """
        Execute action for `skip` frames and accumulate rewards.
        Stops early if episode terminates.
        """
        total_reward = 0.0
        done = False
        truncated = False
        final_obs = None
        final_info = {}
        
        for _ in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            final_obs = obs
            final_info = info
            
            # Stop if episode ends
            if done or truncated:
                break
        
        return final_obs, total_reward, done, truncated, final_info