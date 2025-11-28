# utils/action_mapping.py
"""
Discrete → continuous action mapping for CarRacing-v3.

All agents (DQN, Double DQN, Dueling DQN, etc.) use the same
5-action discretization.

Each action is a continuous vector:
    [steer, gas, brake]

These are fed directly into env.step().
"""

import numpy as np

# 5 discrete actions used by all agents
ACTIONS = [
    np.array([0.0, 0.0, 0.0], dtype=np.float32),   # 0: do nothing
    np.array([+1.0, 0.0, 0.0], dtype=np.float32),  # 1: steer right
    np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # 2: steer left
    np.array([0.0, 1.0, 0.0], dtype=np.float32),   # 3: gas
    np.array([0.0, 0.0, 0.8], dtype=np.float32),   # 4: brake
]

NUM_ACTIONS = len(ACTIONS)
