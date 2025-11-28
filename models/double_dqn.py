# models/double_dqn.py
"""
Double DQN network for CarRacing-v3.

Double DQN uses the *same exact architecture* as standard DQN.
The difference between DQN and Double DQN lies in the target
calculation logic inside the agent, not in the model structure.

This module simply re-exports the DQN architecture with a clear name.
"""

from models.dqn import DQN


# For clarity, DoubleDQN is just a DQN model.
class DoubleDQN(DQN):
    """
    Identical to DQN.

    Double DQN modifies:
        - Agent update rule (select action via online net,
          evaluate with target net)

    but the neural network itself is unchanged.
    """
    pass
