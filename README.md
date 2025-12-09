# CarRacing-v3 Reinforcement Learning Comparison

## Purpose
This project compares the performance of multiple reinforcement learning algorithms (DQN, Double DQN, Dueling DQN, SAC) on the CarRacing-v3 environment from Gymnasium. The goal is to train agents that can autonomously navigate a racing track using only visual input (pixel observations).

## Major Functionality
- **Training Pipeline**: Unified training script supporting multiple RL algorithms
- **Preprocessing**: Frame stacking, grayscale conversion, normalization
- **Evaluation**: Standardized evaluation protocol across all agents
- **Visualization**: Training curves and performance metrics

## Class Methods

### `BaseAgent` (agents/base_agent.py)
- `select_action(state, eval_mode)`: Epsilon-greedy action selection
- `store_transition(...)`: Store experience in replay buffer
- `update()`: Abstract method for agent-specific learning updates
- `hard_update_target()`: Sync target network with online network
- `save(path)` / `load(path)`: Checkpoint management

### `DQNAgent` (agents/dqn_agent.py)
Inherits from `BaseAgent`, implements standard DQN update rule.

### `DoubleDQNAgent` (agents/double_dqn_agent.py)
Reduces Q-value overestimation by separating action selection and evaluation.

### `DuelingDQNAgent` (agents/dueling_dqn_agent.py)
Uses dueling architecture with separate value and advantage streams.

### `SACAgent` (agents/sac_agent.py)
Soft Actor-Critic for continuous action space with entropy regularization.

## Installation
We recommend either running the carracing_rl_colab.ipynb script in colab, which will import packages for you, or using the provided docker-compose file to build a docker container.

```bash
docker compose up
```

## Usage
```bash
# Train DQN agent
python train.py --config configs/dqn.yaml

# Evaluate trained agent
python evaluate.py --config configs/dqn.yaml --checkpoint checkpoints/dqn/dqn_ep1500.pth

# Plot training results
python plot_results.py --files results/dqn_results.csv results/double_dqn_results.csv
```