# Setup

This guide installs the A2 Isaac Gym training environment.

## 1. Prerequisites

- Linux with an NVIDIA GPU and a driver compatible with Isaac Gym.
- Conda or another Python environment manager.
- Isaac Gym Preview installed locally.
- PyTorch compatible with your CUDA and Isaac Gym setup.

## 2. Create Environment

```bash
conda create -n a2_rl_gym python=3.8
conda activate a2_rl_gym
```

Install Isaac Gym following NVIDIA's local installation instructions, then install this repository and the bundled RL package:

```bash
cd a2_rl_gym
pip install -e .
pip install -e rsl_rl
```

## 3. Sanity Check

```bash
python legged_gym/scripts/check_a2.py --headless --num_envs=16
```

## 4. Train

```bash
python legged_gym/scripts/train.py --task=a2_h042_moe_cts_reward_norm --headless
```

## 5. Play

```bash
python legged_gym/scripts/play_a2.py --task=a2_h042_moe_cts_reward_norm --num_envs=1
```
