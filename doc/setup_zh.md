# 安装配置

本文档用于安装 A2 Isaac Gym 训练环境。

## 1. 前置条件

- Linux 系统，配备 NVIDIA GPU，并安装与 Isaac Gym 兼容的驱动。
- Conda 或其他 Python 环境管理工具。
- 本地安装 Isaac Gym Preview。
- 安装与 CUDA 和 Isaac Gym 匹配的 PyTorch。

## 2. 创建环境

```bash
conda create -n a2_rl_gym python=3.8
conda activate a2_rl_gym
```

按照 NVIDIA Isaac Gym 的本地安装说明安装 Isaac Gym，然后安装本仓库和内置的 RL 包：

```bash
cd a2_rl_gym
pip install -e .
pip install -e rsl_rl
```

## 3. 快速检查

```bash
python legged_gym/scripts/check_a2.py --headless --num_envs=16
```

## 4. 训练

```bash
python legged_gym/scripts/train.py --task=a2 --headless
```

训练 MoE-CTS 任务：

```bash
python legged_gym/scripts/train.py --task=a2_h042_moe_cts --headless
```

## 5. 播放

```bash
python legged_gym/scripts/play_a2.py --task=a2 --num_envs=1
```
