# A2 RL Gym

Unitree A2 的 Isaac Gym 强化学习训练环境。

本仓库包含 A2 任务配置、机器人资产、训练脚本以及随仓库提供的 `rsl_rl` 组件，用于在 Isaac Gym 中训练 A2 运动控制策略。

## 安装

安装步骤请参考 [doc/setup_zh.md](doc/setup_zh.md)。

## 可用任务

- `a2_h042_moe_cts_reward_norm`：正式发布的 A2 MoE-CTS 训练任务，带奖励归一化诊断。

A2 机器人资产路径：

```text
resources/robots/a2/urdf/a2.urdf
```

## 快速检查

运行零动作环境和资产检查：

```bash
python legged_gym/scripts/check_a2.py --headless --num_envs=16
```

## 训练

训练正式 A2 任务：

```bash
python legged_gym/scripts/train.py --task=a2_h042_moe_cts_reward_norm --headless
```

常用参数：

- `--num_envs`：并行仿真环境数量。
- `--max_iterations`：最大训练迭代次数。
- `--resume`：从 checkpoint 继续训练。
- `--experiment_name`：覆盖实验日志目录名。
- `--load_run`：继续训练或播放时加载的 run。
- `--checkpoint`：加载的 checkpoint 编号。
- `--sim_device`：仿真设备，例如 `cuda:0` 或 `cpu`。
- `--rl_device`：强化学习设备，例如 `cuda:0` 或 `cpu`。

训练日志默认保存在：

```text
logs/<experiment_name>/<date_time>_<run_name>/
```

## 播放

使用 A2 专用播放脚本：

```bash
python legged_gym/scripts/play_a2.py --task=a2_h042_moe_cts_reward_norm --num_envs=1
```

也可以使用通用播放脚本：

```bash
python legged_gym/scripts/play.py --task=a2_h042_moe_cts_reward_norm --num_envs=1
```

## 仓库结构

- `legged_gym/envs/a2/`：A2 环境和任务配置。
- `resources/robots/a2/`：A2 URDF 和 mesh 资产。
- `legged_gym/scripts/train.py`：Isaac Gym 训练入口。
- `legged_gym/scripts/check_a2.py`：轻量级 A2 资产/环境检查。
- `legged_gym/scripts/play_a2.py`：A2 策略播放和录制辅助脚本。
- `rsl_rl/`：强化学习算法和 runner。

## 致谢

本项目基于 Isaac Gym 足式机器人训练生态，包括 `legged_gym`、`rsl_rl` 以及 Unitree 强化学习示例。

本项目基于 go2-rl-gym / unitree_rl_gym 改造，用于 Unitree A2 在 Isaac Gym 中的强化学习训练。感谢原项目作者，并保留相关许可证声明。

## 许可证

请查看 [LICENSE](LICENSE)。
