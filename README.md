# A2 RL Gym

Unitree A2 Isaac Gym reinforcement learning training environment.

This repository contains the A2 task definitions, robot asset, training scripts, and bundled `rsl_rl` components needed to train locomotion policies in Isaac Gym.

## Installation

Follow the setup guide in [doc/setup_en.md](doc/setup_en.md).

## Available Tasks

- `a2`: baseline A2 velocity tracking task.
- `a2_h046_pose`: A2 task with a higher initial/base-height pose.
- `a2_h046_only`: A2 task with a higher base-height target.
- `a2_h042_moe_cts`: A2 MoE-CTS training task.
- `a2_h042_moe_cts_reward_norm`: A2 MoE-CTS task with reward normalization diagnostics.

The A2 robot asset is loaded from:

```text
resources/robots/a2/urdf/a2.urdf
```

## Quick Checks

Run a short zero-action asset and environment sanity check:

```bash
python legged_gym/scripts/check_a2.py --headless --num_envs=16
```

## Training

Train the baseline A2 task:

```bash
python legged_gym/scripts/train.py --task=a2 --headless
```

Train the A2 MoE-CTS task:

```bash
python legged_gym/scripts/train.py --task=a2_h042_moe_cts --headless
```

Common arguments:

- `--num_envs`: number of parallel Isaac Gym environments.
- `--max_iterations`: maximum training iterations.
- `--resume`: resume from a saved checkpoint.
- `--experiment_name`: override the experiment log folder.
- `--load_run`: run folder to load when resuming or playing.
- `--checkpoint`: checkpoint index to load.
- `--sim_device`: simulation device, for example `cuda:0` or `cpu`.
- `--rl_device`: RL device, for example `cuda:0` or `cpu`.

Training logs are written under:

```text
logs/<experiment_name>/<date_time>_<run_name>/
```

## Play

Play an A2 checkpoint with the A2-specific playback helper:

```bash
python legged_gym/scripts/play_a2.py --task=a2 --num_envs=1
```

For the generic playback script:

```bash
python legged_gym/scripts/play.py --task=a2 --num_envs=1
```

## Repository Layout

- `legged_gym/envs/a2/`: A2 environment and task configs.
- `resources/robots/a2/`: A2 URDF and mesh assets.
- `legged_gym/scripts/train.py`: Isaac Gym training entry point.
- `legged_gym/scripts/check_a2.py`: lightweight A2 asset/environment check.
- `legged_gym/scripts/play_a2.py`: A2 policy playback and recording helper.
- `rsl_rl/`: reinforcement learning algorithms and runners.

## Acknowledgements

This project builds on the Isaac Gym locomotion ecosystem, including `legged_gym`, `rsl_rl`, and Unitree reinforcement learning examples.

This project is adapted from go2-rl-gym / unitree_rl_gym and modified for Unitree A2 training in Isaac Gym. We thank the original authors and retain the corresponding license notices.

## License

See [LICENSE](LICENSE).
