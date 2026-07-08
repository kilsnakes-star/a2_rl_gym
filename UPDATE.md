# Update Notes

This repository is prepared for the Unitree A2 Isaac Gym / RL training codebase.

## A2 Cleanup

- Kept A2 task registration for `a2`, `a2_h046_pose`, `a2_h046_only`, `a2_h042_moe_cts`, and `a2_h042_moe_cts_reward_norm`.
- Kept the A2 URDF and mesh assets under `resources/robots/a2/`.
- Removed legacy robot-specific training assets, deployment examples, checkpoints, and logs that are not needed for A2 training.
- Updated README and setup notes to describe the A2 training workflow.

## Quick Commands

```bash
python legged_gym/scripts/check_a2.py --headless --num_envs=16
python legged_gym/scripts/train.py --task=a2 --headless
python legged_gym/scripts/play_a2.py --task=a2 --num_envs=1
```
