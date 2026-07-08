# Useful Commands

## A2 Checks

```bash
python legged_gym/scripts/check_a2.py --headless --num_envs=16
```

## A2 Training

```bash
python legged_gym/scripts/train.py --task=a2_h042_moe_cts_reward_norm --headless
python legged_gym/scripts/train.py --task=a2_h042_moe_cts_reward_norm --num_envs=4096 --headless
```

## Resume Training

```bash
python legged_gym/scripts/train.py --task=a2_h042_moe_cts_reward_norm --resume --load_run <run_name> --checkpoint <iteration> --headless
```

## Play

```bash
python legged_gym/scripts/play_a2.py --task=a2_h042_moe_cts_reward_norm --num_envs=1
python legged_gym/scripts/play.py --task=a2_h042_moe_cts_reward_norm --num_envs=1
```
