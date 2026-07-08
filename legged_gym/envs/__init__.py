from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.a2.a2_config import (
    A2H042MoECTSRewardNormCfg,
    A2H042MoECTSRewardNormCfgPPO,
)
from legged_gym.envs.a2.a2_env import A2Robot
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register(
    "a2_h042_moe_cts_reward_norm",
    A2Robot,
    A2H042MoECTSRewardNormCfg(),
    A2H042MoECTSRewardNormCfgPPO(),
)
