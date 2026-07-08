from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.a2.a2_config import (
    A2FlatCfg,
    A2FlatCfgPPO,
    A2H042MoECTSCfg,
    A2H042MoECTSCfgPPO,
    A2H042MoECTSRewardNormCfg,
    A2H042MoECTSRewardNormCfgPPO,
    A2H046OnlyCfg,
    A2H046OnlyCfgPPO,
    A2H046PoseCfg,
    A2H046PoseCfgPPO,
)
from legged_gym.envs.a2.a2_env import A2Robot
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register("a2", A2Robot, A2FlatCfg(), A2FlatCfgPPO())
task_registry.register(
    "a2_h046_pose",
    A2Robot,
    A2H046PoseCfg(),
    A2H046PoseCfgPPO(),
)
task_registry.register(
    "a2_h046_only",
    A2Robot,
    A2H046OnlyCfg(),
    A2H046OnlyCfgPPO(),
)
task_registry.register(
    "a2_h042_moe_cts",
    A2Robot,
    A2H042MoECTSCfg(),
    A2H042MoECTSCfgPPO(),
)
task_registry.register(
    "a2_h042_moe_cts_reward_norm",
    A2Robot,
    A2H042MoECTSRewardNormCfg(),
    A2H042MoECTSRewardNormCfgPPO(),
)
