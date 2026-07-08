import math

from legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgMoECTS,
    LeggedRobotCfgPPO,
)


class A2FlatCfg(LeggedRobotCfg):
    """A2 morphology with the standard A2 PPO task structure."""

    class env(LeggedRobotCfg.env):
        num_envs = 8192
        num_observations = 45
        num_privileged_obs = 45 + 3 + 4 + 12 + 12 + 187
        num_actions = 12
        episode_length_s = 25

    class terrain(LeggedRobotCfg.terrain):
        max_init_terrain_level = 5
        terrain_proportions = [
            0.05,
            0.20,
            0.05,
            0.25,
            0.10,
            0.20,
            0.0,
            0.0,
            0.15,
        ]
        move_down_by_accumulated_xy_command = True

    class commands(LeggedRobotCfg.commands):
        num_commands = 4
        resampling_time = 5.0
        heading_command = False
        zero_command_curriculum = {
            "start_iter": 0,
            "end_iter": 1500,
            "start_value": 0.0,
            "end_value": 0.1,
        }
        limit_ang_vel_at_zero_command_prob = 0.2
        limit_vel_prob = 0.2
        limit_vel_invert_when_continuous = True
        limit_vel = {
            "lin_vel_x": [-1, 1],
            "lin_vel_y": [-1, 1],
            "ang_vel_yaw": [-1, 0, 1],
        }
        stop_heading_at_limit = True
        dynamic_resample_commands = True
        command_range_curriculum = [
            {
                "iter": 20000,
                "lin_vel_x": [-1.0, 1.0],
                "lin_vel_y": [-1.0, 1.0],
                "ang_vel_yaw": [-1.5, 1.5],
                "heading": [-1.57, 1.57],
            },
            {
                "iter": 50000,
                "lin_vel_x": [-2.0, 2.0],
                "lin_vel_y": [-1.0, 1.0],
                "ang_vel_yaw": [-2.0, 2.0],
                "heading": [-1.57, 1.57],
            },
        ]
        turn_over_zero_time = {
            "backflip": 5.0,
            "sideflip": 3.0,
        }
        terrain_max_command_ranges = [
            {
                "lin_vel_x": [-1.5, 1.5],
                "lin_vel_y": [-1.0, 1.0],
                "ang_vel_yaw": [-1.5, 1.5],
                "heading": [-1.57, 1.57],
            },
            {
                "lin_vel_x": [-1.5, 1.5],
                "lin_vel_y": [-1.0, 1.0],
                "ang_vel_yaw": [-1.5, 1.5],
                "heading": [-1.57, 1.57],
            },
            {
                "lin_vel_x": [-1.5, 1.5],
                "lin_vel_y": [-1.0, 1.0],
                "ang_vel_yaw": [-1.5, 1.5],
                "heading": [-1.57, 1.57],
            },
            {
                "lin_vel_x": [-1.0, 1.0],
                "lin_vel_y": [-1.0, 1.0],
                "ang_vel_yaw": [-1.5, 1.5],
                "heading": [-1.57, 1.57],
            },
            {
                "lin_vel_x": [-1.0, 1.0],
                "lin_vel_y": [-1.0, 1.0],
                "ang_vel_yaw": [-1.5, 1.5],
                "heading": [-1.57, 1.57],
            },
            {
                "lin_vel_x": [-1.0, 1.0],
                "lin_vel_y": [-1.0, 1.0],
                "ang_vel_yaw": [-1.5, 1.5],
                "heading": [-1.57, 1.57],
            },
            {
                "lin_vel_x": [-1.0, 1.0],
                "lin_vel_y": [-1.0, 1.0],
                "ang_vel_yaw": [-1.5, 1.5],
                "heading": [-1.57, 1.57],
            },
            {
                "lin_vel_x": [-1.0, 1.0],
                "lin_vel_y": [-1.0, 1.0],
                "ang_vel_yaw": [-1.5, 1.5],
                "heading": [-1.57, 1.57],
            },
            {
                "lin_vel_x": [-2.0, 2.0],
                "lin_vel_y": [-1.0, 1.0],
                "ang_vel_yaw": [-2.0, 2.0],
                "heading": [-1.57, 1.57],
            },
        ]

        class ranges:
            lin_vel_x = [-0.5, 0.5]
            lin_vel_y = [-0.5, 0.5]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-1.57, 1.57]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.4]
        turn_over = False
        default_joint_angles = {
            "FL_hip_joint": -0.1,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.8,
            "FR_hip_joint": 0.1,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.8,
            "RL_hip_joint": -0.1,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.8,
            "RR_hip_joint": 0.1,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.8,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {
            "hip_joint": 100.0,
            "thigh_joint": 100.0,
            "calf_joint": 150.0,
        }
        damping = {
            "hip_joint": 4.0,
            "thigh_joint": 4.0,
            "calf_joint": 6.0,
        }
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/a2/urdf/a2.urdf"
        name = "a2"
        base_link_name = "base_link"
        foot_name = "_foot"
        foot_body_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        action_joint_names = [
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ]
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 0
        flip_visual_attachments = False
        armature = 0.03

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.0, 2.0]
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]
        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]
        randomize_base_com = True
        added_base_com_range = [-0.03, 0.03]
        randomize_restitution = True
        restitution_range = [0.0, 0.5]
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]
        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035]
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.6
        randomize_action_delay = True

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.38
        only_positive_rewards = False
        max_contact_force = 147.0
        curriculum_rewards = [
            {
                "reward_name": "lin_vel_z",
                "start_iter": 0,
                "end_iter": 1500,
                "start_value": 1.0,
                "end_value": 0.0,
            },
            {
                "reward_name": "correct_base_height",
                "start_iter": 0,
                "end_iter": 5000,
                "start_value": 1.0,
                "end_value": 10.0,
            },
        ]
        tracking_sigma = 0.25
        dynamic_sigma = {
            "min_lin_vel": 0.5,
            "max_lin_vel": 1.5,
            "min_ang_vel": 1.0,
            "max_ang_vel": 2.0,
            "max_sigma": [
                5 / 12,
                1 / 4,
                1 / 4,
                1 / 2,
                1 / 2,
                3 / 4,
                1,
                1,
                1 / 4,
            ],
        }
        min_legs_distance = 0.1

        class scales:
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            dof_acc = -2.5e-7
            dof_power = -2e-5
            torques = -1e-4
            correct_base_height = -1.0
            action_rate = -0.01
            action_smoothness = -0.01
            collision = -1.0
            dof_pos_limits = -2.0
            feet_regulation = -0.05
            hip_to_default = -0.05

        turn_over_roll_threshold = math.pi / 4

        class turn_over_scales:
            upright = 1.0

    class noise(LeggedRobotCfg.noise):
        add_noise = True

    class sim(LeggedRobotCfg.sim):
        dt = 0.005


class A2FlatCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        run_name = ""
        experiment_name = "a2_velocity"
        max_iterations = 150000
        save_interval = 500


class A2H046PoseCfg(A2FlatCfg):
    class init_state(A2FlatCfg.init_state):
        pos = [0.0, 0.0, 0.48]
        default_joint_angles = {
            "FL_hip_joint": -0.1,
            "FL_thigh_joint": 0.71,
            "FL_calf_joint": -1.42,
            "FR_hip_joint": 0.1,
            "FR_thigh_joint": 0.71,
            "FR_calf_joint": -1.42,
            "RL_hip_joint": -0.1,
            "RL_thigh_joint": 0.71,
            "RL_calf_joint": -1.42,
            "RR_hip_joint": 0.1,
            "RR_thigh_joint": 0.71,
            "RR_calf_joint": -1.42,
        }

    class rewards(A2FlatCfg.rewards):
        base_height_target = 0.46


class A2H046PoseCfgPPO(A2FlatCfgPPO):
    class runner(A2FlatCfgPPO.runner):
        experiment_name = "a2_velocity_h046_pose"


class A2H046OnlyCfg(A2FlatCfg):
    class rewards(A2FlatCfg.rewards):
        base_height_target = 0.42


class A2H046OnlyCfgPPO(A2FlatCfgPPO):
    class runner(A2FlatCfgPPO.runner):
        experiment_name = "a2_velocity_h046_only"


class A2H042MoECTSCfg(A2FlatCfg):
    class rewards(A2FlatCfg.rewards):
        base_height_target = 0.42


class A2H042MoECTSCfgPPO(LeggedRobotCfgMoECTS):
    class policy(LeggedRobotCfgMoECTS.policy):
        expert_num = 8

    class runner(LeggedRobotCfgMoECTS.runner):
        run_name = ""
        experiment_name = "a2_velocity_h042_moe_cts"
        max_iterations = 150000
        save_interval = 500


class A2H042MoECTSRewardNormCfg(A2H042MoECTSCfg):
    """A2 reward-scale experiment with morphology-normalized effort costs."""

    class rewards(A2H042MoECTSCfg.rewards):
        normalize_torque_penalty = True
        normalize_power_penalty = True
        enable_reward_diagnostics = True
        # Used only if an asset DOF has a missing/non-positive velocity limit.
        dof_vel_norm_limit = 30.0

        # Keep the base-height curriculum, but make vertical-velocity damping
        # persistent instead of fading it to zero after iteration 1500.
        curriculum_rewards = [
            {
                "reward_name": "correct_base_height",
                "start_iter": 0,
                "end_iter": 5000,
                "start_value": 1.0,
                "end_value": 10.0,
            },
        ]

        class scales(A2FlatCfg.rewards.scales):
            # Normalized costs are dimensionless. These values approximately
            # preserve the original A2-relative regularization strength.
            torques = -1.0
            dof_power = -0.2
            orientation = -1.0
            lin_vel_z = -0.5


class A2H042MoECTSRewardNormCfgPPO(A2H042MoECTSCfgPPO):
    class runner(A2H042MoECTSCfgPPO.runner):
        experiment_name = "a2_velocity_h042_moe_cts_reward_norm"
