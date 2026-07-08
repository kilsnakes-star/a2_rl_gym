import argparse
import faulthandler
import math
import os
import sys

faulthandler.enable(all_threads=True)

import isaacgym  # noqa: F401
import numpy as np
import torch
from isaacgym import gymapi

from legged_gym.envs import *  # noqa: F403
from legged_gym.envs.a2.a2_env import A2Robot
from legged_gym.scripts.record_a2_stand import (
    _close_writer,
    _make_writer,
    _write_frame,
)
from legged_gym.utils import get_args, task_registry


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--task", type=str, default="a2_h042_moe_cts_reward_norm"
    )
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--output", type=str, default="logs/a2_play.mp4")
    parser.add_argument("--camera_distance", type=float, default=3.0)
    parser.add_argument("--camera_height", type=float, default=1.2)
    parser.add_argument("--camera_yaw", type=float, default=45.0)
    parser.add_argument("--use_cpu_camera", action="store_true", default=True)
    parser.add_argument(
        "--no_gpu_camera_tensor", action="store_true", default=False
    )
    parser.add_argument("--debug_no_camera", action="store_true", default=False)
    play_args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0], *remaining]
    args = get_args()
    args.task = play_args.task
    args.sim_device = play_args.sim_device
    args.rl_device = play_args.rl_device
    args.graphics_device_id = play_args.graphics_device_id
    args.vx = play_args.vx
    args.vy = play_args.vy
    args.yaw = play_args.yaw
    args.seconds = play_args.seconds
    args.output = play_args.output
    args.camera_distance = play_args.camera_distance
    args.camera_height = play_args.camera_height
    args.camera_yaw = play_args.camera_yaw
    args.use_cpu_camera = play_args.use_cpu_camera
    args.no_gpu_camera_tensor = play_args.no_gpu_camera_tensor
    args.debug_no_camera = play_args.debug_no_camera
    if args.num_envs is None:
        args.num_envs = 1
    return args


def _debug_print(message):
    print(message, flush=True)


def _validate_a2_task(task_name):
    try:
        task_class = task_registry.get_task_class(task_name)
    except KeyError as exc:
        raise ValueError(f"Unknown task: {task_name}") from exc
    if not issubclass(task_class, A2Robot):
        raise ValueError(
            "play_a2.py only supports tasks using A2Robot, "
            f"got task={task_name} env={task_class.__name__}"
        )


def _set_fixed_command(env, vx, vy, yaw):
    env.commands[:, 0] = vx
    env.commands[:, 1] = vy
    env.commands[:, 2] = yaw


def _get_policy_observations(env, device, vx, vy, yaw):
    _set_fixed_command(env, vx, vy, yaw)
    env.compute_observations()
    clip = env.cfg.normalization.clip_observations
    observations = torch.clip(env.get_observations(), -clip, clip)
    expected_command_obs = torch.tensor(
        [vx, vy, yaw],
        device=env.device,
        dtype=observations.dtype,
    ) * env.commands_scale
    if not torch.allclose(
        observations[:, 6:9],
        expected_command_obs.unsqueeze(0).expand(env.num_envs, -1),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Fixed command was not propagated to the policy observation: "
            f"expected={expected_command_obs.tolist()}, "
            f"actual={observations[0, 6:9].tolist()}"
        )
    return observations.to(device)


def _set_camera_pose(env, camera_handle, args):
    base_pos = env.root_states[0, :3].detach().cpu().numpy()
    camera_yaw = math.radians(args.camera_yaw)
    camera_pos = gymapi.Vec3(
        float(base_pos[0] + args.camera_distance * math.cos(camera_yaw)),
        float(base_pos[1] + args.camera_distance * math.sin(camera_yaw)),
        args.camera_height,
    )
    camera_target = gymapi.Vec3(
        float(base_pos[0]),
        float(base_pos[1]),
        max(0.2, float(base_pos[2]) * 0.65),
    )
    env.gym.set_camera_location(
        camera_handle, env.envs[0], camera_pos, camera_target
    )


def play_a2(args):
    _validate_a2_task(args.task)

    camera_enabled = not args.debug_no_camera
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = args.num_envs
    env_cfg.env.enable_camera_sensors = camera_enabled
    env_cfg.env.graphics_device_id = args.graphics_device_id
    env_cfg.terrain.num_rows = 7
    env_cfg.terrain.num_cols = 7
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_pd_gains = False
    env_cfg.domain_rand.randomize_motor_zero_offset = False
    env_cfg.env.test = True

    _debug_print(
        "[play_a2] configuration: "
        f"sim_device={args.sim_device} "
        f"rl_device={args.rl_device} "
        f"headless={args.headless} "
        f"camera_enabled={camera_enabled} "
        f"graphics_device_id="
        f"{args.graphics_device_id if camera_enabled else -1} "
        f"command=({args.vx}, {args.vy}, {args.yaw}) "
        f"terrain={env_cfg.terrain.mesh_type} "
        f"terrain_grid={env_cfg.terrain.num_rows}x"
        f"{env_cfg.terrain.num_cols} "
        f"terrain_curriculum={env_cfg.terrain.curriculum} "
        f"no_gpu_camera_tensor={args.no_gpu_camera_tensor} "
        "camera_api=CPU get_camera_image (GPU tensor disabled)"
    )
    _debug_print("[play_a2] before task creation")
    env, _ = task_registry.make_env(
        name=args.task, args=args, env_cfg=env_cfg
    )
    _debug_print(
        "[play_a2] after task creation: "
        f"sim_device_id={env.sim_device_id} "
        f"graphics_device_id={env.graphics_device_id}"
    )
    if hasattr(env, "terrain_levels"):
        _debug_print(
            "[play_a2] evaluation terrain env0: "
            f"level={int(env.terrain_levels[0].item())} "
            f"type={int(env.terrain_types[0].item())}"
        )

    train_cfg.runner.resume = True
    _debug_print(
        "[play_a2] before checkpoint load: "
        f"experiment={train_cfg.runner.experiment_name} "
        f"load_run={args.load_run if args.load_run is not None else -1} "
        f"checkpoint={args.checkpoint if args.checkpoint is not None else -1}"
    )
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    _debug_print("[play_a2] after checkpoint load")

    obs = _get_policy_observations(
        env, env.device, args.vx, args.vy, args.yaw
    )
    _debug_print(
        "[play_a2] fixed command observation env0: "
        f"raw={env.commands[0, :3].tolist()} "
        f"scaled_obs={obs[0, 6:9].tolist()}"
    )

    output_path = os.path.abspath(args.output)
    camera_props = None
    camera_handle = None
    writer_kind = None
    writer = None
    if camera_enabled:
        if not args.use_cpu_camera:
            raise ValueError(
                "play_a2.py supports only CPU camera images; "
                "keep --use_cpu_camera enabled."
            )
        _debug_print("[play_a2] before create camera")
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        camera_props.enable_tensors = False
        camera_handle = env.gym.create_camera_sensor(
            env.envs[0], camera_props
        )
        if camera_handle < 0:
            raise RuntimeError("Isaac Gym failed to create the camera sensor")
        _set_camera_pose(env, camera_handle, args)
        _debug_print(
            "[play_a2] after create camera: "
            f"handle={camera_handle} enable_tensors=False"
        )
        fps = max(1, int(round(1.0 / env.dt)))
        _debug_print("[play_a2] writing mp4: " + output_path)
        writer_kind, writer = _make_writer(output_path, fps)
    else:
        _debug_print(
            "[play_a2] debug_no_camera enabled; no camera or MP4 writer"
        )

    num_steps = max(1, int(round(args.seconds / env.dt)))
    print_interval = max(1, num_steps // 20)
    reset_count = 0
    max_abs_dof_velocity = 0.0

    _debug_print("[play_a2] before policy loop")
    try:
        with torch.inference_mode():
            for step in range(num_steps):
                actions = policy(obs)
                _, _, _, dones, _ = env.step(actions.to(env.device))
                obs = _get_policy_observations(
                    env, env.device, args.vx, args.vy, args.yaw
                )

                reset_count += int(dones.sum().item())
                max_abs_dof_velocity = max(
                    max_abs_dof_velocity,
                    float(env.dof_vel.abs().max().item()),
                )

                if camera_enabled:
                    if step == 0:
                        _debug_print(
                            "[play_a2] before render camera sensors"
                        )
                    _set_camera_pose(env, camera_handle, args)
                    env.gym.fetch_results(env.sim, True)
                    env.gym.step_graphics(env.sim)
                    env.gym.render_all_camera_sensors(env.sim)
                    if step == 0:
                        _debug_print(
                            "[play_a2] after render camera sensors"
                        )
                        _debug_print("[play_a2] before get camera image")
                    image = env.gym.get_camera_image(
                        env.sim,
                        env.envs[0],
                        camera_handle,
                        gymapi.IMAGE_COLOR,
                    )
                    if step == 0:
                        _debug_print("[play_a2] after get camera image")
                    frame = np.asarray(image).reshape(
                        camera_props.height, camera_props.width, 4
                    )[:, :, :3]
                    _write_frame(writer_kind, writer, frame)
                    if step == 0:
                        _debug_print("[play_a2] wrote first mp4 frame")

                if step % print_interval == 0 or step == num_steps - 1:
                    base_velocity = env.base_lin_vel[0]
                    actual_command = env.commands[0, :3]
                    command_obs = obs[0, 6:9]
                    linear_error = torch.linalg.vector_norm(
                        base_velocity[:2]
                        - torch.tensor(
                            [args.vx, args.vy],
                            device=env.device,
                            dtype=base_velocity.dtype,
                        )
                    )
                    yaw_error = torch.abs(
                        env.base_ang_vel[0, 2] - args.yaw
                    )
                    print(
                        f"[A2 play] step={step:4d}/{num_steps} "
                        f"cmd=("
                        f"{actual_command[0].item():.3f},"
                        f"{actual_command[1].item():.3f},"
                        f"{actual_command[2].item():.3f}) "
                        f"obs_cmd=("
                        f"{command_obs[0].item():.3f},"
                        f"{command_obs[1].item():.3f},"
                        f"{command_obs[2].item():.3f}) "
                        f"base_height={env.root_states[0, 2].item():.4f} "
                        f"base_vel=("
                        f"{base_velocity[0].item():.3f},"
                        f"{base_velocity[1].item():.3f},"
                        f"{base_velocity[2].item():.3f}) "
                        f"lin_error={linear_error.item():.3f} "
                        f"yaw_error={yaw_error.item():.3f} "
                        f"resets={reset_count} "
                        f"max|dof_vel|={env.dof_vel.abs().max().item():.3f}",
                        flush=True,
                    )
    finally:
        if writer is not None:
            _close_writer(writer)
        if camera_handle is not None:
            env.gym.destroy_camera_sensor(
                env.sim, env.envs[0], camera_handle
            )

    if camera_enabled:
        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(f"MP4 writer produced no output: {output_path}")
        print("[A2 play] output:", output_path)
        print("[A2 play] output bytes:", os.path.getsize(output_path))
    else:
        print("[A2 play] output: debug_no_camera; no MP4 generated")
    print("[A2 play] resets:", reset_count)
    print("[A2 play] max |dof velocity|:", max_abs_dof_velocity)


if __name__ == "__main__":
    play_a2(_parse_args())
