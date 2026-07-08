import argparse
import math
import os
import sys

import isaacgym  # noqa: F401
import numpy as np
import torch
from isaacgym import gymapi

from legged_gym.envs import *  # noqa: F403
from legged_gym.utils import get_args, task_registry


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--record_mp4", action="store_true", default=False)
    parser.add_argument(
        "--output", type=str, default="logs/a2_check_zero_action.mp4"
    )
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--use_cpu_camera", action="store_true", default=True)
    parser.add_argument(
        "--no_gpu_camera_tensor", action="store_true", default=False
    )
    check_args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0], *remaining]
    args = get_args()
    args.record_mp4 = check_args.record_mp4
    args.output = check_args.output
    args.seconds = check_args.seconds
    args.width = check_args.width
    args.height = check_args.height
    args.use_cpu_camera = check_args.use_cpu_camera
    args.no_gpu_camera_tensor = check_args.no_gpu_camera_tensor
    return args


def _make_writer(output_path, fps, width, height):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        import imageio.v2 as imageio

        return "imageio", imageio.get_writer(output_path, fps=fps)
    except Exception as imageio_error:
        try:
            import cv2

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                output_path, fourcc, fps, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError(f"cv2 could not open video: {output_path}")
            return "cv2", writer
        except Exception as cv2_error:
            raise RuntimeError(
                "MP4 recording requires imageio and imageio-ffmpeg. "
                "Install with:\n"
                "  pip install imageio imageio-ffmpeg\n"
                "Alternatively install opencv-python for cv2.VideoWriter.\n"
                f"imageio error: {imageio_error}\n"
                f"cv2 error: {cv2_error}"
            ) from imageio_error


def _write_frame(writer_kind, writer, frame):
    if writer_kind == "imageio":
        writer.append_data(frame)
    else:
        import cv2

        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def _close_writer(writer):
    writer.close() if hasattr(writer, "close") else writer.release()


def _create_camera(env, width, height):
    camera_props = gymapi.CameraProperties()
    camera_props.width = width
    camera_props.height = height
    camera_props.enable_tensors = False
    camera_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    if camera_handle < 0:
        raise RuntimeError("Isaac Gym failed to create the A2 check camera")

    base_pos = env.root_states[0, :3].detach().cpu().numpy()
    camera_yaw = math.radians(45.0)
    camera_distance = 3.0
    camera_pos = gymapi.Vec3(
        float(base_pos[0] + camera_distance * math.cos(camera_yaw)),
        float(base_pos[1] + camera_distance * math.sin(camera_yaw)),
        1.2,
    )
    camera_target = gymapi.Vec3(
        float(base_pos[0]),
        float(base_pos[1]),
        max(0.2, float(base_pos[2]) * 0.65),
    )
    env.gym.set_camera_location(
        camera_handle, env.envs[0], camera_pos, camera_target
    )
    return camera_props, camera_handle


def check_a2(args):
    args.task = "a2"
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = args.num_envs or 16
    env_cfg.env.enable_camera_sensors = args.record_mp4
    if args.record_mp4:
        env_cfg.env.graphics_device_id = args.compute_device_id
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    for name in (
        "randomize_friction",
        "randomize_base_mass",
        "randomize_link_mass",
        "randomize_base_com",
        "randomize_restitution",
        "randomize_pd_gains",
        "randomize_motor_zero_offset",
        "randomize_motor_strength",
        "push_robots",
        "randomize_action_delay",
    ):
        setattr(env_cfg.domain_rand, name, False)
    env_cfg.commands.heading_command = False
    env_cfg.commands.zero_command_curriculum = None
    env_cfg.commands.limit_ang_vel_at_zero_command_prob = 0.0
    env_cfg.commands.limit_vel_prob = 0.0
    env_cfg.commands.dynamic_resample_commands = False
    env_cfg.commands.command_range_curriculum = []
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    observations, privileged_observations = env.reset()
    env.reset_termination_debug_counts()
    zero_actions = torch.zeros(
        env.num_envs, env.num_actions, device=env.device, dtype=torch.float
    )

    # Preserve the original five-second check unless recording was requested.
    duration = args.seconds if args.record_mp4 else 5.0
    num_steps = max(1, int(round(duration / env.dt)))
    reset_count = 0
    min_base_height = float("inf")
    max_abs_dof_velocity = 0.0

    output_path = os.path.abspath(args.output)
    camera_props = None
    camera_handle = None
    writer_kind = None
    writer = None

    try:
        if args.record_mp4:
            if not args.use_cpu_camera:
                raise ValueError(
                    "A2 check recording supports only the CPU camera path"
                )
            camera_props, camera_handle = _create_camera(
                env, args.width, args.height
            )
            fps = max(1, int(round(1.0 / env.dt)))
            writer_kind, writer = _make_writer(
                output_path, fps, args.width, args.height
            )
            print(
                "[A2 zero-action check] recording CPU camera MP4:",
                output_path,
                flush=True,
            )

        for _ in range(num_steps):
            observations, privileged_observations, rewards, resets, _ = env.step(
                zero_actions
            )
            tensors = (
                observations,
                privileged_observations,
                rewards,
                env.root_states,
                env.dof_pos,
                env.dof_vel,
                env.torques,
            )
            if not all(torch.isfinite(tensor).all() for tensor in tensors):
                raise RuntimeError("A2 zero-action check found NaN or Inf")

            if camera_handle is not None:
                env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                image = env.gym.get_camera_image(
                    env.sim,
                    env.envs[0],
                    camera_handle,
                    gymapi.IMAGE_COLOR,
                )
                frame = (
                    np.asarray(image)
                    .reshape(camera_props.height, camera_props.width, 4)[
                        :, :, :3
                    ]
                    .copy()
                )
                _write_frame(writer_kind, writer, frame)

            reset_count += int(resets.sum().item())
            min_base_height = min(
                min_base_height, float(env.root_states[:, 2].min().item())
            )
            max_abs_dof_velocity = max(
                max_abs_dof_velocity, float(env.dof_vel.abs().max().item())
            )
    finally:
        if writer is not None:
            _close_writer(writer)
        if camera_handle is not None:
            env.gym.destroy_camera_sensor(
                env.sim, env.envs[0], camera_handle
            )

    print("[A2 zero-action check] steps:", num_steps)
    print("[A2 zero-action check] resets:", reset_count)
    print("[A2 zero-action check] min base height:", min_base_height)
    print("[A2 zero-action check] max |dof velocity|:", max_abs_dof_velocity)
    print(
        "[A2 zero-action check] termination counts:",
        env.get_termination_debug_counts(),
    )

    if args.record_mp4:
        if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(f"MP4 writer produced no output: {output_path}")
        print("[A2 zero-action check] MP4 output:", output_path)
        print(
            "[A2 zero-action check] MP4 bytes:",
            os.path.getsize(output_path),
        )


if __name__ == "__main__":
    check_a2(_parse_args())
