import argparse
import faulthandler
import math
import os
import sys

faulthandler.enable(all_threads=True)

import isaacgym  # noqa: F401
import numpy as np
import torch
from isaacgym import gymapi, gymtorch

from legged_gym.envs import *  # noqa: F403
from legged_gym.utils import get_args, task_registry


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", type=str, default="a2")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--output", type=str, default="logs/a2_stand.mp4")
    parser.add_argument("--camera_distance", type=float, default=3.0)
    parser.add_argument("--camera_height", type=float, default=1.2)
    parser.add_argument("--camera_yaw", type=float, default=45.0)
    parser.add_argument("--disable_camera", action="store_true", default=False)
    parser.add_argument(
        "--no_gpu_camera_tensor", action="store_true", default=False
    )
    parser.add_argument("--use_cpu_camera", action="store_true", default=True)
    parser.add_argument("--debug_no_camera", action="store_true", default=False)
    record_args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0], *remaining]
    args = get_args()
    args.sim_device = record_args.sim_device
    args.rl_device = record_args.rl_device
    args.seconds = record_args.seconds
    args.output = record_args.output
    args.camera_distance = record_args.camera_distance
    args.camera_height = record_args.camera_height
    args.camera_yaw = record_args.camera_yaw
    args.graphics_device_id = record_args.graphics_device_id
    args.disable_camera = record_args.disable_camera
    args.no_gpu_camera_tensor = record_args.no_gpu_camera_tensor
    args.use_cpu_camera = record_args.use_cpu_camera
    args.debug_no_camera = record_args.debug_no_camera
    args.task = record_args.task
    if args.num_envs is None:
        args.num_envs = 1
    return args


def _make_writer(output_path, fps):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        import imageio.v2 as imageio

        return "imageio", imageio.get_writer(output_path, fps=fps)
    except Exception as imageio_error:
        try:
            import cv2

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (1280, 720))
            if not writer.isOpened():
                raise RuntimeError(f"cv2 could not open video: {output_path}")
            return "cv2", writer
        except Exception as cv2_error:
            raise RuntimeError(
                "MP4 recording requires imageio and imageio-ffmpeg. Install with:\n"
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


def _set_deterministic_stand_state(env):
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.root_states[env_ids] = env.base_init_state
    env.root_states[env_ids, :3] += env.env_origins[env_ids]
    env.root_states[env_ids, 3:7] = torch.tensor(
        [0.0, 0.0, 0.0, 1.0], device=env.device
    )
    env.root_states[env_ids, 7:13] = 0.0
    env_ids_int32 = env_ids.to(dtype=torch.int32)
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_states),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )


def _debug_print(message):
    print(message, flush=True)


def record_a2_stand(args):
    camera_enabled = not (args.disable_camera or args.debug_no_camera)
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = args.num_envs
    env_cfg.env.enable_camera_sensors = camera_enabled
    env_cfg.env.graphics_device_id = args.graphics_device_id
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

    _debug_print(
        "[record_a2] configuration: "
        f"sim_device={args.sim_device} "
        f"rl_device={args.rl_device} "
        f"headless={args.headless} "
        f"camera_enabled={camera_enabled} "
        f"graphics_device_id="
        f"{args.graphics_device_id if camera_enabled else -1} "
        f"no_gpu_camera_tensor={args.no_gpu_camera_tensor} "
        "camera_api=CPU get_camera_image (GPU tensor disabled)"
    )
    _debug_print("[record_a2] before task creation")
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _debug_print(
        "[record_a2] after task creation: "
        f"sim_device_id={env.sim_device_id} "
        f"graphics_device_id={env.graphics_device_id}"
    )
    _debug_print("[record_a2] before env reset")
    env.reset()
    _debug_print("[record_a2] after env reset")
    _set_deterministic_stand_state(env)
    env.reset_termination_debug_counts()

    output_path = os.path.abspath(args.output)
    fps = max(1, int(round(1.0 / env.dt)))
    camera_props = None
    camera_handle = None
    writer_kind = None
    writer = None
    if camera_enabled:
        if not args.use_cpu_camera:
            raise ValueError(
                "record_a2_stand.py currently supports only the safe CPU camera "
                "path; keep --use_cpu_camera enabled."
            )
        _debug_print("[record_a2] before create camera")
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        camera_props.enable_tensors = False
        camera_handle = env.gym.create_camera_sensor(
            env.envs[0], camera_props
        )
        if camera_handle < 0:
            raise RuntimeError("Isaac Gym failed to create the camera sensor")
        _debug_print(
            "[record_a2] after create camera: "
            f"handle={camera_handle} enable_tensors=False"
        )

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

        _debug_print("[record_a2] writing mp4: " + output_path)
        writer_kind, writer = _make_writer(output_path, fps)
    else:
        _debug_print(
            "[record_a2] camera disabled; running zero-action stand check only"
        )

    zero_actions = torch.zeros(
        env.num_envs, env.num_actions, device=env.device, dtype=torch.float
    )
    num_steps = max(1, int(round(args.seconds / env.dt)))
    print_interval = max(1, num_steps // 10)

    min_base_height = float("inf")
    max_base_height = float("-inf")
    max_abs_dof_velocity = 0.0
    min_foot_height = float("inf")
    max_base_contact_force = 0.0
    max_leg_contact_force = 0.0
    reset_count = 0
    steady_state_heights = []

    _debug_print("[record_a2] before step loop")
    try:
        for step in range(num_steps):
            _, _, _, resets, _ = env.step(zero_actions)
            if camera_enabled:
                if step == 0:
                    _debug_print(
                        "[record_a2] before render camera sensors"
                    )
                env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                if step == 0:
                    _debug_print("[record_a2] after render camera sensors")
                    _debug_print("[record_a2] before get camera image")
                image = env.gym.get_camera_image(
                    env.sim,
                    env.envs[0],
                    camera_handle,
                    gymapi.IMAGE_COLOR,
                )
                if step == 0:
                    _debug_print("[record_a2] after get camera image")
                frame = np.asarray(image).reshape(
                    camera_props.height, camera_props.width, 4
                )[:, :, :3]
                _write_frame(writer_kind, writer, frame)
                if step == 0:
                    _debug_print("[record_a2] wrote first mp4 frame")

            base_height = float(env.root_states[0, 2].item())
            min_base_height = min(min_base_height, base_height)
            max_base_height = max(max_base_height, base_height)
            max_abs_dof_velocity = max(
                max_abs_dof_velocity, float(env.dof_vel.abs().max().item())
            )
            min_foot_height = min(
                min_foot_height, float(env.feet_pos[:, :, 2].min().item())
            )
            max_base_contact_force = max(
                max_base_contact_force,
                float(
                    torch.norm(
                        env.contact_forces[:, env.base_termination_indices, :],
                        dim=-1,
                    )
                    .max()
                    .item()
                ),
            )
            max_leg_contact_force = max(
                max_leg_contact_force,
                float(
                    torch.norm(
                        env.contact_forces[:, env.leg_termination_indices, :],
                        dim=-1,
                    )
                    .max()
                    .item()
                ),
            )
            reset_count += int(resets.sum().item())
            if step >= num_steps // 2:
                steady_state_heights.append(base_height)

            if step % print_interval == 0 or step == num_steps - 1:
                roll = math.degrees(float(env.rpy[0, 0].item()))
                pitch = math.degrees(float(env.rpy[0, 1].item()))
                print(
                    f"[A2 stand] step={step:4d}/{num_steps} "
                    f"base_height={base_height:.4f} "
                    f"roll={roll:.2f}deg pitch={pitch:.2f}deg "
                    f"max|dof_vel|={env.dof_vel.abs().max().item():.4f} "
                    f"resets={reset_count}",
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
        print("[A2 stand] output:", output_path)
        print("[A2 stand] output bytes:", os.path.getsize(output_path))
    else:
        print("[A2 stand] output: camera disabled; no MP4 generated")

    print(
        f"[A2 stand] base height range: "
        f"[{min_base_height:.6f}, {max_base_height:.6f}]"
    )
    print(
        f"[A2 stand] steady-state base height (second half): "
        f"min={min(steady_state_heights):.6f} "
        f"mean={sum(steady_state_heights) / len(steady_state_heights):.6f} "
        f"max={max(steady_state_heights):.6f}"
    )
    print("[A2 stand] minimum foot center height:", min_foot_height)
    print("[A2 stand] max base contact force:", max_base_contact_force)
    print("[A2 stand] max thigh/calf contact force:", max_leg_contact_force)
    print("[A2 stand] resets:", reset_count)
    print("[A2 stand] max |dof velocity|:", max_abs_dof_velocity)
    joint_positions = env.dof_pos[0, env.action_to_dof].detach().cpu()
    joint_targets = env.default_dof_pos[0, env.action_to_dof].detach().cpu()
    print("[A2 stand] final joint positions (action order):", joint_positions.tolist())
    print(
        "[A2 stand] final max |joint position error|:",
        float(torch.max(torch.abs(joint_positions - joint_targets)).item()),
    )
    print(
        "[A2 stand] termination counts:",
        env.get_termination_debug_counts(),
    )


if __name__ == "__main__":
    record_a2_stand(_parse_args())
