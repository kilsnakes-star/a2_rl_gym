import os
import shutil
import subprocess
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import  get_args, task_registry, Logger
from legged_gym.utils.exporter import export_policy_as_jit, export_policy_as_onnx, export_policy_as_pkl

import numpy as np
import torch
from PIL import Image

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    # env_cfg.terrain.mesh_type = 'plane'
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

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)

    record_video = args.record_video
    frame_dir = None
    video_path = None
    video_frame_idx = 0
    camera_handle = None
    camera_props = None
    video_env_id = 0
    if record_video:
        video_root = args.video_dir or os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'videos')
        os.makedirs(video_root, exist_ok=True)

        base_name = args.video_name
        if base_name is None:
            base_name = f"{train_cfg.runner.load_run}_{train_cfg.runner.checkpoint}"
        base_name = str(base_name).replace(os.sep, "_").replace(" ", "_")

        frame_dir = os.path.join(video_root, f"{base_name}_frames")
        os.makedirs(frame_dir, exist_ok=True)
        video_path = os.path.join(video_root, f"{base_name}.mp4")
        print(f"Recording frames to: {frame_dir}")
        print(f"Video will be written to: {video_path}")

        # Use a dedicated camera sensor instead of viewer screenshots.
        # Viewer captures can be black under xvfb even when simulation is running.
        video_env_id = max(0, min(int(args.video_env_id), env.num_envs - 1))
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1600
        camera_props.height = 900
        camera_handle = env.gym.create_camera_sensor(env.envs[video_env_id], camera_props)
        print(f"Recording from env_id={video_env_id}")

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        if hasattr(runner.alg, 'actor_critic'):
            model = runner.alg.actor_critic
        else:
            model = runner.alg.model
        export_policy_as_jit(model, path)
        export_policy_as_onnx(model, path)
        export_policy_as_pkl(model, path)
        print('Exported policy as jit script / onnx to: ', path)

    with torch.no_grad():
        for i in range(10*int(env.max_episode_length)):
            actions = policy(obs.detach())

            if FIX_COMMAND:
                env.commands[:, 0] = 1.0
                env.commands[:, 1] = 0.0
                env.commands[:, 2] = 0.0

            obs, _, rews, dones, infos = env.step(actions.detach())

            if record_video and (i % max(1, int(args.video_frame_skip)) == 0):
                # Keep the camera centered on the selected environment.
                root_pos = env.root_states[video_env_id, :3].detach().cpu().numpy()
                cam_pos = gymapi.Vec3(root_pos[0] + 4.0, root_pos[1] + 4.0, root_pos[2] + 3.0)
                cam_target = gymapi.Vec3(root_pos[0], root_pos[1], root_pos[2] + 0.6)
                env.gym.set_camera_location(camera_handle, env.envs[video_env_id], cam_pos, cam_target)
                env.gym.render_all_camera_sensors(env.sim)
                img = env.gym.get_camera_image(env.sim, env.envs[video_env_id], camera_handle, gymapi.IMAGE_COLOR)
                img = np.asarray(img, dtype=np.uint8).reshape((camera_props.height, camera_props.width, 4))[:, :, :3]
                frame_path = os.path.join(frame_dir, f"frame_{video_frame_idx:06d}.png")
                Image.fromarray(img, mode="RGB").save(frame_path)
                video_frame_idx += 1
                if args.video_max_frames > 0 and video_frame_idx >= args.video_max_frames:
                    print(f"Reached video_max_frames={args.video_max_frames}, stopping capture early.")
                    break

    if record_video:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            print("ffmpeg was not found; PNG frames were kept in:", frame_dir)
            return

        encoder_candidates = [
            ["libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            ["libopenh264", "-pix_fmt", "yuv420p"],
            ["mpeg4", "-q:v", "5"],
        ]
        last_error = None
        for encoder_args in encoder_candidates:
            ffmpeg_cmd = [
                ffmpeg,
                "-y",
                "-framerate",
                str(args.video_fps),
                "-i",
                os.path.join(frame_dir, "frame_%06d.png"),
                "-c:v",
                encoder_args[0],
                *encoder_args[1:],
                video_path,
            ]
            try:
                subprocess.run(ffmpeg_cmd, check=True)
                print(f"Saved video to: {video_path} using {encoder_args[0]}")
                if not args.keep_video_frames:
                    for name in os.listdir(frame_dir):
                        if name.endswith(".png"):
                            os.remove(os.path.join(frame_dir, name))
                    try:
                        os.rmdir(frame_dir)
                    except OSError:
                        pass
                return
            except subprocess.CalledProcessError as exc:
                last_error = exc
                print(f"ffmpeg encoder {encoder_args[0]} failed, trying fallback...")

        print(f"ffmpeg failed with exit code {getattr(last_error, 'returncode', 'unknown')}; PNG frames are still in: {frame_dir}")

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    FIX_COMMAND = True
    args = get_args()
    play(args)
