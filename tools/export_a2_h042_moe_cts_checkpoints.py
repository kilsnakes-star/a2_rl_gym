#!/usr/bin/env python3
"""Export A2 MoE-CTS checkpoints as stateful TorchScript student policies."""

import argparse
import copy
from pathlib import Path
from typing import List

# Isaac Gym must be imported before torch in environments where it is installed.
try:
    import isaacgym  # noqa: F401
except ModuleNotFoundError:
    isaacgym = None

import torch
from torch import Tensor, nn


class StatefulStudentMoECTSPolicy(nn.Module):
    """Student-only MoE-CTS inference policy with an internal observation history."""

    def __init__(
        self,
        student_moe_encoder: nn.Module,
        actor: nn.Module,
        history_length: int,
        obs_dim: int,
    ) -> None:
        super().__init__()
        self.student_moe_encoder = copy.deepcopy(student_moe_encoder).cpu()
        self.actor = copy.deepcopy(actor).cpu()
        self.history_length = history_length
        self.obs_dim = obs_dim
        self.register_buffer(
            "history",
            torch.zeros(0, history_length, obs_dim),
            persistent=False,
        )

    def forward(self, obs: Tensor) -> Tensor:
        if obs.dim() != 2 or obs.size(1) != self.obs_dim:
            raise ValueError("obs must have shape [B, obs_dim]")

        if self.history.size(0) != obs.size(0):
            self.history = obs.new_zeros(
                (obs.size(0), self.history_length, self.obs_dim)
            )

        self.history = torch.cat(
            (self.history[:, 1:, :], obs.unsqueeze(1)),
            dim=1,
        )
        latent, _ = self.student_moe_encoder(self.history.flatten(1))
        return self.actor(torch.cat((latent, obs), dim=1))

    @torch.jit.export
    def reset(self) -> None:
        self.history.zero_()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export A2 MoE-CTS checkpoints as TorchScript student policies."
    )
    parser.add_argument("--task", default="a2_h042_moe_cts")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_task_cfg(task_name: str):
    if isaacgym is None:
        raise ModuleNotFoundError(
            "Isaac Gym is required to import the registered legged_gym tasks."
        )

    import legged_gym.envs  # noqa: F401
    from legged_gym.utils import task_registry
    from legged_gym.utils.helpers import class_to_dict

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    policy_cfg = class_to_dict(train_cfg.policy)
    return env_cfg, train_cfg, policy_cfg


def build_actor_critic(task_name: str):
    from rsl_rl.modules import ActorCriticMoECTS

    env_cfg, train_cfg, policy_cfg = load_task_cfg(task_name)
    policy_class_name = train_cfg.runner.policy_class_name
    if policy_class_name != "ActorCriticMoECTS":
        raise ValueError(
            f"Task {task_name!r} uses {policy_class_name!r}, not 'ActorCriticMoECTS'."
        )

    obs_dim = int(env_cfg.env.num_observations)
    critic_obs_dim = int(env_cfg.env.num_privileged_obs)
    action_dim = int(env_cfg.env.num_actions)
    history_length = int(train_cfg.history_length)

    model = ActorCriticMoECTS(
        obs_dim,
        critic_obs_dim,
        action_dim,
        1,
        history_length,
        **policy_cfg,
    ).cpu()
    model.eval()
    return model, obs_dim, action_dim, history_length


def load_checkpoint(checkpoint_path: Path):
    try:
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # PyTorch versions bundled with older Isaac Gym releases do not expose
        # the weights_only argument.
        return torch.load(checkpoint_path, map_location="cpu")


def export_checkpoints(
    task_name: str,
    run_dir: Path,
    steps: List[int],
    out_dir: Path,
) -> List[Path]:
    model, obs_dim, action_dim, history_length = build_actor_critic(task_name)

    expected_obs_dim = 45
    expected_history_length = 5
    expected_latent_dim = 32
    expected_action_dim = 12
    actual_latent_dim = int(model.actor.network[0].in_features - obs_dim)
    contract = (
        obs_dim,
        history_length,
        actual_latent_dim,
        action_dim,
    )
    expected_contract = (
        expected_obs_dim,
        expected_history_length,
        expected_latent_dim,
        expected_action_dim,
    )
    if contract != expected_contract:
        raise ValueError(
            "Unexpected policy contract "
            f"(obs, history, latent, action)={contract}; expected {expected_contract}."
        )

    run_dir = run_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exported_paths: List[Path] = []
    for step in steps:
        checkpoint_path = run_dir / f"model_{step}.pt"
        if not checkpoint_path.is_file():
            print(f"[WARN] Checkpoint does not exist, skipping: {checkpoint_path}")
            continue

        checkpoint = load_checkpoint(checkpoint_path)
        if "model_state_dict" not in checkpoint:
            raise KeyError(f"{checkpoint_path} does not contain 'model_state_dict'.")

        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        policy = StatefulStudentMoECTSPolicy(
            model.student_moe_encoder,
            model.actor,
            history_length,
            obs_dim,
        ).eval()
        scripted_policy = torch.jit.script(policy)

        output_path = out_dir / f"policy_jit_{step}.pt"
        scripted_policy.save(str(output_path))

        loaded_policy = torch.jit.load(str(output_path), map_location="cpu")
        loaded_policy.eval()
        loaded_policy.reset()
        with torch.inference_mode():
            action = loaded_policy(torch.zeros(1, obs_dim))
        assert tuple(action.shape) == (1, action_dim), (
            f"Unexpected action shape {tuple(action.shape)} for {output_path}; "
            f"expected (1, {action_dim})."
        )

        print(f"checkpoint path: {checkpoint_path}")
        print(f"filename step: {step}")
        print(f"ckpt['iter']: {checkpoint.get('iter', '<missing>')}")
        print(f"output path: {output_path}")
        print(f"test action shape: {tuple(action.shape)}")
        exported_paths.append(output_path)

    return exported_paths


def main() -> None:
    args = parse_args()
    exported_paths = export_checkpoints(
        task_name=args.task,
        run_dir=args.run_dir,
        steps=args.steps,
        out_dir=args.out_dir,
    )

    print(f"Exported {len(exported_paths)} TorchScript student policies:")
    for path in exported_paths:
        print(path)


if __name__ == "__main__":
    main()
