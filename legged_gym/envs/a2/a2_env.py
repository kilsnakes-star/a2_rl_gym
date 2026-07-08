import torch

from legged_gym.envs.base.legged_robot import LeggedRobot


class A2Robot(LeggedRobot):
    """A2 morphology using the standard A2 PPO task structure."""

    def _create_envs(self):
        super()._create_envs()

        action_joint_names = list(self.cfg.asset.action_joint_names)
        default_joint_names = list(self.cfg.init_state.default_joint_angles.keys())
        if action_joint_names != default_joint_names:
            raise RuntimeError(
                "A2 action order must match default_joint_angles key order. "
                f"actions={action_joint_names}, defaults={default_joint_names}"
            )
        if len(self.dof_names) != 12:
            raise RuntimeError(
                f"A2 must load 12 DOFs, got {len(self.dof_names)}: {self.dof_names}"
            )
        if set(action_joint_names) != set(self.dof_names):
            missing = sorted(set(action_joint_names) - set(self.dof_names))
            extra = sorted(set(self.dof_names) - set(action_joint_names))
            raise RuntimeError(f"A2 joint mismatch. Missing={missing}, extra={extra}")
        if self.cfg.asset.base_link_name not in self.body_names:
            raise RuntimeError(
                f"A2 base body '{self.cfg.asset.base_link_name}' not found: "
                f"{self.body_names}"
            )

        self.action_to_dof = torch.tensor(
            [self.dof_names.index(name) for name in action_joint_names],
            dtype=torch.long,
            device=self.device,
        )
        self.dof_to_action = torch.tensor(
            [action_joint_names.index(name) for name in self.dof_names],
            dtype=torch.long,
            device=self.device,
        )
        self.hip_action_indices = torch.tensor(
            [
                action_joint_names.index(name)
                for name in (
                    "FL_hip_joint",
                    "FR_hip_joint",
                    "RL_hip_joint",
                    "RR_hip_joint",
                )
            ],
            dtype=torch.long,
            device=self.device,
        )

        self.isaac_foot_names = [
            name for name in self.body_names if self.cfg.asset.foot_name in name
        ]
        foot_names = list(self.cfg.asset.foot_body_names)
        missing_feet = [name for name in foot_names if name not in self.body_names]
        if missing_feet or set(foot_names) != set(self.isaac_foot_names):
            raise RuntimeError(
                "A2 configured feet do not match Isaac Gym feet. "
                f"configured={foot_names}, isaac={self.isaac_foot_names}"
            )
        self.feet_indices = self._body_indices(foot_names)
        if self.feet_indices.numel() != 4:
            raise RuntimeError(f"A2 must load four foot bodies, got {foot_names}")

        leg_contact_names = [
            f"{leg}_{segment}"
            for leg in ("FL", "FR", "RL", "RR")
            for segment in ("thigh", "calf")
        ]
        missing_leg_bodies = [
            name for name in leg_contact_names if name not in self.body_names
        ]
        if missing_leg_bodies:
            raise RuntimeError(
                f"A2 thigh/calf bodies not found: {missing_leg_bodies}"
            )
        self.penalized_contact_body_names = leg_contact_names
        self.penalised_contact_indices = self._body_indices(leg_contact_names)
        self.leg_termination_indices = self.penalised_contact_indices

        termination_names = []
        for pattern in self.cfg.asset.terminate_after_contacts_on:
            termination_names.extend(
                name for name in self.body_names if pattern in name
            )
        if termination_names != [self.cfg.asset.base_link_name]:
            raise RuntimeError(
                "A2-style termination must select only base_link, got "
                f"{termination_names}"
            )
        self.termination_body_names = termination_names
        self.termination_contact_indices = self._body_indices(termination_names)
        self.base_termination_indices = self.termination_contact_indices

        print("[A2 asset check] Isaac Gym DOF names:", self.dof_names)
        print(
            "[A2 asset check] cfg default joint key order:",
            default_joint_names,
        )
        print("[A2 asset check] action joint order:", action_joint_names)
        print("[A2 asset check] action->DOF indices:", self.action_to_dof.tolist())
        print("[A2 asset check] DOF->action indices:", self.dof_to_action.tolist())
        print("[A2 asset check] base body:", self.cfg.asset.base_link_name)
        print("[A2 asset check] raw Isaac foot order:", self.isaac_foot_names)
        print("[A2 asset check] reward foot order:", foot_names)
        print("[A2 asset check] foot body indices:", self.feet_indices.tolist())
        print(
            "[A2 asset check] penalized contact bodies:",
            self.penalized_contact_body_names,
        )
        print("[A2 asset check] terminate bodies:", self.termination_body_names)
        print(
            "[A2 asset check] terminate body indices:",
            self.termination_contact_indices.tolist(),
        )

    def _body_indices(self, body_names):
        indices = torch.tensor(
            [
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], name
                )
                for name in body_names
            ],
            dtype=torch.long,
            device=self.device,
        )
        if torch.any(indices < 0).item():
            raise RuntimeError(
                f"A2 failed to resolve rigid body indices for {body_names}"
            )
        return indices

    def _init_buffers(self):
        super()._init_buffers()
        self.rigid_body_states_view = self.rigid_body_states.view(
            self.num_envs, -1, 13
        )
        self._update_feet_state()
        self._observation_shapes_checked = False
        self.termination_reason_bufs = {
            name: torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            for name in (
                "base_contact",
                "thigh_calf_contact",
                "tilt",
                "timeout",
                "other",
            )
        }
        self.termination_debug_counts = {
            name: torch.zeros((), dtype=torch.long, device=self.device)
            for name in (*self.termination_reason_bufs.keys(), "total")
        }
        if self.torques.shape[-1] != self.torque_limits.numel():
            raise RuntimeError(
                "A2 torque tensor and URDF torque limits must both use Isaac "
                f"DOF order, got torques={self.torques.shape[-1]}, "
                f"limits={self.torque_limits.numel()}"
            )
        self.reward_torque_limits = self.torque_limits.unsqueeze(0).clamp_min(
            1.0e-6
        )
        fallback_vel_limit = float(
            getattr(self.cfg.rewards, "dof_vel_norm_limit", 30.0)
        )
        fallback_vel_limits = torch.full_like(
            self.dof_vel_limits, fallback_vel_limit
        )
        self.reward_dof_vel_limits = torch.where(
            self.dof_vel_limits > 0.0,
            self.dof_vel_limits,
            fallback_vel_limits,
        ).unsqueeze(0)
        self.enable_reward_diagnostics = bool(
            getattr(self.cfg.rewards, "enable_reward_diagnostics", False)
        )
        if self.enable_reward_diagnostics:
            diagnostic_names = (
                "rew_torques_raw_sum",
                "rew_torques_norm_mean",
                "torque_norm_mean",
                "torque_norm_p95",
                "dof_power_raw_sum",
                "dof_power_norm_mean",
                "projected_gravity_xy_mean",
                "base_lin_vel_z_abs_mean",
            )
            self.reward_diagnostic_sums = {
                name: torch.zeros(
                    self.num_envs,
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                )
                for name in diagnostic_names
            }
            self.reward_diagnostic_maxima = {
                "torque_norm_max": torch.zeros(
                    self.num_envs,
                    dtype=torch.float,
                    device=self.device,
                    requires_grad=False,
                )
            }

        if getattr(
            self.cfg.rewards, "normalize_torque_penalty", False
        ) or getattr(self.cfg.rewards, "normalize_power_penalty", False):
            print(
                "[A2 reward check] torque limits in Isaac DOF order:",
                self.torque_limits.tolist(),
            )
            print(
                "[A2 reward check] velocity limits in Isaac DOF order:",
                self.dof_vel_limits.tolist(),
            )

    def _update_feet_state(self):
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self._update_feet_state()

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = (
            noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        )
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.0
        noise_vec[9:21] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[21:33] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[33:45] = 0.0
        return noise_vec

    def _compute_torques(self, actions):
        if actions.shape[-1] != self.cfg.env.num_actions:
            raise RuntimeError(
                f"A2 expected {self.cfg.env.num_actions} actions, "
                f"got shape {tuple(actions.shape)}"
            )
        return super()._compute_torques(actions[:, self.dof_to_action])

    def reset_idx(self, env_ids):
        episode_steps = self.episode_length_buf[env_ids].clone()
        super().reset_idx(env_ids)
        if len(env_ids) > 0:
            for name, reason_buf in self.termination_reason_bufs.items():
                self.extras["episode"][f"termination/{name}_rate"] = (
                    reason_buf[env_ids].float().mean()
                )
            if self.enable_reward_diagnostics:
                episode_steps = episode_steps.float().clamp_min(1.0)
                for name, diagnostic_sum in self.reward_diagnostic_sums.items():
                    self.extras["episode"][name] = torch.mean(
                        diagnostic_sum[env_ids] / episode_steps
                    )
                    diagnostic_sum[env_ids] = 0.0
                for name, diagnostic_maximum in (
                    self.reward_diagnostic_maxima.items()
                ):
                    self.extras["episode"][name] = torch.mean(
                        diagnostic_maximum[env_ids]
                    )
                    diagnostic_maximum[env_ids] = 0.0

    def check_termination(self):
        super().check_termination()
        base_contact = torch.any(
            torch.norm(
                self.contact_forces[:, self.base_termination_indices, :], dim=-1
            )
            > 1.0,
            dim=1,
        )
        timeout = self.time_out_buf
        known_reset = base_contact | timeout
        false_reason = torch.zeros_like(known_reset)
        other = self.reset_buf.bool() & ~known_reset

        reason_values = {
            "base_contact": base_contact,
            "thigh_calf_contact": false_reason,
            "tilt": false_reason,
            "timeout": timeout,
            "other": other,
        }
        for name, value in reason_values.items():
            self.termination_reason_bufs[name][:] = value
            self.termination_debug_counts[name] += value.sum()
        self.termination_debug_counts["total"] += self.reset_buf.sum()

    def get_termination_debug_counts(self):
        return {
            name: int(value.item())
            for name, value in self.termination_debug_counts.items()
        }

    def reset_termination_debug_counts(self):
        for value in self.termination_debug_counts.values():
            value.zero_()

    def compute_observations(self):
        dof_pos = self.dof_pos[:, self.action_to_dof]
        dof_vel = self.dof_vel[:, self.action_to_dof]
        default_dof_pos = self.default_dof_pos[:, self.action_to_dof]
        actor_obs = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (dof_pos - default_dof_pos) * self.obs_scales.dof_pos,
                dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        self.obs_buf = actor_obs

        heights = (
            torch.clip(
                self.root_states[:, 2].unsqueeze(1)
                - 0.5
                - self.measured_heights,
                -1.0,
                1.0,
            )
            * self.obs_scales.height_measurements
        )
        foot_forces = (
            torch.norm(
                self.contact_forces[:, self.feet_indices, :], dim=-1
            )
            * 1.0e-3
        )
        torque_limits = self.torque_limits[self.action_to_dof].unsqueeze(0)
        torques = self.torques[:, self.action_to_dof] / torque_limits
        accelerations = (
            self.last_dof_vel[:, self.action_to_dof] - dof_vel
        ) / self.dt * 1.0e-4
        self.privileged_obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                actor_obs,
                foot_forces,
                torques,
                accelerations,
                heights,
            ),
            dim=-1,
        )

        if not self._observation_shapes_checked:
            actor_dim = actor_obs.shape[-1]
            critic_dim = self.privileged_obs_buf.shape[-1]
            if actor_dim != self.cfg.env.num_observations:
                raise RuntimeError(
                    f"A2 actor observation mismatch: cfg="
                    f"{self.cfg.env.num_observations}, actual={actor_dim}"
                )
            if critic_dim != self.cfg.env.num_privileged_obs:
                raise RuntimeError(
                    f"A2 critic observation mismatch: cfg="
                    f"{self.cfg.env.num_privileged_obs}, actual={critic_dim}"
                )
            if self.actions.shape[-1] != self.cfg.env.num_actions:
                raise RuntimeError(
                    f"A2 action mismatch: cfg={self.cfg.env.num_actions}, "
                    f"actual={self.actions.shape[-1]}"
                )
            if self.action_to_dof.numel() != self.cfg.env.num_actions:
                raise RuntimeError(
                    "A2 action-to-DOF mapping does not contain 12 entries"
                )
            self._observation_shapes_checked = True

        if self.add_noise:
            self.obs_buf += (
                2.0 * torch.rand_like(self.obs_buf) - 1.0
            ) * self.noise_scale_vec

    def _reward_hip_to_default(self):
        dof_pos = self.dof_pos[:, self.action_to_dof]
        default_dof_pos = self.default_dof_pos[:, self.action_to_dof]
        hip_pos = dof_pos[:, self.hip_action_indices]
        default_hip_pos = default_dof_pos[:, self.hip_action_indices]
        return torch.sum(torch.abs(hip_pos - default_hip_pos), dim=1)

    def _accumulate_reward_diagnostic(self, name, values):
        if self.enable_reward_diagnostics:
            self.reward_diagnostic_sums[name] += values.detach()

    def _reward_torques(self):
        if not getattr(
            self.cfg.rewards, "normalize_torque_penalty", False
        ):
            return super()._reward_torques()

        torque_norm = torch.abs(self.torques) / self.reward_torque_limits
        torque_norm_sq_mean = torch.mean(torch.square(torque_norm), dim=1)
        if self.enable_reward_diagnostics:
            torque_norm_top2 = torch.topk(
                torque_norm, k=2, dim=1
            ).values
            # Exact linear-interpolation p95 for 12 samples:
            # index = 0.95 * (12 - 1) = 10.45.
            torque_norm_p95 = (
                0.45 * torque_norm_top2[:, 0]
                + 0.55 * torque_norm_top2[:, 1]
            )
            self._accumulate_reward_diagnostic(
                "rew_torques_raw_sum",
                torch.sum(torch.square(self.torques), dim=1),
            )
            self._accumulate_reward_diagnostic(
                "rew_torques_norm_mean", torque_norm_sq_mean
            )
            self._accumulate_reward_diagnostic(
                "torque_norm_mean", torch.mean(torque_norm, dim=1)
            )
            self._accumulate_reward_diagnostic(
                "torque_norm_p95", torque_norm_p95
            )
            torque_norm_max = torch.max(torque_norm, dim=1).values
            self.reward_diagnostic_maxima["torque_norm_max"] = torch.maximum(
                self.reward_diagnostic_maxima["torque_norm_max"],
                torque_norm_max.detach(),
            )
        return torque_norm_sq_mean

    def _reward_dof_power(self):
        if not getattr(
            self.cfg.rewards, "normalize_power_penalty", False
        ):
            return super()._reward_dof_power()

        power_abs = torch.abs(self.torques * self.dof_vel)
        power_norm = power_abs / (
            self.reward_torque_limits * self.reward_dof_vel_limits
        )
        power_norm_mean = torch.mean(power_norm, dim=1)
        self._accumulate_reward_diagnostic(
            "dof_power_raw_sum", torch.sum(power_abs, dim=1)
        )
        self._accumulate_reward_diagnostic(
            "dof_power_norm_mean", power_norm_mean
        )
        return power_norm_mean

    def _reward_orientation(self):
        projected_gravity_xy = torch.sum(
            torch.square(self.projected_gravity[:, :2]), dim=1
        )
        self._accumulate_reward_diagnostic(
            "projected_gravity_xy_mean", projected_gravity_xy
        )
        return projected_gravity_xy

    def _reward_lin_vel_z(self):
        lin_vel_z_sq = torch.square(self.base_lin_vel[:, 2])
        self._accumulate_reward_diagnostic(
            "base_lin_vel_z_abs_mean", torch.abs(self.base_lin_vel[:, 2])
        )
        return lin_vel_z_sq
