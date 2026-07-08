from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING


class FlatTurnVelocityCommand(UniformVelocityCommand):
    """Velocity command that gives flat-turn terrains a yaw-turning curriculum."""

    cfg: "FlatTurnVelocityCommandCfg"

    def __init__(self, cfg: "FlatTurnVelocityCommandCfg", env):
        super().__init__(cfg, env)
        self._turn_terrain_type_ids = self._resolve_turn_terrain_type_ids()

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)

        turn_env_ids = self._turn_env_ids(env_ids)
        if turn_env_ids.numel() == 0:
            return

        terrain = self._env.scene.terrain
        terrain_level = terrain.terrain_levels[turn_env_ids].float()
        max_level = max(1, int(getattr(terrain, "max_terrain_level", 1)) - 1)
        difficulty = torch.clamp(terrain_level / float(max_level), 0.0, 1.0)

        lin_y_sign = torch.where(
            torch.rand(turn_env_ids.numel(), device=self.device) < 0.5,
            -torch.ones(turn_env_ids.numel(), device=self.device),
            torch.ones(turn_env_ids.numel(), device=self.device),
        )
        ang_sign = torch.where(
            torch.rand(turn_env_ids.numel(), device=self.device) < 0.5,
            -torch.ones(turn_env_ids.numel(), device=self.device),
            torch.ones(turn_env_ids.numel(), device=self.device),
        )

        lin_x_range = self._lerp_range(self.cfg.turn_lin_vel_x_start, self.cfg.turn_lin_vel_x_end, difficulty)
        lin_y_abs_range = self._lerp_range(
            self.cfg.turn_lin_vel_y_start_abs, self.cfg.turn_lin_vel_y_end_abs, difficulty
        )
        ang_abs_range = self._lerp_range(self.cfg.turn_ang_vel_z_start_abs, self.cfg.turn_ang_vel_z_end_abs, difficulty)

        lin_x = self._sample_range(lin_x_range)
        lin_y = lin_y_sign * self._sample_range(lin_y_abs_range)
        ang_z = ang_sign * self._sample_range(ang_abs_range)

        mode = self._sample_flat_locomotion_mode(turn_env_ids.numel())
        backward_only = mode == 0
        lateral_only = mode == 1
        turn_only = mode == 2

        lin_x[lateral_only | turn_only] = 0.0
        lin_y[backward_only | turn_only] = 0.0
        ang_z[backward_only | lateral_only] = 0.0

        self.vel_command_b[turn_env_ids, 0] = lin_x
        self.vel_command_b[turn_env_ids, 1] = lin_y
        self.vel_command_b[turn_env_ids, 2] = ang_z
        self.is_standing_env[turn_env_ids] = False

    def _update_command(self):
        super()._update_command()
        turn_env_ids = self._turn_env_ids()
        if turn_env_ids.numel() > 0:
            self.is_standing_env[turn_env_ids] = False

    def _turn_env_ids(self, env_ids=None) -> torch.Tensor:
        if not self._turn_terrain_type_ids or not hasattr(self._env.scene, "terrain"):
            return torch.empty(0, dtype=torch.long, device=self.device)
        terrain = self._env.scene.terrain
        if not hasattr(terrain, "terrain_types"):
            return torch.empty(0, dtype=torch.long, device=self.device)

        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        type_ids = torch.tensor(self._turn_terrain_type_ids, dtype=torch.long, device=self.device)
        return env_ids_tensor[torch.isin(terrain.terrain_types[env_ids_tensor], type_ids)]

    def _resolve_turn_terrain_type_ids(self) -> list[int]:
        terrain_cfg = getattr(getattr(self._env.scene, "terrain", None), "cfg", None)
        generator_cfg = getattr(terrain_cfg, "terrain_generator", None)
        if generator_cfg is None or not getattr(generator_cfg, "sub_terrains", None):
            return []

        names = list(generator_cfg.sub_terrains.keys())
        proportions = [float(sub_cfg.proportion) for sub_cfg in generator_cfg.sub_terrains.values()]
        total = sum(proportions)
        if total <= 0.0:
            return []
        cumulative = []
        running = 0.0
        for proportion in proportions:
            running += proportion / total
            cumulative.append(running)

        turn_names = set(self.cfg.turn_terrain_names)
        type_ids = []
        for col in range(int(generator_cfg.num_cols)):
            threshold = col / float(generator_cfg.num_cols) + 0.001
            sub_index = next(i for i, value in enumerate(cumulative) if threshold < value)
            if names[sub_index] in turn_names:
                type_ids.append(col)
        return type_ids

    def _lerp_range(
        self, start_range: tuple[float, float], end_range: tuple[float, float], difficulty: torch.Tensor
    ) -> torch.Tensor:
        start = torch.tensor(start_range, device=self.device)
        end = torch.tensor(end_range, device=self.device)
        return start + (end - start) * difficulty.unsqueeze(1)

    def _sample_range(self, ranges: torch.Tensor) -> torch.Tensor:
        return ranges[:, 0] + torch.rand(ranges.shape[0], device=self.device) * (ranges[:, 1] - ranges[:, 0])

    def _sample_flat_locomotion_mode(self, num_commands: int) -> torch.Tensor:
        probabilities = torch.tensor(self.cfg.flat_locomotion_mode_probabilities, device=self.device)
        probabilities = probabilities / torch.sum(probabilities)
        return torch.multinomial(probabilities, num_commands, replacement=True)


@configclass
class FlatTurnVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    class_type: type = FlatTurnVelocityCommand

    turn_terrain_names: tuple[str, ...] = ("flat_turn",)
    turn_lin_vel_x_start: tuple[float, float] = (0.3, 0.6)
    turn_lin_vel_x_end: tuple[float, float] = (0.0, 0.05)
    turn_lin_vel_y_start_abs: tuple[float, float] = (0.10, 0.25)
    turn_lin_vel_y_end_abs: tuple[float, float] = (0.0, 0.03)
    turn_ang_vel_z_start_abs: tuple[float, float] = (0.25, 0.50)
    turn_ang_vel_z_end_abs: tuple[float, float] = (0.8, 1.2)
    # backward_only, lateral_only, turn_only, mixed
    flat_locomotion_mode_probabilities: tuple[float, float, float, float] = (0.2, 0.2, 0.2, 0.4)
