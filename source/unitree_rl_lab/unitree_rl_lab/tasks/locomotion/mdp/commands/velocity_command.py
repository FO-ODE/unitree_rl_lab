from __future__ import annotations

from dataclasses import MISSING
from collections.abc import Sequence

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class TerrainAwareUniformVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command with optional terrain-type-specific y/yaw restriction."""

    cfg: "UniformLevelVelocityCommandCfg"

    def __init__(self, cfg: "UniformLevelVelocityCommandCfg", env):
        super().__init__(cfg, env)
        self._restricted_terrain_cols: torch.Tensor | None = None

        terrain_cfg = getattr(getattr(env.cfg, "scene", None), "terrain", None)
        terrain_generator_cfg = getattr(terrain_cfg, "terrain_generator", None)
        sub_terrains = getattr(terrain_generator_cfg, "sub_terrains", None)
        num_cols = getattr(terrain_generator_cfg, "num_cols", None)
        restricted_names = set(self.cfg.restricted_terrain_types)

        if sub_terrains is not None and restricted_names and num_cols is not None:
            # In curriculum terrains, terrain.terrain_types stores column indices (0..num_cols-1),
            # not direct sub-terrain indices. Rebuild the same column->subterrain mapping as generator.
            sub_terrain_names = list(sub_terrains.keys())
            proportions = torch.tensor(
                [sub_cfg.proportion for sub_cfg in sub_terrains.values()],
                dtype=torch.float32,
                device=self.device,
            )
            proportions = proportions / torch.sum(proportions)
            cumulative = torch.cumsum(proportions, dim=0)

            restricted_cols = []
            for col in range(int(num_cols)):
                ratio = col / float(num_cols) + 0.001
                sub_index = int(torch.searchsorted(cumulative, torch.tensor(ratio, device=self.device), right=False))
                sub_index = min(sub_index, len(sub_terrain_names) - 1)
                if sub_terrain_names[sub_index] in restricted_names:
                    restricted_cols.append(col)

            if restricted_cols:
                self._restricted_terrain_cols = torch.tensor(restricted_cols, dtype=torch.long, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

        if self._restricted_terrain_cols is None or len(env_ids) == 0:
            return

        env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        terrain_types = self._env.scene.terrain.terrain_types[env_ids_tensor]

        restricted_mask = torch.zeros(len(env_ids_tensor), dtype=torch.bool, device=self.device)
        for terrain_col in self._restricted_terrain_cols:
            restricted_mask |= terrain_types == terrain_col

        if not torch.any(restricted_mask):
            return

        restricted_env_ids = env_ids_tensor[restricted_mask]
        r = torch.empty(len(restricted_env_ids), device=self.device)

        # Only constrain sideways and yaw commands on selected terrain types.
        self.vel_command_b[restricted_env_ids, 1] = r.uniform_(*self.cfg.restricted_ranges.lin_vel_y)
        self.vel_command_b[restricted_env_ids, 2] = r.uniform_(*self.cfg.restricted_ranges.ang_vel_z)

        # Keep heading mode from overriding restricted yaw command.
        if self.cfg.heading_command:
            self.is_heading_env[restricted_env_ids] = False


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = TerrainAwareUniformVelocityCommand

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    restricted_terrain_types: tuple[str, ...] = ()
    restricted_ranges: UniformVelocityCommandCfg.Ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-0.01, 0.01),
        ang_vel_z=(-0.01, 0.01),
    )
