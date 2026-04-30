from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terminate_feet_on_base_plane_selected_terrains(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
    restricted_terrain_types: tuple[str, ...] = (
        "stones_2rows",
        "stones_balance",
        "beams_balance",
        "air_beams_balance",
    ),
    force_threshold: float = 1.0,
    plane_height_threshold: float = 0.03,
) -> torch.Tensor:
    """Terminate if feet touch the low base plane on selected terrain types.

    Notes:
    - Terrain assignment in generator mode is column-based (`terrain_types` stores column index).
    - This function reconstructs the curriculum column->sub-terrain mapping from proportions.
    - Base plane contact is approximated using low foot height near env origin z and non-trivial contact force.
    """

    device = env.device
    num_envs = env.scene.num_envs

    terrain_generator_cfg = env.cfg.scene.terrain.terrain_generator
    sub_terrains = terrain_generator_cfg.sub_terrains
    terrain_names = list(sub_terrains.keys())
    if len(terrain_names) == 0:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)

    restricted_name_set = set(restricted_terrain_types)
    restricted_sub_indices = [i for i, name in enumerate(terrain_names) if name in restricted_name_set]
    if len(restricted_sub_indices) == 0:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)

    proportions = torch.tensor(
        [sub_cfg.proportion for sub_cfg in sub_terrains.values()],
        dtype=torch.float32,
        device=device,
    )
    proportions = proportions / torch.sum(proportions)
    cumulative = torch.cumsum(proportions, dim=0)

    terrain_cols = env.scene.terrain.terrain_types.to(device)
    ratios = terrain_cols.float() / float(terrain_generator_cfg.num_cols) + 0.001
    sub_indices = torch.searchsorted(cumulative, ratios, right=False)
    sub_indices = torch.clamp(sub_indices, max=len(terrain_names) - 1)

    restricted_sub_indices_t = torch.tensor(restricted_sub_indices, dtype=torch.long, device=device)
    selected_terrain_mask = torch.any(sub_indices.unsqueeze(1) == restricted_sub_indices_t.unsqueeze(0), dim=1)
    if not torch.any(selected_terrain_mask):
        return torch.zeros(num_envs, dtype=torch.bool, device=device)

    contact_sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    feet_in_contact = forces_z > force_threshold

    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    env_origin_z = env.scene.env_origins[:, 2].unsqueeze(1)
    on_base_plane = (feet_z - env_origin_z) <= plane_height_threshold

    feet_base_plane_contact = feet_in_contact & on_base_plane
    return torch.any(feet_base_plane_contact, dim=1) & selected_terrain_mask
