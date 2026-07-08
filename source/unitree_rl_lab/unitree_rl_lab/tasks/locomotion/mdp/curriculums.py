from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    lin_vel_x_delta: tuple[float, float] = (-0.1, 0.1),
    lin_vel_y_delta: tuple[float, float] = (-0.1, 0.1),
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            ranges.lin_vel_x = _update_range_towards_limit(
                ranges.lin_vel_x, lin_vel_x_delta, limit_ranges.lin_vel_x, env.device
            )
            ranges.lin_vel_y = _update_range_towards_limit(
                ranges.lin_vel_y, lin_vel_y_delta, limit_ranges.lin_vel_y, env.device
            )

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def _update_range_towards_limit(
    current: tuple[float, float] | list[float],
    delta: tuple[float, float],
    limit: tuple[float, float],
    device: torch.device | str,
) -> list[float]:
    current_tensor = torch.tensor(current, device=device)
    delta_tensor = torch.tensor(delta, device=device)
    limit_tensor = torch.tensor(limit, device=device)
    next_tensor = current_tensor + delta_tensor

    lower = _move_endpoint_towards_limit(next_tensor[0], delta[0], limit_tensor[0])
    upper = _move_endpoint_towards_limit(next_tensor[1], delta[1], limit_tensor[1])
    return torch.stack([lower, upper]).tolist()


def _move_endpoint_towards_limit(value: torch.Tensor, delta: float, limit: torch.Tensor) -> torch.Tensor:
    if delta > 0.0:
        return torch.minimum(value, limit)
    if delta < 0.0:
        return torch.maximum(value, limit)
    return value


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    excluded_terrain_names: tuple[str, ...] = (),
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    if excluded_terrain_names:
        env_ids = env_ids[~_terrain_name_mask(env, excluded_terrain_names, env_ids)]
        if env_ids.numel() == 0:
            return torch.mean(terrain.terrain_levels.float())

    command = env.command_manager.get_command("base_velocity")
    # Compute the distance walked relative to the reset pose inside each terrain tile.
    # This mirrors MGDP's gap-parkour curriculum, where the configured initial x/y offset
    # is subtracted before checking progress.
    root_pos_in_terrain = asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    init_pos_in_terrain = asset.data.default_root_state[env_ids, :2]
    distance = torch.norm(root_pos_in_terrain - init_pos_in_terrain, dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2.0
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def flat_turn_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    terrain_names: tuple[str, ...] = ("flat_turn",),
    reward_term_name: str = "a_track_ang_vel_z",
    lin_reward_term_name: str | None = None,
    success_ratio: float = 0.75,
    failure_ratio: float = 0.45,
) -> torch.Tensor:
    """Curriculum for flat locomotion terrains based on velocity tracking."""
    terrain: TerrainImporter = env.scene.terrain
    if terrain.terrain_origins is None:
        return torch.mean(terrain.terrain_levels.float())

    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    flat_env_ids = env_ids[_terrain_name_mask(env, terrain_names, env_ids)]
    if flat_env_ids.numel() == 0:
        return torch.mean(terrain.terrain_levels.float())

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = env.reward_manager._episode_sums[reward_term_name][flat_env_ids] / env.max_episode_length_s
    success = reward > reward_term.weight * success_ratio
    failure = reward < reward_term.weight * failure_ratio

    if lin_reward_term_name is not None:
        lin_reward_term = env.reward_manager.get_term_cfg(lin_reward_term_name)
        lin_reward = env.reward_manager._episode_sums[lin_reward_term_name][flat_env_ids] / env.max_episode_length_s
        success &= lin_reward > lin_reward_term.weight * success_ratio
        failure |= lin_reward < lin_reward_term.weight * failure_ratio

    move_up = success
    move_down = failure
    move_down *= ~move_up
    terrain.update_env_origins(flat_env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def _terrain_name_mask(
    env: ManagerBasedRLEnv, terrain_names: tuple[str, ...], env_ids: torch.Tensor | None = None
) -> torch.Tensor:
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "terrain_types"):
        size = env.num_envs if env_ids is None else env_ids.numel()
        return torch.zeros(size, dtype=torch.bool, device=env.device)

    generator_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    if generator_cfg is None or not getattr(generator_cfg, "sub_terrains", None):
        size = env.num_envs if env_ids is None else env_ids.numel()
        return torch.zeros(size, dtype=torch.bool, device=env.device)

    names = list(generator_cfg.sub_terrains.keys())
    proportions = [float(sub_cfg.proportion) for sub_cfg in generator_cfg.sub_terrains.values()]
    total = sum(proportions)
    if total <= 0.0:
        size = env.num_envs if env_ids is None else env_ids.numel()
        return torch.zeros(size, dtype=torch.bool, device=env.device)

    cumulative = []
    running = 0.0
    for proportion in proportions:
        running += proportion / total
        cumulative.append(running)

    selected_type_ids = []
    selected_names = set(terrain_names)
    for col in range(int(generator_cfg.num_cols)):
        threshold = col / float(generator_cfg.num_cols) + 0.001
        sub_index = next(i for i, value in enumerate(cumulative) if threshold < value)
        if names[sub_index] in selected_names:
            selected_type_ids.append(col)

    if env_ids is None:
        target_types = terrain.terrain_types
    else:
        target_types = terrain.terrain_types[env_ids]
    if not selected_type_ids:
        return torch.zeros(target_types.shape[0], dtype=torch.bool, device=env.device)
    type_ids = torch.tensor(selected_type_ids, dtype=torch.long, device=env.device)
    return torch.isin(target_types, type_ids)
