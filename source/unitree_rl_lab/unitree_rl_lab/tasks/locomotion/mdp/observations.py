from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase



# ======================= go2 Observation Space ===========================
# =========================================================================

def oracle_terrain_map(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot-centered oracle terrain map as relative heights: base_z - terrain_z."""

    sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2]


def feet_contact_labels(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """4D foot-contact label from the vertical contact force.

    A foot is marked as in contact when |f_z| > threshold.
    """

    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    return (forces_z > threshold).float()


def critical_mass_summary(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """4D critical mass summary.

    Layout:
    - m0: base / trunk mass
    - m1: mean thigh mass
    - m2: mean calf mass
    - m3: added payload mass on the base relative to the default mass
    """

    asset = env.scene[asset_cfg.name]
    device = env.device
    masses = asset.root_physx_view.get_masses().to(device)

    cache_name = "_critical_mass_body_ids"
    if not hasattr(env, cache_name):
        base_ids, _ = asset.find_bodies("base")
        thigh_ids, _ = asset.find_bodies(".*_thigh")
        calf_ids, _ = asset.find_bodies(".*_calf")
        setattr(
            env,
            cache_name,
            {
                "base": base_ids,
                "thigh": thigh_ids,
                "calf": calf_ids,
            },
        )
    body_ids = getattr(env, cache_name)

    base_mass = masses[:, body_ids["base"]].sum(dim=1, keepdim=True)
    thigh_mass = masses[:, body_ids["thigh"]].mean(dim=1, keepdim=True)
    calf_mass = masses[:, body_ids["calf"]].mean(dim=1, keepdim=True)

    if hasattr(asset.data, "default_mass") and asset.data.default_mass is not None:
        default_base_mass = asset.data.default_mass[:, body_ids["base"]].sum(dim=1, keepdim=True).to(device)
        added_base_mass = base_mass - default_base_mass
    else:
        added_base_mass = torch.zeros_like(base_mass)

    return torch.cat((base_mass, thigh_mass, calf_mass, added_base_mass), dim=1)


def terrain_friction_label(env) -> torch.Tensor:
    """1D terrain friction label.

    This must come from env state populated by the terrain material randomization.
    """

    if hasattr(env, "_terrain_friction"):
        return env._terrain_friction
    raise RuntimeError("terrain_friction_label requires env._terrain_friction from material randomization")


def base_com_shift_xy(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base COM xy shift in the base frame."""

    asset = env.scene[asset_cfg.name]
    return asset.data.body_com_pos_b[:, 0, :2]


def disturbance_force_xoy(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Applied disturbance force on the base body in x-y."""

    asset = env.scene[asset_cfg.name]
    return asset._external_force_b[:, asset_cfg.body_ids[0], :2]


def actuator_params_26(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """26D actuator/state parameter vector.

    Layout:
    - 1 kp
    - 1 kd
    - 12 motor strength values
    - 12 motor offset values
    """

    asset = env.scene[asset_cfg.name]

    if hasattr(asset.data, "joint_stiffness") and hasattr(asset.data, "joint_damping"):
        kp = asset.data.joint_stiffness[:, asset_cfg.joint_ids].mean(dim=1, keepdim=True)
        kd = asset.data.joint_damping[:, asset_cfg.joint_ids].mean(dim=1, keepdim=True)
    else:
        kp = asset.data.default_joint_stiffness[:, asset_cfg.joint_ids].mean(dim=1, keepdim=True)
        kd = asset.data.default_joint_damping[:, asset_cfg.joint_ids].mean(dim=1, keepdim=True)

    if hasattr(env, "_motor_strength"):
        motor_strength = env._motor_strength[:, asset_cfg.joint_ids]
    else:
        effort = asset.data.joint_effort_limits[:, asset_cfg.joint_ids]
        motor_strength = effort / effort.mean(dim=1, keepdim=True)

    if hasattr(env, "_motor_offset"):
        motor_offset = env._motor_offset[:, asset_cfg.joint_ids]
    else:
        motor_offset = torch.zeros_like(motor_strength)

    return torch.cat((kp, kd, motor_strength, motor_offset), dim=1)
