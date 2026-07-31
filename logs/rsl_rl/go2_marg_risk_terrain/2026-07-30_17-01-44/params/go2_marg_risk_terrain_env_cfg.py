import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.assets.robots.unitree_actuators import UnitreeActuator
from unitree_rl_lab.tasks.locomotion import mdp
from .mgdp_terrain import MGDP_TERRAIN_GENERATOR_CFG


class action_smoothness_l2(ManagerTermBase):
    """Penalize the second-order difference of consecutive policy actions."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._last_last_action = torch.zeros_like(env.action_manager.action)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._last_last_action[env_ids] = 0.0

    def __call__(self, env):
        second_difference = (
            env.action_manager.action
            - 2.0 * env.action_manager.prev_action
            + self._last_last_action
        )
        penalty = torch.sum(torch.square(second_difference), dim=1)
        self._last_last_action.copy_(env.action_manager.prev_action)
        return penalty


def _seed_from_start_minute_second() -> int:
    now = datetime.now()
    return now.minute * 100 + now.second


GO2_MARG_RISK_TERRAIN_SEED = _seed_from_start_minute_second()

GO2_MODIFIED_DESCRIPTION_DIR = Path(__file__).resolve().parents[7] / "LidarSim2Real/go2_urdf_modified"
GO2_MODIFIED_URDF_PATH = GO2_MODIFIED_DESCRIPTION_DIR / "urdf/go2_description.urdf"
GO2_MODIFIED_DAE_DIR = GO2_MODIFIED_DESCRIPTION_DIR / "dae"


def _set_mgdp_terrain_seed(terrain_generator_cfg, seed: int) -> None:
    for sub_cfg in terrain_generator_cfg.sub_terrains.values():
        sub_cfg.seed = seed


def _active_subterrain_count(terrain_generator_cfg) -> int:
    proportions = [float(sub_cfg.proportion) for sub_cfg in terrain_generator_cfg.sub_terrains.values()]
    positive_proportions = [proportion for proportion in proportions if proportion > 0.0]
    if not positive_proportions:
        return 1
    unit_proportion = min(positive_proportions)
    return max(1, int(round(sum(positive_proportions) / unit_proportion)))


def _subterrain_column_indices(terrain_generator_cfg, terrain_name: str) -> list[int]:
    sub_terrains = terrain_generator_cfg.sub_terrains
    names = list(sub_terrains.keys())
    if terrain_name not in names:
        return []

    proportions = [float(sub_cfg.proportion) for sub_cfg in sub_terrains.values()]
    total = sum(proportions)
    if total <= 0.0:
        return []

    cumulative = []
    running = 0.0
    for proportion in proportions:
        running += proportion / total
        cumulative.append(running)

    columns = []
    for col in range(int(terrain_generator_cfg.num_cols)):
        ratio = col / float(terrain_generator_cfg.num_cols) + 0.001
        sub_index = next(i for i, value in enumerate(cumulative) if ratio < value)
        if names[sub_index] == terrain_name:
            columns.append(col)
    return columns


def assign_flat_turn_envs_to_center_column(env, env_ids, terrain_name: str = "flat_turn") -> None:
    terrain = env.scene.terrain
    if terrain.terrain_origins is None or not hasattr(terrain, "terrain_types"):
        return

    flat_columns = _subterrain_column_indices(env.cfg.scene.terrain.terrain_generator, terrain_name)
    if len(flat_columns) <= 1:
        return

    center_column = flat_columns[len(flat_columns) // 2]
    flat_columns_tensor = torch.tensor(flat_columns, dtype=torch.long, device=env.device)
    is_flat_turn = torch.isin(terrain.terrain_types, flat_columns_tensor)
    not_center = terrain.terrain_types != center_column
    remap_env_ids = torch.nonzero(is_flat_turn & not_center, as_tuple=False).flatten()
    if remap_env_ids.numel() == 0:
        return

    terrain.terrain_types[remap_env_ids] = center_column
    terrain.env_origins[remap_env_ids] = terrain.terrain_origins[terrain.terrain_levels[remap_env_ids], center_column]


_set_mgdp_terrain_seed(MGDP_TERRAIN_GENERATOR_CFG, GO2_MARG_RISK_TERRAIN_SEED)
FEET_ON_BASE_PLANE_TERRAINS = tuple(
    name for name in MGDP_TERRAIN_GENERATOR_CFG.sub_terrains.keys() if name != "flat_turn"
)


GO2_MARG_SPAWN_CFG = ROBOT_CFG.spawn.replace(asset_path=str(GO2_MODIFIED_URDF_PATH))
GO2_MARG_SPAWN_CFG.replace_asset(
    meshes_dir=str(GO2_MODIFIED_DAE_DIR),
    urdf_path=str(GO2_MODIFIED_URDF_PATH),
    mesh_link_name="dae",
)


class StartupRandomizedUnitreeActuator(UnitreeActuator):
    """Unitree actuator that preserves its startup-sampled delay across episode resets."""

    def reset(self, env_ids):
        # Clear commands from the previous episode without changing the per-environment time lag.
        self.positions_delay_buffer.reset(env_ids)
        self.velocities_delay_buffer.reset(env_ids)
        self.efforts_delay_buffer.reset(env_ids)


# Allocate actuator command buffers large enough for the startup delay randomization in EventCfg.
ACTUATOR_DELAY_BUFFER_STEPS = 2
GO2_MARG_ROBOT_CFG = ROBOT_CFG.replace(
    spawn=GO2_MARG_SPAWN_CFG,
    actuators={
        "GO2HV": ROBOT_CFG.actuators["GO2HV"].replace(
            class_type=StartupRandomizedUnitreeActuator,
            min_delay=0,
            max_delay=ACTUATOR_DELAY_BUFFER_STEPS,
        )
    },
)


class randomize_rigid_body_material_with_cache(mdp.randomize_rigid_body_material):
    """Randomize contact materials and cache the sampled friction summary."""

    def __call__(
        self,
        env,
        env_ids,
        static_friction_range,
        dynamic_friction_range,
        restitution_range,
        num_buckets,
        asset_cfg,
        make_consistent: bool = False,
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        total_num_shapes = self.asset.root_physx_view.max_shapes
        bucket_ids = torch.randint(0, num_buckets, (len(env_ids), total_num_shapes), device="cpu")
        material_samples = self.material_buckets[bucket_ids]

        materials = self.asset.root_physx_view.get_material_properties()
        if self.num_shapes_per_body is not None:
            for body_id in self.asset_cfg.body_ids:
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                materials[env_ids, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
        else:
            materials[env_ids] = material_samples[:]
        self.asset.root_physx_view.set_material_properties(materials, env_ids)

        if not hasattr(env, "_terrain_friction"):
            env._terrain_friction = torch.full(
                (env.scene.num_envs, 1),
                env.cfg.scene.terrain.physics_material.static_friction,
                device=env.device,
            )
        env._terrain_friction[env_ids.to(env.device)] = material_samples[..., 0].mean(dim=1, keepdim=True).to(env.device)


def randomize_motor_strength(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    strength_distribution_params: tuple[float, float],
):
    """Randomize motor strength factors and apply them to actuator torque limits."""

    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if asset_cfg.joint_ids == slice(None):
        global_joint_ids = torch.arange(asset.num_joints, device=asset.device)
    else:
        global_joint_ids = torch.tensor(asset_cfg.joint_ids, device=asset.device)

    if not hasattr(env, "_motor_strength"):
        env._motor_strength = torch.ones((env.scene.num_envs, asset.num_joints), device=asset.device)
    if not hasattr(env, "_motor_offset"):
        env._motor_offset = torch.zeros((env.scene.num_envs, asset.num_joints), device=asset.device)

    sampled_strength = torch.empty((len(env_ids), len(global_joint_ids)), device=asset.device).uniform_(
        strength_distribution_params[0], strength_distribution_params[1]
    )
    env._motor_strength[env_ids[:, None], global_joint_ids] = sampled_strength

    for actuator in asset.actuators.values():
        if isinstance(actuator.joint_indices, slice):
            actuator_global_ids = torch.arange(actuator.num_joints, device=asset.device)
        else:
            actuator_global_ids = torch.tensor(actuator.joint_indices, device=asset.device)

        local_mask = torch.isin(actuator_global_ids, global_joint_ids)
        if not torch.any(local_mask):
            continue

        local_ids = torch.nonzero(local_mask).view(-1)
        selected_global_ids = actuator_global_ids[local_ids]
        selected_strength = env._motor_strength[env_ids][:, selected_global_ids]

        if hasattr(actuator, "_effort_y1"):
            if not hasattr(actuator, "_default_effort_y1"):
                actuator._default_effort_y1 = actuator._effort_y1.clone()
                actuator._default_effort_y2 = actuator._effort_y2.clone()
            actuator._effort_y1[env_ids[:, None], local_ids] = (
                actuator._default_effort_y1[env_ids[:, None], local_ids] * selected_strength
            )
            actuator._effort_y2[env_ids[:, None], local_ids] = (
                actuator._default_effort_y2[env_ids[:, None], local_ids] * selected_strength
            )

        if hasattr(actuator, "effort_limit"):
            if not hasattr(actuator, "_default_effort_limit"):
                actuator._default_effort_limit = actuator.effort_limit.clone()
            actuator.effort_limit[env_ids[:, None], local_ids] = (
                actuator._default_effort_limit[env_ids[:, None], local_ids] * selected_strength
            )


def randomize_actuator_delay(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    delay_range_steps: tuple[int, int],
):
    """Randomize actuator command delay in integer physics steps at startup."""

    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    else:
        env_ids = env_ids.to(asset.device)

    min_delay, max_delay = delay_range_steps
    if min_delay < 0 or max_delay < min_delay:
        raise ValueError(f"Invalid actuator delay range: {delay_range_steps}")

    sampled_delay = torch.randint(
        low=min_delay,
        high=max_delay + 1,
        size=(len(env_ids),),
        dtype=torch.int,
        device=asset.device,
    )

    for actuator_name, actuator in asset.actuators.items():
        delay_buffers = (
            getattr(actuator, "positions_delay_buffer", None),
            getattr(actuator, "velocities_delay_buffer", None),
            getattr(actuator, "efforts_delay_buffer", None),
        )
        if any(buffer is None for buffer in delay_buffers):
            raise TypeError(f"Actuator '{actuator_name}' does not provide delayed command buffers.")
        if any(buffer.history_length < max_delay for buffer in delay_buffers):
            raise ValueError(
                f"Actuator '{actuator_name}' delay buffer is shorter than requested max delay {max_delay}."
            )
        for buffer in delay_buffers:
            buffer.set_time_lag(sampled_delay, env_ids)
            buffer.reset(env_ids)

    if not hasattr(env, "_actuator_delay_steps"):
        env._actuator_delay_steps = torch.zeros(env.scene.num_envs, dtype=torch.int, device=asset.device)
    env._actuator_delay_steps[env_ids] = sampled_delay


def reset_base_with_terrain_orientation(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
):
    """Reset base position and orientation for directional MGDP terrains.

    The robot's initial yaw is aligned to +x direction with ±5° deviation.
    Position offset is ±10cm from spawn center in xy plane.
    """
    asset = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    else:
        env_ids = env_ids.to(asset.device)

    num_envs = len(env_ids)
    root_states = asset.data.default_root_state[env_ids].clone()
    pos_offsets = torch.zeros((num_envs, 3), device=asset.device)
    velocities = root_states[:, 7:13].clone()

    angle_tolerance = 5.0 * math.pi / 180.0
    yaws = torch.empty((num_envs,), device=asset.device).uniform_(-angle_tolerance, angle_tolerance)
    pos_offsets[:, 0:2] = torch.empty((num_envs, 2), device=asset.device).uniform_(-0.1, 0.1)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + pos_offsets
    orientations = math_utils.quat_from_euler_xyz(
        torch.zeros_like(yaws),
        torch.zeros_like(yaws),
        yaws,
    )

    # Apply root state through Articulation APIs.
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})

    stationary = DoneTerm(
        func=mdp.terminate_stationary_for_duration,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "duration": 1.0,
            "distance_threshold": 1.50,
            "command_speed_threshold": 0.05,
        },
    )

    feet_on_base_plane_linear = DoneTerm(
        func=mdp.terminate_feet_on_base_plane_selected_terrains,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "restricted_terrain_types": FEET_ON_BASE_PLANE_TERRAINS,
            "force_threshold": 1.0,
            "plane_height_threshold": -0.5,
        },
    )


# =========================== Domain Randomization ===================
# ====================================================================
@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    flat_turn_center_column = EventTerm(
        func=assign_flat_turn_envs_to_center_column,
        mode="startup",
        params={
            "terrain_name": "flat_turn",
        },
    )

    actuator_delay = EventTerm(
        func=randomize_actuator_delay,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            # With sim dt = 0.005 s, 0..2 physics steps corresponds to 0..10 ms.
            "delay_range_steps": (0, 2),
        },
    )

    physics_material = EventTerm(
        func=randomize_rigid_body_material_with_cache,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.25),
            "dynamic_friction_range": (0.2, 1.25),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1), # Kp
            "damping_distribution_params": (0.9, 1.1), # Kd
            "operation": "scale",
        },
    )

    base_com_shift = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    motor_strength = EventTerm(
        func=randomize_motor_strength,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "strength_distribution_params": (0.9, 1.1),
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=reset_base_with_terrain_orientation,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


# =========================== Scene Config ===========================
# ====================================================================
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Scene config for the Go2 Marg Risk Terrain task."""

    num_envs: int = 4096
    env_spacing: float = 2.5

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=MGDP_TERRAIN_GENERATOR_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = GO2_MARG_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# =========================== Command Space ===============================
# =========================================================================
# Exposed command interface for this training task:
# parkour terrains stay near-forward-only, while the center flat_turn column trains flat-ground locomotion.
FORWARD_ONLY_LIN_VEL_X = (0.1, 1.0)
FORWARD_ONLY_LIN_VEL_X_LIMIT = (0.1, 1.5)
FORWARD_ONLY_LIN_VEL_Y = (-0.01, 0.01)
FORWARD_ONLY_ANG_VEL_Z = (-0.01, 0.01)
FLAT_LOCOMOTION_LIN_VEL_X_START = (-0.3, 0.6)
FLAT_LOCOMOTION_LIN_VEL_X_END = (-0.6, 1.0)
FLAT_LOCOMOTION_LIN_VEL_Y_START_ABS = (0.0, 0.2)
FLAT_LOCOMOTION_LIN_VEL_Y_END_ABS = (0.0, 0.5)
FLAT_LOCOMOTION_ANG_VEL_Z_START_ABS = (0.15, 0.5)
FLAT_LOCOMOTION_ANG_VEL_Z_END_ABS = (0.0, 1.2)
FLAT_LOCOMOTION_MODE_PROBABILITIES = (0.2, 0.2, 0.2, 0.4)  # backward, lateral, turn, mixed


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.FlatTurnVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=FORWARD_ONLY_LIN_VEL_X,
            lin_vel_y=FORWARD_ONLY_LIN_VEL_Y,
            ang_vel_z=FORWARD_ONLY_ANG_VEL_Z,
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=FORWARD_ONLY_LIN_VEL_X_LIMIT,
            lin_vel_y=FORWARD_ONLY_LIN_VEL_Y,
            ang_vel_z=FORWARD_ONLY_ANG_VEL_Z,
        ),
        turn_lin_vel_x_start=FLAT_LOCOMOTION_LIN_VEL_X_START,
        turn_lin_vel_x_end=FLAT_LOCOMOTION_LIN_VEL_X_END,
        turn_lin_vel_y_start_abs=FLAT_LOCOMOTION_LIN_VEL_Y_START_ABS,
        turn_lin_vel_y_end_abs=FLAT_LOCOMOTION_LIN_VEL_Y_END_ABS,
        turn_ang_vel_z_start_abs=FLAT_LOCOMOTION_ANG_VEL_Z_START_ABS,
        turn_ang_vel_z_end_abs=FLAT_LOCOMOTION_ANG_VEL_Z_END_ABS,
        flat_locomotion_mode_probabilities=FLAT_LOCOMOTION_MODE_PROBABILITIES,
    )


# =========================== Observation Space ===========================
# =========================================================================
@configclass
class ObservationsCfg:
    """Observation layout for the Go2 MARG task."""

    @configclass
    class ProprioObsCfg(ObsGroup):
        """45D proprioceptive observation."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ProprioHistoryObsCfg(ProprioObsCfg):
        """5(+1)-step proprio history, flattened to 270D."""

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5 + 1  # include current step
            self.flatten_history_dim = True

    @configclass
    class TerrainMapObsCfg(ObsGroup):
        """187D terrain map."""

        terrain_map = ObsTerm(
            func=mdp.terrain_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "asset_cfg": SceneEntityCfg("robot")},
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedObsCfg(ObsGroup):
        """Privileged observations set used by the critic / auxiliary estimators."""

        real_linear_velocity = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        feet_contacts = ObsTerm(
            func=mdp.feet_contact_labels,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 1.0},
        )
        critical_masses = ObsTerm(func=mdp.critical_mass_summary, params={"asset_cfg": SceneEntityCfg("robot")})
        friction = ObsTerm(func=mdp.terrain_friction_label)
        com_shift = ObsTerm(func=mdp.base_com_shift_xy, params={"asset_cfg": SceneEntityCfg("robot")})
        disturbance_force = ObsTerm(
            func=mdp.disturbance_force_xoy,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        actuator_params = ObsTerm(
            func=mdp.actuator_params_26,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )

        def __post_init__(self):
            self.concatenate_terms = True

    privileged_obs: PrivilegedObsCfg = PrivilegedObsCfg()

    @configclass
    class PolicyRawObsCfg(ProprioObsCfg):
        """Current policy raw obs, same as proprio obs."""

    policy_raw_obs: PolicyRawObsCfg = PolicyRawObsCfg()

    @configclass
    class PolicyHistoryObsCfg(ProprioHistoryObsCfg):
        """Current policy history obs, same as proprio history obs."""

    policy_history_obs: PolicyHistoryObsCfg = PolicyHistoryObsCfg()

    @configclass
    class PolicyTerrainObsCfg(TerrainMapObsCfg):
        """Current policy terrain obs, same as terrain map obs."""

    policy_terrain_obs: PolicyTerrainObsCfg = PolicyTerrainObsCfg()

    @configclass
    class CriticObsCfg(ProprioObsCfg):
        """Critic observation: proprio + terrain + privileged."""

        # terrain map is included in the critic obs
        terrain_map = ObsTerm(
            func=mdp.terrain_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "asset_cfg": SceneEntityCfg("robot")},
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        # privileged terms
        real_linear_velocity = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        feet_contacts = ObsTerm(
            func=mdp.feet_contact_labels,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 1.0},
        )
        critical_masses = ObsTerm(func=mdp.critical_mass_summary, params={"asset_cfg": SceneEntityCfg("robot")})
        friction = ObsTerm(func=mdp.terrain_friction_label)
        com_shift = ObsTerm(func=mdp.base_com_shift_xy, params={"asset_cfg": SceneEntityCfg("robot")})
        disturbance_force = ObsTerm(
            func=mdp.disturbance_force_xoy,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        actuator_params = ObsTerm(
            func=mdp.actuator_params_26,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic_obs: CriticObsCfg = CriticObsCfg()

    @configclass
    class PolicyCfg(PolicyRawObsCfg):
        """Compatibility group required by current RL wrappers."""

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(CriticObsCfg):
        """Compatibility group required by current RL wrappers."""

    critic: CriticCfg = CriticCfg()


# =========================== Action Space ================================
# =========================================================================
@configclass
class ActionsCfg:
    """12D joint action space for Go2 locomotion."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )


# =========================== Reward Config ===============================
# =========================================================================
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-1.0,
        params={"command_name": "base_velocity", "cmd_threshold": 0.05, "excluded_terrain_names": ("flat_turn",)},
    )
    a_track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    a_track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # -- smoothness
    base_linear_velocity_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness = RewTerm(func=action_smoothness_l2, weight=-0.01)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)

    # -- safety
    collisions = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]),
        },
    )

    # -- pose
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.2)
    joint_motion_limit = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    # -- footholds
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_low_speed_gating,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
            "speed_threshold": 0.1,
        },
    )
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-3.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    feet_center = RewTerm(
        func=mdp.feet_center,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
            ),
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "debug_vis": False,
            "debug_env_count": 1,
        },
    )


# =========================== Curriculum Config =============================
# =========================================================================
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel,
        params={"excluded_terrain_names": ("flat_turn",)},
    )
    flat_turn_terrain_levels = CurrTerm(
        func=mdp.flat_turn_terrain_levels,
        params={
            "terrain_names": ("flat_turn",),
            "reward_term_name": "a_track_ang_vel_z",
            "lin_reward_term_name": "a_track_lin_vel_xy",
        },
    )
    lin_vel_cmd_levels = CurrTerm(
        func=mdp.lin_vel_cmd_levels,
        params={
            "reward_term_name": "a_track_lin_vel_xy",
            "lin_vel_x_delta": (0.1, 0.1),
            "lin_vel_y_delta": (0.0, 0.0),
        },
    )


# =========================== Task & Play Config ==========================
# =========================================================================
@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Go2 Marg task config."""

    scene: RobotSceneCfg = RobotSceneCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # MGDP heightfield terrains create denser contact patches than the box-based
        # risk terrains, so the default 2**26 collision stack can overflow on GPU.
        self.sim.physx.gpu_collision_stack_size = max(self.sim.physx.gpu_collision_stack_size, 2**27)

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        self.scene.terrain.terrain_generator.num_rows = 10  # terrain levels
        self.scene.terrain.terrain_generator.num_cols = _active_subterrain_count(self.scene.terrain.terrain_generator)


PLAY_TERRAIN_TYPE = "mgdp"


def _play_terrain_generator_cfg(terrain_type: str):
    from .test_terrain import TEST_TERRAIN_GENERATOR_CFG

    terrain_generator_cfgs = {
        "mgdp": MGDP_TERRAIN_GENERATOR_CFG,
        "test": TEST_TERRAIN_GENERATOR_CFG,
    }
    terrain_type = terrain_type.strip().lower()
    if terrain_type not in terrain_generator_cfgs:
        valid_names = ", ".join(sorted(terrain_generator_cfgs))
        raise ValueError(f"Unknown play terrain type '{terrain_type}'. Valid options: {valid_names}.")
    return terrain_type, deepcopy(terrain_generator_cfgs[terrain_type])


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """Play config for the Go2 Marg risk terrain task."""

    play_terrain_type: str = PLAY_TERRAIN_TYPE

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        play_terrain_type, terrain_generator_cfg = _play_terrain_generator_cfg(self.play_terrain_type)
        self.scene.terrain.terrain_generator = terrain_generator_cfg
        if play_terrain_type == "test":
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.num_rows = 3
        self.scene.terrain.terrain_generator.num_cols = _active_subterrain_count(self.scene.terrain.terrain_generator)
        self.commands.base_velocity.ranges = deepcopy(self.commands.base_velocity.limit_ranges)
        self.observations.policy_terrain_obs.enable_corruption = False
        self.observations.policy_raw_obs.enable_corruption = False
        self.events.push_robot = None
        # self.terminations.feet_on_base_plane_linear = None
        self.rewards.feet_center.params["debug_vis"] = True
        self.rewards.feet_center.params["debug_env_count"] = 1
