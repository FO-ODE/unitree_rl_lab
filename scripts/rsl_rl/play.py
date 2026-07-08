# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
from importlib.metadata import version
import pathlib
import sys

from isaaclab.app import AppLauncher

# Import task packages early so custom workspace tasks are registered for --task in play mode.
sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Use keyboard teleop for base velocity.")
parser.add_argument("--keyboard_vx", type=float, default=1.0, help="Keyboard forward/backward velocity scale.")
parser.add_argument("--keyboard_vy", type=float, default=1.0, help="Keyboard lateral velocity scale.")
parser.add_argument("--keyboard_wz", type=float, default=1.0, help="Keyboard yaw velocity scale.")
parser.add_argument("--keyboard_speed_step", type=float, default=0.1, help="Keyboard speed change per A/D press.")
parser.add_argument("--keyboard_speed_min", type=float, default=0.0, help="Minimum keyboard command speed scale.")
parser.add_argument("--keyboard_speed_max", type=float, default=3.0, help="Maximum keyboard command speed scale.")
parser.add_argument("--keyboard_smoothing", type=float, default=0.3, help="Low-pass factor for keyboard commands.")
parser.add_argument("--follow_camera_distance", type=float, default=3, help="Follow camera distance behind robot.")
parser.add_argument("--follow_camera_height", type=float, default=1.4, help="Follow camera height above robot.")
parser.add_argument("--follow_camera_yaw", type=float, default=-30.0, help="Follow camera yaw offset in degrees.")
parser.add_argument("--follow_camera_target_x", type=float, default=1.0, help="Look-ahead target offset in robot x axis.")
parser.add_argument("--follow_camera_target_z", type=float, default=0.35, help="Look-at target height offset.")
parser.add_argument(
    "--side_push",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Apply a periodic lateral push disturbance to the robot base COM during play.",
)
parser.add_argument("--no_side_push", dest="side_push", action="store_false", help="Disable the lateral push disturbance.")
parser.add_argument("--side_push_force", type=float, default=100.0, help="Lateral push force magnitude in Newtons.")
parser.add_argument("--side_push_duration", type=float, default=0.1, help="Duration of each lateral push in seconds.")
parser.add_argument("--side_push_period", type=float, default=10.0, help="Period between lateral pushes in seconds.")
parser.add_argument(
    "--side_push_start_time",
    type=float,
    default=10.0,
    help="Simulation time of the first lateral push in seconds.",
)
parser.add_argument(
    "--side_push_direction_y",
    type=float,
    default=-1.0,
    help="Push direction along robot base Y. Use -1 for left-to-right in the usual x-forward/y-left base frame.",
)
parser.add_argument(
    "--side_push_arrow_length",
    type=float,
    default=0.7,
    help="Viewport arrow length used to visualize the lateral push.",
)
parser.add_argument(
    "--side_push_arrow_z_offset",
    type=float,
    default=0.15,
    help="Vertical offset from the base COM for the lateral push arrow visualization.",
)
parser.add_argument(
    "--jit_policy",
    type=str,
    default=None,
    help="Path to a TorchScript policy.pt. For MARG-Oracle this expects policy_raw/history/terrain obs groups.",
)
parser.add_argument(
    "--policy_obs_visualizer",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Show a live matplotlib view of policy raw/history/terrain observations and actions.",
)
parser.add_argument("--policy_obs_refresh_hz", type=float, default=10.0, help="Policy observation visualizer refresh rate.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.keyboard:
    args_cli.num_envs = 1
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import importlib
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_apply_yaw
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaacsim.core.utils.viewports import set_camera_view

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


_MARG_ORACLE_TASK = "Unitree-Go2-MARG-Oracle-Risk-Terrain"
_POLICY_RAW_DIM = 45
_POLICY_HISTORY_FRAMES = 6
_POLICY_TERRAIN_ROWS = 11
_POLICY_TERRAIN_COLS = 17
_POLICY_ACTION_DIM = 12
_POLICY_TERM_LAYOUT = (
    ("ang_vel", 3),
    ("gravity", 3),
    ("cmd", 3),
    ("joint_pos", 12),
    ("joint_vel", 12),
    ("last_action", 12),
)
_POLICY_RAW_LABELS = (
    ["wx", "wy", "wz"]
    + ["gx", "gy", "gz"]
    + ["cmd_x", "cmd_y", "cmd_wz"]
    + [f"q{i}" for i in range(12)]
    + [f"dq{i}" for i in range(12)]
    + [f"a{i}" for i in range(12)]
)
if args_cli.policy_obs_visualizer is None:
    args_cli.policy_obs_visualizer = bool(args_cli.keyboard and args_cli.task == _MARG_ORACLE_TASK)


def _import_class(import_path: str):
    module_name, class_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _set_base_velocity_command(env, command: torch.Tensor | None = None):
    """Set the command source used by generated_commands observations."""
    if command is None:
        return

    command = command.to(env.unwrapped.device)
    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    command_term.vel_command_b[:, :] = command
    is_standing = torch.linalg.norm(command) < 1.0e-4
    if hasattr(command_term, "is_standing_env"):
        command_term.is_standing_env[:] = is_standing
    if hasattr(command_term, "is_heading_env"):
        command_term.is_heading_env[:] = False
    if hasattr(command_term, "time_left"):
        command_term.time_left[:] = command_term.cfg.resampling_time_range[1]


def _compute_observations(env, update_history: bool = False) -> tuple[torch.Tensor, dict]:
    """Compute observations from the current command source."""
    obs_dict = env.unwrapped.observation_manager.compute(update_history=update_history)
    return obs_dict["policy"], {"observations": obs_dict}


def _load_marg_oracle_jit_policy(policy_path: str, device: str | torch.device):
    jit_policy = torch.jit.load(policy_path, map_location=device).eval()

    def policy(obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        missing = [
            key
            for key in ("policy_raw_obs", "policy_history_obs", "policy_terrain_obs")
            if key not in obs_dict
        ]
        if missing:
            raise KeyError(f"TorchScript policy requires missing observation group(s): {missing}")

        raw_obs = obs_dict["policy_raw_obs"].to(device=device, dtype=torch.float32)
        history_obs = obs_dict["policy_history_obs"].to(device=device, dtype=torch.float32)
        terrain_obs = obs_dict["policy_terrain_obs"].to(device=device, dtype=torch.float32)
        expected_dims = {
            "policy_raw_obs": (raw_obs, _POLICY_RAW_DIM),
            "policy_history_obs": (history_obs, _POLICY_RAW_DIM * _POLICY_HISTORY_FRAMES),
            "policy_terrain_obs": (terrain_obs, _POLICY_TERRAIN_ROWS * _POLICY_TERRAIN_COLS),
        }
        for name, (tensor, expected_dim) in expected_dims.items():
            if tensor.shape[-1] != expected_dim:
                raise ValueError(f"{name} must have last dimension {expected_dim}, got {tuple(tensor.shape)}")

        return jit_policy(raw_obs, history_obs, terrain_obs)

    return policy


class PolicyObsVisualizer:
    """Live matplotlib view of the observations fed to the MARG-Oracle policy."""

    def __init__(self, refresh_hz: float):
        import matplotlib.pyplot as plt
        import numpy as np

        self.plt = plt
        self.np = np
        self.refresh_dt = 1.0 / max(refresh_hz, 1.0e-3)
        self.last_draw_time = 0.0
        plt.ion()

        self.fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        self.raw_ax, self.history_ax, self.terrain_ax, self.action_ax = axes.ravel()
        self.fig.suptitle("IsaacLab policy observation monitor")

        raw_x = np.arange(_POLICY_RAW_DIM)
        self.raw_bars = self.raw_ax.bar(raw_x, np.zeros(_POLICY_RAW_DIM))
        self.raw_ax.set_title("policy_raw_obs")
        self.raw_ax.set_xticks(raw_x)
        self.raw_ax.set_xticklabels(_POLICY_RAW_LABELS, rotation=90, fontsize=7)
        for boundary in (3, 6, 9, 21, 33):
            self.raw_ax.axvline(boundary - 0.5, color="0.7", linewidth=0.8)

        self.history_img = self.history_ax.imshow(
            np.zeros((_POLICY_HISTORY_FRAMES, _POLICY_RAW_DIM)),
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
        )
        self.history_ax.set_title("policy_history_obs as 6 x 45")
        self.history_ax.set_xlabel("raw obs index")
        self.history_ax.set_ylabel("history frame: old -> new")
        self.fig.colorbar(self.history_img, ax=self.history_ax, fraction=0.046, pad=0.04)

        self.terrain_img = self.terrain_ax.imshow(
            np.zeros((_POLICY_TERRAIN_ROWS, _POLICY_TERRAIN_COLS)),
            aspect="equal",
            interpolation="nearest",
            cmap="viridis",
            vmin=-1.0,
            vmax=1.0,
            origin="lower",
        )
        self.terrain_ax.set_title("policy_terrain_obs, base_z - terrain_z")
        self.terrain_ax.set_xlabel("x grid index")
        self.terrain_ax.set_ylabel("y grid index")
        self.fig.colorbar(self.terrain_img, ax=self.terrain_ax, fraction=0.046, pad=0.04)

        action_x = np.arange(_POLICY_ACTION_DIM)
        self.action_bars = self.action_ax.bar(action_x, np.zeros(_POLICY_ACTION_DIM))
        self.action_ax.set_title("policy action")
        self.action_ax.set_xticks(action_x)
        self.action_ax.set_xticklabels([f"a{i}" for i in action_x], rotation=90, fontsize=8)
        plt.tight_layout()
        plt.show(block=False)

    def update(self, obs_dict: dict[str, torch.Tensor] | None, actions: torch.Tensor | None):
        now = time.time()
        if now - self.last_draw_time < self.refresh_dt:
            return
        self.last_draw_time = now

        raw = self._first_env(obs_dict.get("policy_raw_obs") if obs_dict else None, _POLICY_RAW_DIM)
        history = self._first_env(
            obs_dict.get("policy_history_obs") if obs_dict else None,
            _POLICY_RAW_DIM * _POLICY_HISTORY_FRAMES,
        )
        terrain = self._first_env(
            obs_dict.get("policy_terrain_obs") if obs_dict else None,
            _POLICY_TERRAIN_ROWS * _POLICY_TERRAIN_COLS,
        )
        action = self._first_env(actions, _POLICY_ACTION_DIM)

        for bar, value in zip(self.raw_bars, raw):
            bar.set_height(float(value))
        self._set_symmetric_ylim(self.raw_ax, raw)

        history_frames = self._history_to_frames(history)
        history_abs = max(1.0, float(self.np.nanmax(self.np.abs(history_frames))))
        self.history_img.set_data(history_frames)
        self.history_img.set_clim(-history_abs, history_abs)

        self.terrain_img.set_data(terrain.reshape(_POLICY_TERRAIN_ROWS, _POLICY_TERRAIN_COLS))

        for bar, value in zip(self.action_bars, action):
            bar.set_height(float(value))
        self._set_symmetric_ylim(self.action_ax, action)

        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def close(self):
        self.plt.close(self.fig)

    def _first_env(self, tensor: torch.Tensor | None, expected_dim: int):
        if tensor is None:
            return self.np.zeros(expected_dim, dtype=self.np.float32)
        value = tensor.detach()
        if value.ndim > 1:
            value = value[0]
        value = value.flatten().cpu().numpy().astype(self.np.float32)
        if value.size == expected_dim:
            return value
        out = self.np.zeros(expected_dim, dtype=self.np.float32)
        size = min(expected_dim, value.size)
        out[:size] = value[:size]
        return out

    def _history_to_frames(self, history):
        frames = self.np.zeros((_POLICY_HISTORY_FRAMES, _POLICY_RAW_DIM), dtype=self.np.float32)
        history_offset = 0
        raw_offset = 0
        for _name, term_dim in _POLICY_TERM_LAYOUT:
            term_values = history[history_offset : history_offset + _POLICY_HISTORY_FRAMES * term_dim]
            frames[:, raw_offset : raw_offset + term_dim] = term_values.reshape(_POLICY_HISTORY_FRAMES, term_dim)
            history_offset += _POLICY_HISTORY_FRAMES * term_dim
            raw_offset += term_dim
        return frames

    def _set_symmetric_ylim(self, axis, values, minimum=1.0):
        finite = values[self.np.isfinite(values)]
        limit = minimum if finite.size == 0 else max(minimum, float(self.np.max(self.np.abs(finite))) * 1.1)
        axis.set_ylim(-limit, limit)


def _get_terrain_column_names(env) -> list[str]:
    """Return generated terrain names in column order."""
    terrain = env.unwrapped.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        return []

    generator_cfg = getattr(terrain.cfg, "terrain_generator", None)
    if generator_cfg is None or not getattr(generator_cfg, "sub_terrains", None):
        return []

    items = [
        (name, float(sub_cfg.proportion))
        for name, sub_cfg in generator_cfg.sub_terrains.items()
        if float(sub_cfg.proportion) > 0.0
    ]
    if not items:
        return []

    num_cols = int(terrain.terrain_origins.shape[1])
    total = sum(proportion for _, proportion in items)
    cumulative = []
    running = 0.0
    for name, proportion in items:
        running += proportion / total
        cumulative.append((name, running))

    column_names = []
    for index in range(num_cols):
        column_position = index / num_cols + 0.001
        selected_name = cumulative[-1][0]
        for name, threshold in cumulative:
            if column_position < threshold:
                selected_name = name
                break
        column_names.append(selected_name)
    return column_names


def _select_terrain_column(env, column: int):
    """Move env 0 to a specific generated terrain column before reset."""
    terrain = env.unwrapped.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        raise RuntimeError("Keyboard terrain selection requires generated terrain origins.")

    if column < 0 or column >= terrain.terrain_origins.shape[1]:
        raise ValueError(f"Terrain column {column} is out of range.")

    env_id = torch.tensor([0], dtype=torch.long, device=terrain.device)
    terrain.terrain_types[env_id] = column
    terrain.terrain_levels[env_id] = torch.clamp(
        terrain.terrain_levels[env_id],
        min=0,
        max=terrain.terrain_origins.shape[0] - 1,
    )
    terrain.env_origins[env_id] = terrain.terrain_origins[terrain.terrain_levels[env_id], terrain.terrain_types[env_id]]


def _adjust_terrain_level(env, delta: int) -> int:
    """Move env 0 up/down generated terrain difficulty rows before reset."""
    terrain = env.unwrapped.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        raise RuntimeError("Keyboard terrain level selection requires generated terrain origins.")

    env_id = torch.tensor([0], dtype=torch.long, device=terrain.device)
    max_level = terrain.terrain_origins.shape[0] - 1
    terrain.terrain_levels[env_id] = torch.clamp(terrain.terrain_levels[env_id] + int(delta), min=0, max=max_level)
    terrain.env_origins[env_id] = terrain.terrain_origins[terrain.terrain_levels[env_id], terrain.terrain_types[env_id]]
    return int(terrain.terrain_levels[0].item())


def _terrain_key_for_column(column: int) -> str:
    return "0" if column == 9 else str(column + 1)


def _update_follow_camera(env):
    """Keep the viewport camera in a fixed third-person view behind the keyboard-controlled robot."""
    if not args_cli.keyboard:
        return

    robot = env.unwrapped.scene["robot"]
    base_pos = robot.data.root_pos_w[0]
    base_quat = robot.data.root_quat_w[0]
    yaw_offset = math.radians(args_cli.follow_camera_yaw)
    eye_offset_b = torch.tensor(
        [
            -args_cli.follow_camera_distance * math.cos(yaw_offset),
            -args_cli.follow_camera_distance * math.sin(yaw_offset),
            args_cli.follow_camera_height,
        ],
        device=base_pos.device,
    )
    target_offset_b = torch.tensor(
        [args_cli.follow_camera_target_x, 0.0, args_cli.follow_camera_target_z], device=base_pos.device
    )
    eye = base_pos + quat_apply_yaw(base_quat.unsqueeze(0), eye_offset_b.unsqueeze(0)).squeeze(0)
    target = base_pos + quat_apply_yaw(base_quat.unsqueeze(0), target_offset_b.unsqueeze(0)).squeeze(0)
    eye = eye.detach().cpu()
    target = target.detach().cpu()
    set_camera_view(eye=eye.tolist(), target=target.tolist())


class PeriodicSidePush:
    """Apply and visualize a short periodic side push on the robot base COM."""

    def __init__(
        self,
        env,
        force_magnitude: float,
        duration: float,
        period: float,
        start_time: float,
        direction_y: float,
        arrow_length: float,
        arrow_z_offset: float,
    ):
        self.robot = env.unwrapped.scene["robot"]
        self.device = env.unwrapped.device
        self.force_magnitude = float(force_magnitude)
        self.duration = max(float(duration), 0.0)
        self.period = max(float(period), 1.0e-6)
        self.start_time = max(float(start_time), 0.0)
        self.direction_y = 1.0 if direction_y >= 0.0 else -1.0
        self.arrow_length = max(float(arrow_length), 1.0e-3)
        self.arrow_z_offset = float(arrow_z_offset)
        body_ids, body_names = self.robot.find_bodies("base", preserve_order=True)
        if not body_ids:
            body_ids = [0]
            body_names = [self.robot.body_names[0]]
        self.body_ids = body_ids
        self.force_b = torch.tensor(
            [[[0.0, self.direction_y * self.force_magnitude, 0.0]]], device=self.device, dtype=torch.float32
        )
        self.zero_force = torch.zeros_like(self.force_b)
        self.zero_torque = torch.zeros_like(self.force_b)
        marker_cfg = RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Play/side_push")
        marker_cfg.markers["arrow"].scale = (self.arrow_length, 0.08, 0.08)
        self.visualizer = VisualizationMarkers(marker_cfg)
        self.visualizer.set_visibility(False)
        self.was_active = False
        print(
            f"[INFO]: Side push enabled on body '{body_names[0]}': "
            f"{self.force_magnitude:.1f} N for {self.duration:.3f} s every {self.period:.1f} s, "
            f"first at {self.start_time:.1f} s."
        )

    def update(self, sim_time: float):
        active = self._is_active(sim_time)
        force_b = self.force_b if active else self.zero_force
        self.robot.set_external_force_and_torque(
            force_b,
            self.zero_torque,
            body_ids=self.body_ids,
            env_ids=[0],
            is_global=False,
        )
        if active:
            self._show_arrow()
        elif self.was_active:
            self.visualizer.set_visibility(False)
        self.was_active = active

    def clear(self):
        self.robot.set_external_force_and_torque(
            self.zero_force,
            self.zero_torque,
            body_ids=self.body_ids,
            env_ids=[0],
            is_global=False,
        )
        self.visualizer.set_visibility(False)
        self.was_active = False

    def _is_active(self, sim_time: float) -> bool:
        if self.duration <= 0.0 or sim_time < self.start_time:
            return False
        phase = (sim_time - self.start_time) % self.period
        return phase < self.duration

    def _show_arrow(self):
        base_quat_w = self.robot.data.root_link_quat_w[:1]
        force_w = math_utils.quat_apply(base_quat_w, self.force_b[:, 0])
        heading = torch.atan2(force_w[:, 1], force_w[:, 0])
        zeros = torch.zeros_like(heading)
        arrow_quat_w = math_utils.quat_from_euler_xyz(zeros, zeros, heading)
        arrow_pos_w = self.robot.data.root_com_pos_w[:1].clone()
        arrow_pos_w[:, 2] += self.arrow_z_offset
        arrow_scale = torch.tensor(
            [[self.arrow_length, 0.08, 0.08]], device=self.device, dtype=torch.float32
        )
        self.visualizer.set_visibility(True)
        self.visualizer.visualize(arrow_pos_w, arrow_quat_w, arrow_scale)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    if args_cli.keyboard and hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if args_cli.keyboard and hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    using_jit_policy = args_cli.jit_policy is not None
    if using_jit_policy:
        resume_path = retrieve_file_path(args_cli.jit_policy)
    elif args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading {'TorchScript policy' if using_jit_policy else 'model checkpoint'} from: {resume_path}")
    # load previously trained model
    using_custom_runner = hasattr(agent_cfg, "runner_class_name") and agent_cfg.runner_class_name
    if using_jit_policy:
        policy = _load_marg_oracle_jit_policy(resume_path, env.unwrapped.device)
        using_custom_runner = False
    elif using_custom_runner:
        runner_cls = _import_class(agent_cfg.runner_class_name)
        runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    if not using_jit_policy:
        runner.load(resume_path)

        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)

    if using_jit_policy:
        print("[INFO]: Using TorchScript policy with policy_raw_obs, policy_history_obs, and policy_terrain_obs.")
    elif using_custom_runner:
        print("[INFO]: Skipping default policy export because the custom policy expects dict observations.")
    else:
        # extract the neural network module
        # we do this in a try-except to maintain backwards compatibility.
        try:
            # version 2.3 onwards
            policy_nn = runner.alg.policy
        except AttributeError:
            # version 2.2 and below
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export policy to onnx/jit
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    keyboard = None
    filtered_keyboard_command = None
    policy_obs_visualizer = None
    keyboard_speed_scale = 1.0
    reset_requested = False
    pending_terrain_column = None
    pending_terrain_level_delta = 0
    terrain_column_names = []
    sim_time = 0.0

    def request_reset():
        nonlocal reset_requested
        reset_requested = True

    def request_terrain_reset(column: int):
        nonlocal pending_terrain_column, reset_requested
        pending_terrain_column = column
        reset_requested = True
        print(f"[INFO]: Requested terrain {column + 1}: {terrain_column_names[column]}")

    def request_terrain_level_reset(delta: int):
        nonlocal pending_terrain_level_delta, reset_requested
        pending_terrain_level_delta = int(delta)
        reset_requested = True
        direction = (
            "unchanged" if pending_terrain_level_delta == 0 else ("up" if pending_terrain_level_delta > 0 else "down")
        )
        print(f"[INFO]: Requested terrain level {direction}: {pending_terrain_level_delta:+d}")

    def adjust_keyboard_speed(delta: float):
        nonlocal keyboard_speed_scale
        keyboard_speed_scale = max(
            args_cli.keyboard_speed_min,
            min(args_cli.keyboard_speed_max, keyboard_speed_scale + delta),
        )
        print(f"[INFO]: Keyboard speed scale: {keyboard_speed_scale:.2f}")

    if args_cli.keyboard:
        keyboard = Se2Keyboard(
            Se2KeyboardCfg(
                v_x_sensitivity=args_cli.keyboard_vx,
                v_y_sensitivity=args_cli.keyboard_vy,
                omega_z_sensitivity=args_cli.keyboard_wz,
                sim_device=env.unwrapped.device,
            )
        )
        keyboard.add_callback("ENTER", request_reset)
        keyboard.add_callback("A", lambda: adjust_keyboard_speed(-args_cli.keyboard_speed_step))
        keyboard.add_callback("D", lambda: adjust_keyboard_speed(args_cli.keyboard_speed_step))
        keyboard.add_callback("W", lambda: request_terrain_level_reset(1))
        keyboard.add_callback("S", lambda: request_terrain_level_reset(-1))
        terrain_column_names = _get_terrain_column_names(env)
        for column, _ in enumerate(terrain_column_names[:10]):
            callback = lambda column=column: request_terrain_reset(column)
            key = _terrain_key_for_column(column)
            keyboard.add_callback(f"KEY_{key}", callback)
            keyboard.add_callback(key, callback)
        print(keyboard)
        print("\tReset robot pose: ENTER")
        print(
            f"\tAdjust keyboard speed: A/D "
            f"({args_cli.keyboard_speed_min:.1f}-{args_cli.keyboard_speed_max:.1f}, "
            f"step {args_cli.keyboard_speed_step:.1f})"
        )
        print("\tChange terrain level and respawn: W/S")
        if terrain_column_names:
            print("\tSelect terrain and respawn:")
            for column, terrain_name in enumerate(terrain_column_names[:10]):
                print(f"\t  {_terrain_key_for_column(column)}: {terrain_name}")
        else:
            print("\tSelect terrain and respawn: unavailable for this terrain config")
        filtered_keyboard_command = torch.zeros(3, device=env.unwrapped.device)

    if args_cli.policy_obs_visualizer:
        try:
            policy_obs_visualizer = PolicyObsVisualizer(args_cli.policy_obs_refresh_hz)
            print("[INFO]: Policy observation visualizer enabled.")
        except Exception as exc:
            policy_obs_visualizer = None
            print(f"[WARNING]: Failed to start policy observation visualizer: {exc}")

    side_push = None
    if args_cli.side_push:
        try:
            side_push = PeriodicSidePush(
                env,
                force_magnitude=args_cli.side_push_force,
                duration=args_cli.side_push_duration,
                period=args_cli.side_push_period,
                start_time=args_cli.side_push_start_time,
                direction_y=args_cli.side_push_direction_y,
                arrow_length=args_cli.side_push_arrow_length,
                arrow_z_offset=args_cli.side_push_arrow_z_offset,
            )
        except Exception as exc:
            side_push = None
            print(f"[WARNING]: Failed to start side push disturbance: {exc}")

    # reset environment
    obs = env.get_observations()
    obs_dict = None
    if version("rsl-rl-lib").startswith("2.3."):
        obs, extras = env.get_observations()
        obs_dict = extras["observations"]
    if using_jit_policy and obs_dict is None:
        obs, extras = _compute_observations(env, update_history=True)
        obs_dict = extras["observations"]
    if keyboard is not None:
        _set_base_velocity_command(env, filtered_keyboard_command)
        env.unwrapped.observation_manager.reset()
        obs, extras = _compute_observations(env, update_history=True)
        obs_dict = extras["observations"]
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if reset_requested:
                selected_terrain_name = None
                selected_terrain_column = None
                if pending_terrain_column is not None:
                    _select_terrain_column(env, pending_terrain_column)
                    selected_terrain_column = pending_terrain_column
                    selected_terrain_name = terrain_column_names[pending_terrain_column]
                if pending_terrain_level_delta != 0:
                    _adjust_terrain_level(env, pending_terrain_level_delta)
                    if selected_terrain_column is None:
                        terrain = env.unwrapped.scene.terrain
                        selected_terrain_column = int(terrain.terrain_types[0].item())
                        if selected_terrain_column < len(terrain_column_names):
                            selected_terrain_name = terrain_column_names[selected_terrain_column]
                        else:
                            selected_terrain_name = f"column {selected_terrain_column + 1}"
                obs, extras = env.reset()
                obs_dict = extras["observations"]
                if keyboard is not None:
                    keyboard.reset()
                if filtered_keyboard_command is not None:
                    filtered_keyboard_command.zero_()
                    _set_base_velocity_command(env, filtered_keyboard_command)
                    env.unwrapped.observation_manager.reset()
                    obs, extras = _compute_observations(env, update_history=True)
                    obs_dict = extras["observations"]
                if selected_terrain_name is not None:
                    terrain = env.unwrapped.scene.terrain
                    terrain_level = int(terrain.terrain_levels[0].item())
                    print(
                        f"[INFO]: Respawned on terrain {selected_terrain_column + 1}: "
                        f"{selected_terrain_name} (level {terrain_level})"
                    )
                pending_terrain_column = None
                pending_terrain_level_delta = 0
                reset_requested = False

            command = None
            if keyboard is not None:
                raw_command = keyboard.advance() * keyboard_speed_scale
                filtered_keyboard_command = (
                    (1.0 - args_cli.keyboard_smoothing) * filtered_keyboard_command
                    + args_cli.keyboard_smoothing * raw_command
                )
                # Clamp very small velocities to zero to avoid micro-oscillations in standing pose
                filtered_keyboard_command[torch.abs(filtered_keyboard_command) < 1.0e-3] = 0.0
                command = filtered_keyboard_command
                if torch.linalg.norm(raw_command) < 1.0e-4 and torch.linalg.norm(command) < 1.0e-3:
                    filtered_keyboard_command.zero_()
                    command = filtered_keyboard_command
                _set_base_velocity_command(env, command)
                obs, extras = _compute_observations(env, update_history=False)
                obs_dict = extras["observations"]
            _update_follow_camera(env)
            # agent stepping
            actions = policy(obs_dict if using_custom_runner or using_jit_policy else obs)
            if policy_obs_visualizer is not None:
                policy_obs_visualizer.update(obs_dict, actions)
            if side_push is not None:
                side_push.update(sim_time)
            # env stepping
            obs, _, _, infos = env.step(actions)
            if using_custom_runner or using_jit_policy or keyboard is not None:
                obs_dict = infos["observations"]
            sim_time += dt
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    if side_push is not None:
        side_push.clear()
    if policy_obs_visualizer is not None:
        policy_obs_visualizer.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
