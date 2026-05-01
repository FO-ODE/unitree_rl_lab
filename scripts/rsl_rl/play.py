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
parser.add_argument("--keyboard_smoothing", type=float, default=0.3, help="Low-pass factor for keyboard commands.")
parser.add_argument("--follow_camera_distance", type=float, default=3, help="Follow camera distance behind robot.")
parser.add_argument("--follow_camera_height", type=float, default=1.4, help="Follow camera height above robot.")
parser.add_argument("--follow_camera_yaw", type=float, default=-30.0, help="Follow camera yaw offset in degrees.")
parser.add_argument("--follow_camera_target_x", type=float, default=1.0, help="Look-ahead target offset in robot x axis.")
parser.add_argument("--follow_camera_target_z", type=float, default=0.35, help="Look-at target height offset.")
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
from isaaclab.utils.math import quat_apply_yaw
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaacsim.core.utils.viewports import set_camera_view

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _import_class(import_path: str):
    module_name, class_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _override_base_velocity_command(env, obs_dict: dict[str, torch.Tensor] | None, command: torch.Tensor | None = None):
    """Override the velocity command for keyboard teleoperation."""
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

    if obs_dict is None:
        return
    for key in ("policy", "policy_raw_obs"):
        if key in obs_dict:
            obs_dict[key][:, 6:9] = command.to(obs_dict[key].device)
    if "policy_history_obs" in obs_dict:
        # IsaacLab flattens history per observation term before concatenating terms:
        # base_ang_vel[5*3], projected_gravity[5*3], velocity_commands[5*3], ...
        history = obs_dict["policy_history_obs"]
        command_history = command.to(history.device).repeat(5)
        history[:, 30:45] = command_history


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
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
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

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    using_custom_runner = hasattr(agent_cfg, "runner_class_name") and agent_cfg.runner_class_name
    if using_custom_runner:
        runner_cls = _import_class(agent_cfg.runner_class_name)
        runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    if using_custom_runner:
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
    reset_requested = False

    def request_reset():
        nonlocal reset_requested
        reset_requested = True

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
        print(keyboard)
        print("\tReset robot pose: ENTER")
        filtered_keyboard_command = torch.zeros(3, device=env.unwrapped.device)

    # reset environment
    obs = env.get_observations()
    obs_dict = None
    if version("rsl-rl-lib").startswith("2.3."):
        obs, extras = env.get_observations()
        obs_dict = extras["observations"]
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if reset_requested:
                obs, extras = env.reset()
                obs_dict = extras["observations"]
                if filtered_keyboard_command is not None:
                    filtered_keyboard_command.zero_()
                reset_requested = False

            command = None
            if keyboard is not None:
                raw_command = keyboard.advance()
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
            _override_base_velocity_command(env, obs_dict, command)
            _update_follow_camera(env)
            # agent stepping
            actions = policy(obs_dict if using_custom_runner else obs)
            # env stepping
            obs, _, _, infos = env.step(actions)
            if using_custom_runner:
                obs_dict = infos["observations"]
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
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
