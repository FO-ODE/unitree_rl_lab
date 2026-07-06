#!/usr/bin/env python3
"""Export the Go2 MARG-Oracle actor-only policy as TorchScript."""

from __future__ import annotations

import argparse
import os
import sys
from importlib import import_module
from importlib import util as importlib_util
from pathlib import Path

import torch
import yaml


def _install_rsl_rl_activation_fallback() -> None:
    try:
        import_module("rsl_rl.utils")
        return
    except ModuleNotFoundError:
        pass

    import types

    def resolve_nn_activation(name: str) -> torch.nn.Module:
        activations = {
            "elu": torch.nn.ELU,
            "relu": torch.nn.ReLU,
            "selu": torch.nn.SELU,
            "crelu": torch.nn.ReLU,
            "lrelu": torch.nn.LeakyReLU,
            "tanh": torch.nn.Tanh,
            "sigmoid": torch.nn.Sigmoid,
        }
        key = name.lower()
        if key not in activations:
            raise ValueError(f"Unsupported activation: {name}")
        return activations[key]()

    rsl_rl_module = types.ModuleType("rsl_rl")
    utils_module = types.ModuleType("rsl_rl.utils")
    utils_module.resolve_nn_activation = resolve_nn_activation
    rsl_rl_module.utils = utils_module
    sys.modules["rsl_rl"] = rsl_rl_module
    sys.modules["rsl_rl.utils"] = utils_module


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _import_class(import_path: str):
    module_name, class_name = import_path.rsplit(":", 1)
    try:
        module = import_module(module_name)
    except ModuleNotFoundError:
        source_root = _repo_root() / "source/unitree_rl_lab"
        module_path = source_root / Path(*module_name.split(".")).with_suffix(".py")
        if not module_path.exists():
            raise
        spec = importlib_util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {module_path}")
        module = importlib_util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return getattr(module, class_name)


class ActorOnlyWrapper(torch.nn.Module):
    def __init__(self, policy: torch.nn.Module):
        super().__init__()
        self.policy = policy

    def forward(
        self,
        policy_raw_obs: torch.Tensor,
        policy_history_obs: torch.Tensor,
        policy_terrain_obs: torch.Tensor,
    ) -> torch.Tensor:
        return self.policy.act_inference(
            {
                "policy_raw_obs": policy_raw_obs,
                "policy_history_obs": policy_history_obs,
                "policy_terrain_obs": policy_terrain_obs,
            }
        )


def _latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(run_dir.glob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoints found in {run_dir}")

    def iteration(path: Path) -> int:
        return int(path.stem.split("_")[-1])

    return max(checkpoints, key=iteration)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=_repo_root() / "logs/rsl_rl/go2_marg_oracle_risk_terrain/2026-07-04_17-04-33",
        help="Training run directory containing params/agent.yaml and model_*.pt.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path. Defaults to latest model_*.pt.")
    parser.add_argument("--output", type=Path, default=None, help="Output TorchScript path.")
    args = parser.parse_args()

    repo_root = _repo_root()
    source_path = repo_root / "source/unitree_rl_lab"
    sys.path.insert(0, str(source_path))
    _install_rsl_rl_activation_fallback()

    run_dir = args.run_dir.resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else _latest_checkpoint(run_dir)
    output = args.output.resolve() if args.output else run_dir / "exported/policy.pt"
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "params/agent.yaml", "r", encoding="utf-8") as f:
        agent_cfg = yaml.safe_load(f)

    policy_cfg = dict(agent_cfg["policy"])
    policy_class = _import_class(policy_cfg.pop("class_name"))
    policy = policy_class(num_actions=12, **policy_cfg).cpu().eval()

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy.load_state_dict(checkpoint_data["model_state_dict"])

    wrapper = ActorOnlyWrapper(policy).cpu().eval()
    raw = torch.zeros(1, int(policy.proprioception), dtype=torch.float32)
    history = torch.zeros(1, int(policy.proprioception_history), dtype=torch.float32)
    terrain = torch.zeros(1, int(policy.terrain_height), dtype=torch.float32)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (raw, history, terrain), strict=True)
        traced = torch.jit.freeze(traced)
        torch_out = wrapper(raw, history, terrain)
        jit_out = traced(raw, history, terrain)

    max_error = (torch_out - jit_out).abs().max().item()
    if max_error > 1.0e-5:
        raise RuntimeError(f"TorchScript parity check failed: max abs error {max_error}")

    traced.save(str(output))
    rel_output = os.path.relpath(output, repo_root)
    rel_checkpoint = os.path.relpath(checkpoint, repo_root)
    print(f"[INFO] Exported actor-only TorchScript policy: {rel_output}")
    print(f"[INFO] Source checkpoint: {rel_checkpoint}")
    print(f"[INFO] Inputs: policy_raw_obs[1,45], policy_history_obs[1,270], policy_terrain_obs[1,187]")
    print(f"[INFO] Max TorchScript parity error: {max_error:.3g}")


if __name__ == "__main__":
    main()
