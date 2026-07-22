"""Left-right symmetry data augmentation for the Go2 MARG policy."""

import torch


PROPRIO_DIM = 45
HISTORY_LENGTH = 6

# Modified Go2 URDF revolute joint order:
# FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, RL_thigh, RR_thigh, FL_calf, FR_calf, RL_calf, RR_calf.
GO2_LEFT_JOINT_IDS = [0, 2, 4, 6, 8, 10]
GO2_RIGHT_JOINT_IDS = [1, 3, 5, 7, 9, 11]
GO2_HAA_JOINT_IDS = [0, 1, 2, 3]

# GridPatternCfg(ordering="xy") flattens as y rows x x columns: 11 rows * 17 cols = 187 heights.
TERRAIN_GRID_Y_POINTS = 11
TERRAIN_GRID_X_POINTS = 17


@torch.no_grad()
def compute_symmetric_states_go2_marg(env, obs, actions):
    """Append left-right mirrored Go2 MARG observations and actions to a batch."""

    obs_aug = _augment_go2_marg_obs(obs) if obs is not None else None
    actions_aug = _augment_go2_marg_actions(actions) if actions is not None else None
    return obs_aug, actions_aug


def _augment_go2_marg_obs(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    batch_size = next(iter(obs.values())).shape[0]
    obs_aug = {}

    for key, value in obs.items():
        if key in ("policy_raw_obs", "policy"):
            mirrored = _transform_proprio_obs_left_right(value)
        elif key == "policy_history_obs":
            mirrored = _transform_proprio_history_left_right(value)
        elif key == "policy_terrain_obs":
            mirrored = _transform_terrain_map_left_right(value)
        elif key == "privileged_obs":
            mirrored = _transform_privileged_obs_left_right(value)
        else:
            mirrored = value

        obs_aug[key] = torch.empty(batch_size * 2, *value.shape[1:], device=value.device, dtype=value.dtype)
        obs_aug[key][:batch_size] = value
        obs_aug[key][batch_size:] = mirrored

    return obs_aug


def _augment_go2_marg_actions(actions: torch.Tensor) -> torch.Tensor:
    batch_size = actions.shape[0]
    actions_aug = torch.empty(batch_size * 2, *actions.shape[1:], device=actions.device, dtype=actions.dtype)
    actions_aug[:batch_size] = actions
    actions_aug[batch_size:] = _switch_go2_joints_left_right(actions, flip_haa=True)
    return actions_aug


def _transform_proprio_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    obs = obs.clone()
    device = obs.device

    # Layout: base_ang_vel(3), projected_gravity(3), command(3), joint_pos(12), joint_vel(12), last_action(12).
    obs[..., 0:3] = obs[..., 0:3] * torch.tensor([-1.0, 1.0, -1.0], device=device)
    obs[..., 3:6] = obs[..., 3:6] * torch.tensor([1.0, -1.0, 1.0], device=device)
    obs[..., 6:9] = obs[..., 6:9] * torch.tensor([1.0, -1.0, -1.0], device=device)
    obs[..., 9:21] = _switch_go2_joints_left_right(obs[..., 9:21], flip_haa=True)
    obs[..., 21:33] = _switch_go2_joints_left_right(obs[..., 21:33], flip_haa=True)
    obs[..., 33:45] = _switch_go2_joints_left_right(obs[..., 33:45], flip_haa=True)
    return obs


def _transform_proprio_history_left_right(obs: torch.Tensor) -> torch.Tensor:
    original_shape = obs.shape
    expected_dim = HISTORY_LENGTH * PROPRIO_DIM
    if original_shape[-1] != expected_dim:
        raise ValueError(f"Expected proprio history dim {expected_dim}, got {original_shape[-1]}")

    # ObservationManager flattens each term's history before concatenating
    # terms. Reconstruct frames, mirror each frame, then restore term-major order.
    term_dims = (3, 3, 3, 12, 12, 12)
    history_terms = []
    history_offset = 0
    for term_dim in term_dims:
        term_width = HISTORY_LENGTH * term_dim
        term = obs[..., history_offset : history_offset + term_width]
        history_terms.append(term.reshape(*original_shape[:-1], HISTORY_LENGTH, term_dim))
        history_offset += term_width
    history_frames = torch.cat(history_terms, dim=-1)
    mirrored_frames = _transform_proprio_obs_left_right(history_frames)

    mirrored_terms = []
    frame_offset = 0
    for term_dim in term_dims:
        term = mirrored_frames[..., frame_offset : frame_offset + term_dim]
        mirrored_terms.append(term.reshape(*original_shape[:-1], HISTORY_LENGTH * term_dim))
        frame_offset += term_dim
    return torch.cat(mirrored_terms, dim=-1)


def _transform_terrain_map_left_right(obs: torch.Tensor) -> torch.Tensor:
    original_shape = obs.shape
    obs = obs.view(*original_shape[:-1], TERRAIN_GRID_Y_POINTS, TERRAIN_GRID_X_POINTS)
    obs = torch.flip(obs, dims=[-2])
    return obs.contiguous().view(original_shape)


def _transform_privileged_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    obs = obs.clone()
    device = obs.device

    # Layout: lin_vel(3), feet_contacts(4), mass_summary(4), friction(1), com_xy(2), force_xy(2), actuator_params(26).
    obs[..., 0:3] = obs[..., 0:3] * torch.tensor([1.0, -1.0, 1.0], device=device)
    obs[..., 3:7] = _switch_go2_feet_left_right(obs[..., 3:7])
    obs[..., 12:14] = obs[..., 12:14] * torch.tensor([1.0, -1.0], device=device)
    obs[..., 14:16] = obs[..., 14:16] * torch.tensor([1.0, -1.0], device=device)
    obs[..., 18:30] = _switch_go2_joints_left_right(obs[..., 18:30], flip_haa=False)
    obs[..., 30:42] = _switch_go2_joints_left_right(obs[..., 30:42], flip_haa=True)
    return obs


def _switch_go2_feet_left_right(feet_data: torch.Tensor) -> torch.Tensor:
    feet_data_switched = torch.zeros_like(feet_data)
    feet_data_switched[..., [0, 2]] = feet_data[..., [1, 3]]
    feet_data_switched[..., [1, 3]] = feet_data[..., [0, 2]]
    return feet_data_switched


def _switch_go2_joints_left_right(joint_data: torch.Tensor, flip_haa: bool) -> torch.Tensor:
    joint_data_switched = torch.zeros_like(joint_data)
    joint_data_switched[..., GO2_LEFT_JOINT_IDS] = joint_data[..., GO2_RIGHT_JOINT_IDS]
    joint_data_switched[..., GO2_RIGHT_JOINT_IDS] = joint_data[..., GO2_LEFT_JOINT_IDS]

    if flip_haa:
        joint_data_switched[..., GO2_HAA_JOINT_IDS] *= -1.0

    return joint_data_switched
