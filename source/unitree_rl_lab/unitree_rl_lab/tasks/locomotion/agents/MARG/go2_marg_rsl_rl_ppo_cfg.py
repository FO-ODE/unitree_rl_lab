# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from .symmetry import compute_symmetric_states_go2_marg


@configclass
class Go2MargActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "unitree_rl_lab.tasks.locomotion.agents.MARG.go2_marg_actor_critic:Go2MargActorCritic"

    # Network architecture
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"
    init_noise_std = 1.0

    # Observation dimensions
    proprioception = 45
    proprioception_history = 270
    terrain_height = 187
    privileged = 42

    # Sub-network dimensions
    terrain_hidden_dims = [128, 64]
    terrain_feat_dim = 16

    estimator_hidden_dims = [256, 128]
    estimator_output_dim = 7


@configclass
class Go2MargPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    estimator_loss_coef = 1.0
    velocity_loss_coef = 1.0
    contact_loss_coef = 0.5


@configclass
class Go2MargPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Task-specific PPO runner config for Unitree-Go2-MARG tasks."""

    runner_class_name = "unitree_rl_lab.tasks.locomotion.agents.MARG.go2_marg_runner:Go2MargRunner"
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 100
    task_type = "velocity"  # Can be "velocity" or "risk_terrain"
    experiment_name = "go2_marg_velocity"
    empirical_normalization = False

    def __post_init__(self):
        """Dynamically set experiment_name based on task_type."""
        if self.task_type == "risk_terrain":
            self.experiment_name = "go2_marg_risk_terrain"
        elif self.task_type == "velocity":
            self.experiment_name = "go2_marg_velocity"

    policy = Go2MargActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = Go2MargPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Go2MargVelocityPPORunnerCfg(Go2MargPPORunnerCfg):
    """PPO runner config for Unitree-Go2-MARG-Velocity task."""

    task_type = "velocity"


@configclass
class Go2MargRiskTerrainPPORunnerCfg(Go2MargPPORunnerCfg):
    """PPO runner config for Unitree-Go2-MARG-Risk-Terrain task."""

    task_type = "risk_terrain"
    max_iterations = 30000
    terrain_stage_switch_iteration = 1000
    algorithm = Go2MargPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            mirror_loss_coeff=0.1,
            use_mirror_loss=True,
            data_augmentation_func=compute_symmetric_states_go2_marg,
        ),
    )
