import gymnasium as gym

gym.register(
    id="Unitree-Go2-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-MARG-Oracle-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_marg_oracle_velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.go2_marg_oracle_velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.go2_marg_oracle_rsl_rl_ppo_cfg:Go2MargOracleVelocityPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Unitree-Go2-MARG-Oracle-Risk-Terrain",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_marg_oracle_risk_terrain_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.go2_marg_oracle_risk_terrain_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.go2_marg_oracle_rsl_rl_ppo_cfg:Go2MargOracleRiskTerrainPPORunnerCfg"
        ),
    },
)
