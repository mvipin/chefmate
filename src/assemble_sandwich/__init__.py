import gymnasium as gym

# Register single-arm sandwich assembly environment
gym.register(
    id='LeIsaac-SO101-AssembleSandwich-v0',
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.assemble_sandwich_env_cfg:AssembleSandwichEnvCfg",
    },
)

# Register bi-arm sandwich assembly environment
gym.register(
    id='LeIsaac-SO101-AssembleSandwich-BiArm-v0',
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.assemble_sandwich_bi_arm_env_cfg:AssembleSandwichBiArmEnvCfg",
    },
)

# Register assemble sandwich environment with MimicGen support
# This environment uses language prompts to differentiate between ingredient types (bread, patty, cheese)
gym.register(
    id='LeIsaac-SO101-AssembleSandwich-Mimic-v0',
    entry_point=f"leisaac.enhance.envs:ManagerBasedRLLeIsaacMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.assemble_sandwich_mimic_env_cfg:AssembleSandwichMimicEnvCfg",
    },
)
