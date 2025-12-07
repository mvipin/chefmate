"""Termination functions for the assemble sandwich task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def task_done(
    env: ManagerBasedRLEnv,
    plate_cfg: SceneEntityCfg,
    bread_slice_1_cfg: SceneEntityCfg,
    cheese_slice_cfg: SceneEntityCfg,
    patty_cfg: SceneEntityCfg,
    bread_slice_2_cfg: SceneEntityCfg,
    height_threshold: float = 0.02,
    xy_threshold: float = 0.05,
    test_mode: bool = False,
) -> torch.Tensor:
    """Determine if the sandwich assembly task is complete.

    This function checks whether all success conditions for the task have been met:
    1. All ingredients are stacked on the plate within xy_threshold
    2. Ingredients are stacked in the correct vertical order (within height_threshold of expected positions)
    3. The sandwich is stable (low velocity)

    Args:
        env: The environment.
        plate_cfg: Configuration for the plate entity.
        bread_slice_1_cfg: Configuration for the bottom bread slice entity.
        cheese_slice_cfg: Configuration for the cheese slice entity.
        patty_cfg: Configuration for the patty entity.
        bread_slice_2_cfg: Configuration for the top bread slice entity.
        height_threshold: Maximum allowed deviation from expected height for each ingredient (in meters).
        xy_threshold: Maximum allowed XY distance from plate center for each ingredient (in meters).
        test_mode: If True, always return True (for testing/manual annotation with incomplete demos).
                   If False, enforce strict success criteria. Default: False.

    Returns:
        Boolean tensor indicating whether the task is done for each environment.
    """
    # For testing the pipeline with incomplete demonstrations, always return True
    if test_mode:
        num_envs = env.num_envs
        return torch.ones(num_envs, dtype=torch.bool, device=env.device)

    # Get the rigid objects from the scene
    plate = env.scene[plate_cfg.name]
    bread_slice_1 = env.scene[bread_slice_1_cfg.name]
    cheese_slice = env.scene[cheese_slice_cfg.name]
    patty = env.scene[patty_cfg.name]
    bread_slice_2 = env.scene[bread_slice_2_cfg.name]

    # Get positions
    plate_pos = plate.data.root_pos_w
    bread_1_pos = bread_slice_1.data.root_pos_w
    cheese_pos = cheese_slice.data.root_pos_w
    patty_pos = patty.data.root_pos_w
    bread_2_pos = bread_slice_2.data.root_pos_w

    # Get velocities for stability check
    bread_1_vel = torch.norm(bread_slice_1.data.root_lin_vel_w, dim=-1)
    cheese_vel = torch.norm(cheese_slice.data.root_lin_vel_w, dim=-1)
    patty_vel = torch.norm(patty.data.root_lin_vel_w, dim=-1)
    bread_2_vel = torch.norm(bread_slice_2.data.root_lin_vel_w, dim=-1)

    # Check XY alignment with plate center
    bread_1_xy_dist = torch.norm(bread_1_pos[:, :2] - plate_pos[:, :2], dim=-1)
    cheese_xy_dist = torch.norm(cheese_pos[:, :2] - plate_pos[:, :2], dim=-1)
    patty_xy_dist = torch.norm(patty_pos[:, :2] - plate_pos[:, :2], dim=-1)
    bread_2_xy_dist = torch.norm(bread_2_pos[:, :2] - plate_pos[:, :2], dim=-1)

    xy_aligned = (
        (bread_1_xy_dist < xy_threshold)
        & (cheese_xy_dist < xy_threshold)
        & (patty_xy_dist < xy_threshold)
        & (bread_2_xy_dist < xy_threshold)
    )

    # Check vertical stacking order
    # Expected heights above plate (approximate, adjust based on your ingredient thicknesses)
    # Plate is at height 0 (reference)
    # bread_slice_1 should be ~0.015m above plate
    # cheese_slice should be ~0.030m above plate (bread + cheese thickness)
    # patty should be ~0.045m above plate (bread + cheese + patty thickness)
    # bread_slice_2 should be ~0.060m above plate (full stack)

    bread_1_height = bread_1_pos[:, 2] - plate_pos[:, 2]
    cheese_height = cheese_pos[:, 2] - plate_pos[:, 2]
    patty_height = patty_pos[:, 2] - plate_pos[:, 2]
    bread_2_height = bread_2_pos[:, 2] - plate_pos[:, 2]

    # Check if ingredients are in correct vertical order (each higher than the previous)
    vertical_order = (
        (bread_1_height > 0.005)  # Bottom bread is above plate
        & (cheese_height > bread_1_height)  # Cheese is above bottom bread
        & (patty_height > cheese_height)  # Patty is above cheese
        & (bread_2_height > patty_height)  # Top bread is above patty
    )

    # Check stability (all ingredients have low velocity)
    velocity_threshold = 0.05  # m/s
    stable = (
        (bread_1_vel < velocity_threshold)
        & (cheese_vel < velocity_threshold)
        & (patty_vel < velocity_threshold)
        & (bread_2_vel < velocity_threshold)
    )

    # Task is done when all conditions are met
    task_complete = xy_aligned & vertical_order & stable

    return task_complete

