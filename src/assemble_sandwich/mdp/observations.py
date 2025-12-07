import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def ingredient_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    diff_threshold: float = 0.05,
    grasp_threshold: float = 0.60
) -> torch.Tensor:
    """Check if any ingredient is grasped by the specified robot.

    This function determines if any ingredient (bread_slice_1, bread_slice_2, patty, or cheese_slice)
    has been successfully grasped by checking both the distance between the ingredient and
    the end-effector, and the gripper closure state.

    Args:
        env: The RL environment instance.
        robot_cfg: Configuration for the robot entity.
        ee_frame_cfg: Configuration for the end-effector frame entity.
        diff_threshold: Maximum distance threshold between ingredient and end-effector (meters).
        grasp_threshold: Maximum gripper position threshold for considering gripper closed.

    Returns:
        Boolean tensor indicating which environments have successfully grasped any ingredient.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # List of possible ingredient objects in the scene
    ingredient_names = ["bread_slice_1", "bread_slice_2", "patty", "cheese_slice"]

    # Get end-effector position
    end_effector_pos = ee_frame.data.target_pos_w[:, 1, :]  # Using index 1 like lift_cube

    # Check gripper closure state
    gripper_pos = robot.data.joint_pos[:, -1]  # Last joint is gripper
    gripper_closed = gripper_pos < grasp_threshold

    # Check if any ingredient is grasped
    any_ingredient_grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for ingredient_name in ingredient_names:
        # Check if this ingredient exists in the scene
        try:
            ingredient: RigidObject = env.scene[ingredient_name]
        except KeyError:
            # Ingredient not in scene, skip it
            continue

        # Get ingredient position
        ingredient_pos = ingredient.data.root_pos_w

        # Calculate distance between ingredient and end-effector
        pos_diff = torch.linalg.vector_norm(ingredient_pos - end_effector_pos, dim=1)

        # Check if ingredient is within distance threshold
        within_distance = pos_diff < diff_threshold

        # Ingredient is grasped if both conditions are met
        is_grasped = torch.logical_and(within_distance, gripper_closed)

        # Update overall grasped status
        any_ingredient_grasped = torch.logical_or(any_ingredient_grasped, is_grasped)

    return any_ingredient_grasped.clone().detach()
