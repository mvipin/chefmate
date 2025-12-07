from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .assemble_sandwich_env_cfg import AssembleSandwichEnvCfg
from ..template import SingleArmObservationsCfg
from . import mdp


@configclass
class AssembleSandwichMimicObservationsCfg(SingleArmObservationsCfg):
    """Observation specifications for the assemble sandwich mimic task."""

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        grasp_ingredient = ObsTerm(
            func=mdp.ingredient_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "diff_threshold": 0.05,
                "grasp_threshold": 0.60,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class AssembleSandwichMimicEnvCfg(AssembleSandwichEnvCfg, MimicEnvCfg):
    """
    Configuration for assemble sandwich task with mimic environment support.
    This environment supports language prompt-based differentiation between ingredient types
    (bread, patty, cheese) using the same physical manipulation skills. The specific ingredient
    type is determined by language prompts during training and inference.
    """

    # Override observations to include subtask terms for automatic annotation
    observations: AssembleSandwichMimicObservationsCfg = AssembleSandwichMimicObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Data generation configuration
        self.datagen_config.name = "sandwich_ingredient_leisaac_task_v0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 20  # More trials for better generalization
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 50
        self.datagen_config.seed = 42

        # Generalized subtask configurations for ingredient manipulation
        # These work for any ingredient type (bread, patty, cheese) with language prompt differentiation
        subtask_configs = []
        
        # Subtask 1: Grasp ingredient from cartridge
        # Note: The specific ingredient object name must match the USD scene object
        # For bread demos: use "bread_slice_1" or "bread_slice_2"
        # For patty demos: use "patty"
        # For cheese demos: use "cheese_slice"
        subtask_configs.append(
            SubTaskConfig(
                object_ref="bread_slice_1",  # IMPORTANT: Change this to match the object in your demo
                subtask_term_signal="grasp_ingredient",  # Generic termination signal
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 5},  # More neighbors for better selection
                action_noise=0.002,  # Lower noise for precise manipulation
                num_interpolation_steps=8,  # More steps for smoother motion
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp bread slice from cartridge",
                next_subtask_description="Place bread slice on plate",
            )
        )
        
        # Subtask 2: Place ingredient on plate (final subtask)
        subtask_configs.append(
            SubTaskConfig(
                object_ref="plate",  # Reference to destination plate
                subtask_term_signal=None,  # No termination signal for final subtask
                subtask_term_offset_range=(0, 0),  # No offset for final subtask
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 5},
                action_noise=0.002,
                num_interpolation_steps=8,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place bread slice on plate",
            )
        )
        
        # Assign subtask configurations to the SO101 follower robot
        self.subtask_configs["so101_follower"] = subtask_configs
