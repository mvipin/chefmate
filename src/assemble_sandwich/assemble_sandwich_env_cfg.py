from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg

from leisaac.assets.scenes.kitchen import KITCHEN_WITH_SANDWICH_CFG, KITCHEN_WITH_SANDWICH_USD_PATH
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ..template import SingleArmTaskSceneCfg, SingleArmTaskEnvCfg, SingleArmTerminationsCfg, SingleArmObservationsCfg
from . import mdp


@configclass
class AssembleSandwichSceneCfg(SingleArmTaskSceneCfg):
    """Scene configuration for the assemble sandwich task."""

    scene: AssetBaseCfg = KITCHEN_WITH_SANDWICH_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")


@configclass
class TerminationsCfg(SingleArmTerminationsCfg):
    """Termination configuration for the assemble sandwich task."""

    success = DoneTerm(
        func=mdp.task_done,
        params={
            "plate_cfg": SceneEntityCfg("plate"),
            "bread_slice_1_cfg": SceneEntityCfg("bread_slice_1"),
            "cheese_slice_cfg": SceneEntityCfg("cheese_slice"),
            "patty_cfg": SceneEntityCfg("patty"),
            "bread_slice_2_cfg": SceneEntityCfg("bread_slice_2"),
            "height_threshold": 0.02,
            "xy_threshold": 0.05,
            "test_mode": False,  # Default: enforce strict success criteria
        },
    )


@configclass
class AssembleSandwichEnvCfg(SingleArmTaskEnvCfg):
    """Configuration for the assemble sandwich environment."""

    scene: AssembleSandwichSceneCfg = AssembleSandwichSceneCfg(env_spacing=8.0)

    observations: SingleArmObservationsCfg = SingleArmObservationsCfg()

    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Robot position for simplified table workspace
        self.scene.robot.init_state.pos = (2.7, -2, 0.81)  # Closer to desk/table

        # Optimized viewer camera for teleoperation - front-diagonal view of table workspace
        # Positioned to give clear view of table surface, ingredients, and robot arm movements
        # Camera positions are in ABSOLUTE world coordinates (not relative to env origin)
        # Table/workspace is located around (2.7, -1.5, 0.9) in world coordinates
        self.viewer.eye = (3.2, -3.5, 1.6)     # Front-diagonal elevated view of the actual table location
        self.viewer.lookat = (2.7, -1.8, 0.9)  # Looking at table workspace center (near robot base)

        # Optimized camera positions for sandwich assembly task
        # Wrist camera: Better view of ingredients and manipulation
        self.scene.wrist.offset.pos = (0.02, 0.08, -0.03)
        self.scene.wrist.offset.rot = (-0.35, -0.93, -0.05, 0.08)
        self.scene.wrist.spawn.focal_length = 32.0
        self.scene.wrist.spawn.horizontal_aperture = 40.0  # ~80° FOV
        self.scene.wrist.spawn.clipping_range = (0.005, 20.0)

        # Front camera: Complete table workspace overview
        # Centered between plate (X=2.9) and ingredients holder (X=2.4)
        # Camera world position: (2.65, -2.8, 1.51) for balanced workspace view
        self.scene.front.offset.pos = (-0.05, -0.8, 0.7)
        self.scene.front.offset.rot = (0.2, -0.98, 0.0, 0.0)
        self.scene.front.spawn.focal_length = 24.0
        self.scene.front.spawn.horizontal_aperture = 45.0  # ~85° FOV

        parse_usd_and_create_subassets(KITCHEN_WITH_SANDWICH_USD_PATH, self)
