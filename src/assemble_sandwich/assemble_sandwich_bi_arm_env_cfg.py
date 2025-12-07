from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from leisaac.assets.scenes.kitchen import KITCHEN_WITH_SANDWICH_CFG, KITCHEN_WITH_SANDWICH_USD_PATH
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ..template import BiArmTaskSceneCfg, BiArmTaskEnvCfg, BiArmObservationsCfg, BiArmTerminationsCfg


@configclass
class AssembleSandwichBiArmSceneCfg(BiArmTaskSceneCfg):
    """Scene configuration for the assemble sandwich task using two arms."""

    scene: AssetBaseCfg = KITCHEN_WITH_SANDWICH_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")


@configclass
class AssembleSandwichBiArmEnvCfg(BiArmTaskEnvCfg):
    """Configuration for the assemble sandwich environment using two arms."""

    scene: AssembleSandwichBiArmSceneCfg = AssembleSandwichBiArmSceneCfg(env_spacing=8.0)

    observations: BiArmObservationsCfg = BiArmObservationsCfg()

    terminations: BiArmTerminationsCfg = BiArmTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Optimized viewer camera for simplified table workspace
        self.viewer.eye = (1.5, -1.5, 1.8)     # Elevated diagonal view
        self.viewer.lookat = (0.0, 0.0, 0.9)   # Looking at table center

        # Robot positions for simplified table workspace (1.2m × 0.8m × 0.85m)
        self.scene.left_arm.init_state.pos = (-0.3, -1.0, 0.0)   # Left side of table
        self.scene.right_arm.init_state.pos = (0.3, -1.0, 0.0)   # Right side of table

        # Optimized camera positions for sandwich assembly task
        # Left wrist camera
        self.scene.left_wrist.offset.pos = (0.02, 0.08, -0.03)
        self.scene.left_wrist.offset.rot = (-0.35, -0.93, -0.05, 0.08)
        self.scene.left_wrist.spawn.focal_length = 32.0
        self.scene.left_wrist.spawn.horizontal_aperture = 40.0

        # Right wrist camera
        self.scene.right_wrist.offset.pos = (0.02, 0.08, -0.03)
        self.scene.right_wrist.offset.rot = (-0.35, -0.93, -0.05, 0.08)
        self.scene.right_wrist.spawn.focal_length = 32.0
        self.scene.right_wrist.spawn.horizontal_aperture = 40.0

        # Top camera: Complete table workspace overview
        self.scene.top.offset.pos = (0.0, -0.8, 0.8)
        self.scene.top.offset.rot = (0.2, -0.98, 0.0, 0.0)
        self.scene.top.spawn.focal_length = 20.0
        self.scene.top.spawn.horizontal_aperture = 50.0  # ~90° FOV for full table

        parse_usd_and_create_subassets(KITCHEN_WITH_SANDWICH_USD_PATH, self)
