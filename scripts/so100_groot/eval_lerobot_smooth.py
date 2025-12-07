#!/usr/bin/env python3
"""
Smoothed version of eval_lerobot.py with action filtering to reduce jitter.

This script adds:
1. Exponential moving average (EMA) filter for action smoothing
2. Configurable action execution rate
3. Optional action interpolation between chunks
4. Velocity limiting to prevent sudden movements
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Optional

import draccus
import numpy as np
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots import (
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.utils.utils import init_logging, log_say

import sys
import os
sys.path.append(os.path.expanduser("~/Isaac-GR00T/examples/SO-100"))
from service import ExternalRobotInferenceClient


class ActionSmoother:
    """Exponential Moving Average filter for action smoothing."""
    
    def __init__(self, alpha: float = 0.3, num_joints: int = 6):
        """
        Args:
            alpha: Smoothing factor (0-1). Lower = smoother but slower response.
                   0.1 = very smooth, 0.5 = balanced, 0.9 = minimal smoothing
            num_joints: Number of joints to smooth
        """
        self.alpha = alpha
        self.prev_action: Optional[np.ndarray] = None
        self.num_joints = num_joints
        
    def smooth(self, action: np.ndarray) -> np.ndarray:
        """Apply EMA smoothing to action."""
        if self.prev_action is None:
            self.prev_action = action.copy()
            return action
        
        # EMA: smoothed = alpha * new + (1 - alpha) * prev
        smoothed = self.alpha * action + (1 - self.alpha) * self.prev_action
        self.prev_action = smoothed.copy()
        return smoothed
    
    def reset(self):
        """Reset the filter state."""
        self.prev_action = None


class VelocityLimiter:
    """Limit maximum velocity between consecutive actions."""
    
    def __init__(self, max_delta: float = 5.0, num_joints: int = 6):
        """
        Args:
            max_delta: Maximum allowed change per joint per step (in robot units)
            num_joints: Number of joints
        """
        self.max_delta = max_delta
        self.prev_action: Optional[np.ndarray] = None
        self.num_joints = num_joints
        
    def limit(self, action: np.ndarray) -> np.ndarray:
        """Limit velocity by clamping delta."""
        if self.prev_action is None:
            self.prev_action = action.copy()
            return action
        
        # Calculate delta
        delta = action - self.prev_action
        
        # Clamp delta to max_delta
        delta = np.clip(delta, -self.max_delta, self.max_delta)
        
        # Apply limited delta
        limited = self.prev_action + delta
        self.prev_action = limited.copy()
        return limited
    
    def reset(self):
        """Reset the limiter state."""
        self.prev_action = None


class Gr00tRobotInferenceClient:
    """GR00T policy client with action smoothing."""

    def __init__(
        self,
        host="localhost",
        port=5555,
        camera_keys=[],
        robot_state_keys=[],
        show_images=False,
        enable_smoothing=True,
        smoothing_alpha=0.3,
        enable_velocity_limit=True,
        max_velocity=5.0,
    ):
        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.camera_keys = camera_keys
        self.robot_state_keys = robot_state_keys
        self.show_images = show_images
        assert len(robot_state_keys) == 6, f"robot_state_keys should be size 6, but got {len(robot_state_keys)}"
        self.modality_keys = ["single_arm", "gripper"]
        
        # Action filtering
        self.enable_smoothing = enable_smoothing
        self.enable_velocity_limit = enable_velocity_limit
        if enable_smoothing:
            self.smoother = ActionSmoother(alpha=smoothing_alpha, num_joints=6)
            print(f"[SMOOTHING] Enabled with alpha={smoothing_alpha}")
        if enable_velocity_limit:
            self.velocity_limiter = VelocityLimiter(max_delta=max_velocity, num_joints=6)
            print(f"[VELOCITY LIMIT] Enabled with max_delta={max_velocity}")

    def get_action(self, observation_dict, lang: str):
        # First add the images
        obs_dict = {f"video.{key}": observation_dict[key] for key in self.camera_keys}

        # Make all single float value of dict[str, float] state into a single array
        state = np.array([observation_dict[k] for k in self.robot_state_keys])
        obs_dict["state.single_arm"] = state[:5].astype(np.float64)
        obs_dict["state.gripper"] = state[5:6].astype(np.float64)
        obs_dict["annotation.human.task_description"] = lang

        # Add dummy dimension for history
        for k in obs_dict:
            if isinstance(obs_dict[k], np.ndarray):
                obs_dict[k] = obs_dict[k][np.newaxis, ...]
            else:
                obs_dict[k] = [obs_dict[k]]

        # Get action chunk from policy server
        action_chunk = self.policy.get_action(obs_dict)

        # Convert to lerobot actions
        lerobot_actions = []
        action_horizon = action_chunk[f"action.{self.modality_keys[0]}"].shape[0]
        
        for i in range(action_horizon):
            action_dict = self._convert_to_lerobot_action(action_chunk, i)
            
            # Apply smoothing and velocity limiting
            if self.enable_smoothing or self.enable_velocity_limit:
                action_array = np.array([action_dict[k] for k in self.robot_state_keys])
                
                if self.enable_smoothing:
                    action_array = self.smoother.smooth(action_array)
                
                if self.enable_velocity_limit:
                    action_array = self.velocity_limiter.limit(action_array)
                
                # Convert back to dict
                action_dict = {key: action_array[j] for j, key in enumerate(self.robot_state_keys)}
            
            lerobot_actions.append(action_dict)

        # Log first action
        if lerobot_actions:
            print(f"[CLIENT] First smoothed action:")
            for joint, value in lerobot_actions[0].items():
                print(f"  {joint}: {value:.2f}")

        return lerobot_actions

    def _convert_to_lerobot_action(self, action_chunk: dict[str, np.array], idx: int) -> dict[str, float]:
        concat_action = np.concatenate(
            [np.atleast_1d(action_chunk[f"action.{key}"][idx]) for key in self.modality_keys],
            axis=0,
        )
        assert len(concat_action) == len(self.robot_state_keys), "this should be size 6"
        action_dict = {key: concat_action[i] for i, key in enumerate(self.robot_state_keys)}
        return action_dict


@dataclass
class EvalConfig:
    robot: RobotConfig
    policy_host: str = "localhost"
    policy_port: int = 5555
    action_horizon: int = 8
    lang_instruction: str = "pick up the yellow cheese and put it into the white plate"
    play_sounds: bool = False
    timeout: int = 60
    show_images: bool = False
    
    # Smoothing parameters
    enable_smoothing: bool = True
    smoothing_alpha: float = 0.3  # 0.1=very smooth, 0.5=balanced, 0.9=minimal
    enable_velocity_limit: bool = True
    max_velocity: float = 5.0  # Maximum change per joint per step
    action_sleep: float = 0.05  # Sleep between actions (seconds)


@draccus.wrap()
def eval(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Initialize robot
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    camera_keys = list(cfg.robot.cameras.keys())
    print("camera_keys: ", camera_keys)

    log_say("Initializing robot", cfg.play_sounds, blocking=True)

    language_instruction = cfg.lang_instruction
    robot_state_keys = list(robot._motors_ft.keys())
    print("robot_state_keys: ", robot_state_keys)

    # Initialize policy with smoothing
    policy = Gr00tRobotInferenceClient(
        host=cfg.policy_host,
        port=cfg.policy_port,
        camera_keys=camera_keys,
        robot_state_keys=robot_state_keys,
        enable_smoothing=cfg.enable_smoothing,
        smoothing_alpha=cfg.smoothing_alpha,
        enable_velocity_limit=cfg.enable_velocity_limit,
        max_velocity=cfg.max_velocity,
    )
    
    print(f"\n{'='*60}")
    print(f"SMOOTHING CONFIGURATION:")
    print(f"  Enable smoothing: {cfg.enable_smoothing}")
    print(f"  Smoothing alpha: {cfg.smoothing_alpha}")
    print(f"  Enable velocity limit: {cfg.enable_velocity_limit}")
    print(f"  Max velocity: {cfg.max_velocity}")
    print(f"  Action sleep: {cfg.action_sleep}s")
    print(f"{'='*60}\n")
    
    log_say(
        "Initializing policy client with language instruction: " + language_instruction,
        cfg.play_sounds,
        blocking=True,
    )

    # Eval loop
    step_count = 0
    while True:
        step_count += 1
        observation_dict = robot.get_observation()
        print(f"\n[STEP {step_count}] Getting observation...")

        # Get current robot state
        current_state = [observation_dict[k] for k in robot_state_keys]
        if step_count % 10 == 1:  # Print every 10 steps to reduce clutter
            print(f"  Current robot state:")
            for i, key in enumerate(robot_state_keys):
                print(f"    {key}: {current_state[i]:.2f}")

        action_chunk = policy.get_action(observation_dict, language_instruction)

        print(f"[STEP {step_count}] Executing {cfg.action_horizon} actions from chunk...")
        for i in range(cfg.action_horizon):
            action_dict = action_chunk[i]
            robot.send_action(action_dict)
            time.sleep(cfg.action_sleep)


if __name__ == "__main__":
    eval()

