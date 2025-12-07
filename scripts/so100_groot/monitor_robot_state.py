#!/usr/bin/env python3
"""
Replay recorded episode and display action values being sent to robot.

This script replays a recorded episode from the dataset and shows:
- Action values from the dataset (what should be executed)
- Current robot state (actual position)
- Comparison between commanded and actual values

Usage:
    python monitor_robot_state.py --episode 0
    python monitor_robot_state.py --episode 5 --speed 0.5

Press Ctrl+C to stop.
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Add lerobot to path
sys.path.insert(0, str(Path.home() / "lerobot" / "src"))

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig


def clear_lines(n):
    """Clear n lines in terminal."""
    for _ in range(n):
        sys.stdout.write("\033[F")  # Move cursor up
        sys.stdout.write("\033[K")  # Clear line


def load_episode_data(episode_num):
    """Load episode data from the recorded dataset."""
    dataset_path = Path.home() / "Isaac-GR00T" / "demo_data" / "striped-block" / "data" / "chunk-000"
    episode_file = dataset_path / f"episode_{episode_num:06d}.parquet"

    if not episode_file.exists():
        print(f"ERROR: Episode file not found: {episode_file}")
        print(f"\nAvailable episodes:")
        for f in sorted(dataset_path.glob("episode_*.parquet")):
            print(f"  {f.name}")
        sys.exit(1)

    print(f"Loading episode data from: {episode_file}")
    df = pd.read_parquet(episode_file)
    print(f"  Loaded {len(df)} timesteps")
    print()

    return df


def main():
    parser = argparse.ArgumentParser(description="Replay recorded episode")
    parser.add_argument("--episode", type=int, default=0, help="Episode number to replay (default: 0)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--start_step", type=int, default=0, help="Start from this timestep (default: 0)")
    args = parser.parse_args()

    print("=" * 70)
    print("SO-100 Episode Replay")
    print("=" * 70)
    print(f"Episode: {args.episode}")
    print(f"Playback speed: {args.speed}x")
    print(f"Start step: {args.start_step}")
    print("=" * 70)
    print()

    # Load episode data
    df = load_episode_data(args.episode)

    # Create robot config matching deployment settings
    config = SO101FollowerConfig(
        port="/dev/follower",
        use_degrees=False,  # Use RANGE_M100_100 for arm
    )

    # Check calibration
    calibration_path = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so101_follower/so101_follower.json"
    if not calibration_path.exists():
        print(f"ERROR: Calibration file not found at {calibration_path}")
        print("Please run teleoperation first to create calibration.")
        sys.exit(1)

    print(f"Using calibration: {calibration_path}")
    print()

    # Connect to robot
    print("Connecting to robot...")
    print("When prompted, press ENTER to use existing calibration")
    print()
    robot = SO101Follower(config)

    try:
        robot.connect()
    except Exception as e:
        print(f"ERROR connecting to robot: {e}")
        print("\nMake sure:")
        print("  1. Robot is powered on")
        print("  2. USB cable is connected to /dev/follower")
        print("  3. You pressed ENTER when prompted for calibration")
        sys.exit(1)

    print()
    print("✓ Connected successfully!")
    print()
    print("=" * 70)
    print("Reading joint positions... (Press Ctrl+C to stop)")
    print("Move the robot manually to see values update")
    print("=" * 70)
    print()
    
    # Joint names in order
    joint_names = [
        "shoulder_pan.pos",
        "shoulder_lift.pos", 
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos"
    ]
    
    # Display header
    header_lines = 3
    print()
    print("Current Robot State (as sent to GR00T server):")
    print("-" * 70)
    
    first_iteration = True
    try:
        while True:
            # Get observation (this returns normalized values)
            try:
                obs = robot.get_observation()
            except Exception as e:
                print(f"\nERROR reading observation: {e}")
                print("Retrying...")
                time.sleep(0.5)
                continue

            # Clear previous output (skip on first iteration)
            if not first_iteration:
                clear_lines(len(joint_names) + header_lines)
            first_iteration = False

            # Print header
            print()
            print("Current Robot State (as sent to GR00T server):")
            print("-" * 70)

            # Print each joint
            for i, joint_name in enumerate(joint_names):
                value = obs[joint_name]

                # Format based on joint type
                if i < 5:  # Arm joints (RANGE_M100_100)
                    norm_type = "RANGE_M100_100"
                    range_str = "[-100, +100]"
                else:  # Gripper (RANGE_0_100)
                    norm_type = "RANGE_0_100"
                    range_str = "[0, 100]"

                # Create visual bar
                if i < 5:
                    # For -100 to +100 range
                    normalized = (value + 100) / 200  # Convert to 0-1
                    normalized = max(0, min(1, normalized))  # Clamp to 0-1
                else:
                    # For 0 to 100 range
                    normalized = value / 100
                    normalized = max(0, min(1, normalized))  # Clamp to 0-1

                bar_width = 30
                filled = int(normalized * bar_width)
                bar = "█" * filled + "░" * (bar_width - filled)

                print(f"{joint_name:20s}: {value:7.2f} {range_str:15s} |{bar}|")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print()
        print()
        print("=" * 70)
        print("Monitoring stopped by user")
        print("=" * 70)
    
    finally:
        # Disconnect
        print()
        print("Disconnecting...")
        robot.disconnect()
        print("✓ Disconnected")


if __name__ == "__main__":
    main()

