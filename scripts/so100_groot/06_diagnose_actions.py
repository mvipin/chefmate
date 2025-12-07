#!/usr/bin/env python3
"""
Diagnostic script to check action predictions and compare with training data.
This helps identify if the model is predicting reasonable actions.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add Isaac-GR00T to path
sys.path.insert(0, str(Path.home() / "Isaac-GR00T"))

from gr00t.eval.service import ExternalRobotInferenceClient

def load_training_data_stats():
    """Load statistics from the training dataset to compare."""
    import pandas as pd
    
    dataset_path = Path.home() / "so100-groot-datasets" / "striped-block-groot"
    
    # Load first episode to get action statistics
    episode_path = dataset_path / "episode_000000.parquet"
    if not episode_path.exists():
        print(f"⚠️  Training data not found at {episode_path}")
        return None
    
    df = pd.read_parquet(episode_path)
    
    # Extract action columns
    action_cols = [col for col in df.columns if col.startswith('action.single_arm') or col.startswith('action.gripper')]
    
    if not action_cols:
        print("⚠️  No action columns found in training data")
        return None
    
    # Get statistics
    stats = {}
    for col in action_cols:
        stats[col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'std': df[col].std()
        }
    
    return stats

def test_inference_with_mock_data():
    """Test inference with mock observation data."""
    
    print("=" * 60)
    print("GR00T Action Diagnostics")
    print("=" * 60)
    print()
    
    # Initialize client
    policy_host = "localhost"
    policy_port = 8000
    
    print(f"Connecting to inference server at {policy_host}:{policy_port}...")
    try:
        policy_client = ExternalRobotInferenceClient(
            host=policy_host,
            port=policy_port
        )
        print("✓ Connected to inference server")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return
    
    print()
    
    # Create mock observation (similar to what robot sends)
    print("Creating mock observation data...")
    
    # Mock images (480x640x3 RGB)
    mock_scene_image = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_wrist_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Mock robot state (current joint positions in radians)
    # These are typical "home" positions for SO-100
    # NOTE: Must use float64 to match model expectations!
    mock_single_arm = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)  # 5 joints
    mock_gripper = np.array([0.0], dtype=np.float64)  # 1 gripper value
    
    task_instruction = "pick up the striped box and put it into the white plate"
    
    # Prepare request
    observation = {
        'video.scene': mock_scene_image,
        'video.wrist': mock_wrist_image,
        'state.single_arm': mock_single_arm,
        'state.gripper': mock_gripper,
    }
    
    # Add batch dimension
    request_data = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            request_data[key] = value[np.newaxis, ...]
        else:
            request_data[key] = [value]
    
    request_data['annotation.human.task_description'] = [task_instruction]
    
    print("✓ Mock observation created")
    print()
    
    # Get action prediction
    print("Requesting action prediction from model...")
    try:
        action_chunk = policy_client.get_action(request_data)
        print("✓ Received action prediction")
    except Exception as e:
        print(f"✗ Failed to get action: {e}")
        return
    
    print()
    print("=" * 60)
    print("ACTION PREDICTION ANALYSIS")
    print("=" * 60)
    print()
    
    # Analyze the action chunk
    single_arm_actions = action_chunk['action.single_arm']  # Shape: (horizon, 5)
    gripper_actions = action_chunk['action.gripper']  # Shape: (horizon, 1)
    
    horizon = single_arm_actions.shape[0]
    
    print(f"Action Horizon: {horizon} steps")
    print()
    
    # Joint names
    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
    
    print("Single Arm Actions (radians):")
    print("-" * 60)
    for i, joint_name in enumerate(joint_names):
        joint_actions = single_arm_actions[:, i]
        print(f"{joint_name:15s}: min={joint_actions.min():7.4f}, max={joint_actions.max():7.4f}, "
              f"mean={joint_actions.mean():7.4f}, std={joint_actions.std():7.4f}")
    
    print()
    print("Gripper Actions:")
    print("-" * 60)
    gripper_vals = gripper_actions[:, 0]
    print(f"{'gripper':15s}: min={gripper_vals.min():7.4f}, max={gripper_vals.max():7.4f}, "
          f"mean={gripper_vals.mean():7.4f}, std={gripper_vals.std():7.4f}")
    
    print()
    print("First 5 action steps:")
    print("-" * 60)
    for step in range(min(5, horizon)):
        print(f"Step {step}:")
        for i, joint_name in enumerate(joint_names):
            print(f"  {joint_name:15s}: {single_arm_actions[step, i]:7.4f} rad ({np.degrees(single_arm_actions[step, i]):7.2f}°)")
        print(f"  {'gripper':15s}: {gripper_actions[step, 0]:7.4f}")
        print()
    
    # Check for potential issues
    print("=" * 60)
    print("DIAGNOSTIC CHECKS")
    print("=" * 60)
    print()
    
    issues_found = False
    
    # Check 1: Are all actions zero or near-zero?
    all_actions = np.concatenate([single_arm_actions.flatten(), gripper_actions.flatten()])
    if np.abs(all_actions).max() < 0.01:
        print("⚠️  WARNING: All actions are near zero!")
        print("   This suggests the model is not predicting meaningful movements.")
        issues_found = True
    else:
        print("✓ Actions have non-zero values")
    
    # Check 2: Are actions changing over time?
    action_variance = np.var(single_arm_actions, axis=0)
    if np.all(action_variance < 0.0001):
        print("⚠️  WARNING: Actions are not changing over time!")
        print("   The model is predicting the same action for all steps.")
        issues_found = True
    else:
        print("✓ Actions vary over time")
    
    # Check 3: Are actions within reasonable bounds for SO-100?
    # SO-100 joints typically have range of about ±π radians
    if np.abs(single_arm_actions).max() > np.pi:
        print("⚠️  WARNING: Some actions exceed ±π radians!")
        print("   This might cause the robot to move to extreme positions.")
        issues_found = True
    else:
        print("✓ Actions are within reasonable bounds (±π rad)")
    
    # Check 4: Are actions very small (twitching)?
    max_action_change = np.abs(np.diff(single_arm_actions, axis=0)).max()
    if max_action_change < 0.01:
        print("⚠️  WARNING: Action changes are very small!")
        print(f"   Max change between steps: {max_action_change:.6f} rad ({np.degrees(max_action_change):.4f}°)")
        print("   This will cause twitching behavior.")
        issues_found = True
    else:
        print(f"✓ Actions have reasonable changes (max: {max_action_change:.4f} rad)")
    
    print()
    
    if issues_found:
        print("=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)
        print()
        print("The model appears to be undertrained or has issues. Consider:")
        print("1. Train for more steps (current: 2000, try: 5000-10000)")
        print("2. Check that training data has sufficient variation")
        print("3. Verify camera views match between training and inference")
        print("4. Check action normalization in the training config")
        print("5. Record more diverse demonstration episodes")
    else:
        print("✓ No obvious issues detected with action predictions")
    
    print()

if __name__ == "__main__":
    test_inference_with_mock_data()

