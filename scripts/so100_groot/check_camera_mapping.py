#!/usr/bin/env python3
"""
Camera Mapping Diagnostic Tool for GR00T Inference

This script helps diagnose camera mapping issues when running GR00T inference
with phosphobot. It checks:
1. Which cameras are physically available
2. What cameras the model expects
3. Suggests the correct camera mapping

Usage:
    python check_camera_mapping.py
    python check_camera_mapping.py --model-id your-hf-model-id
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV not installed. Install with: pip install opencv-python")
    sys.exit(1)


def detect_available_cameras(max_index=10):
    """Detect all available camera indices."""
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


def get_model_camera_names(model_id=None):
    """
    Get expected camera names from the model.
    For SO-100 dual camera setup, this is typically image_cam_0 and image_cam_1
    """
    # Default for SO-100 dual camera setup
    default_cameras = ["image_cam_0", "image_cam_1"]
    
    if model_id is None:
        print("ℹ️  No model ID provided, using default SO-100 dual camera setup")
        return default_cameras
    
    # Try to fetch from HuggingFace (requires huggingface_hub)
    try:
        from huggingface_hub import hf_hub_download
        
        config_path = hf_hub_download(
            repo_id=model_id,
            filename="experiment_cfg/metadata.json",
            repo_type="model"
        )
        
        with open(config_path) as f:
            metadata = json.load(f)
        
        # Extract video keys from modalities
        if "embodiment" in metadata and "modalities" in metadata["embodiment"]:
            video_keys = list(metadata["embodiment"]["modalities"]["video"].keys())
            print(f"✓ Found {len(video_keys)} camera(s) in model: {video_keys}")
            return video_keys
        else:
            print("⚠️  Could not find video keys in model metadata, using defaults")
            return default_cameras
            
    except ImportError:
        print("⚠️  huggingface_hub not installed, using default camera names")
        return default_cameras
    except Exception as e:
        print(f"⚠️  Could not fetch model config: {e}")
        print("   Using default camera names")
        return default_cameras


def generate_camera_mapping(available_cameras, model_cameras):
    """Generate the correct camera mapping."""
    mapping = {}
    
    # Map model cameras to available cameras in order
    for i, model_cam in enumerate(model_cameras):
        if i < len(available_cameras):
            # Only add to mapping if it's not a direct match
            if available_cameras[i] != i:
                mapping[model_cam] = available_cameras[i]
    
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose camera mapping for GR00T inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_camera_mapping.py
  python check_camera_mapping.py --model-id vm-atmosic/so100-groot-cheese-pickup
        """
    )
    
    parser.add_argument(
        "--model-id",
        help="HuggingFace model ID to check camera requirements"
    )
    
    parser.add_argument(
        "--max-camera-index",
        type=int,
        default=10,
        help="Maximum camera index to check (default: 10)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GR00T Camera Mapping Diagnostic Tool")
    print("=" * 60)
    print()
    
    # Step 1: Detect available cameras
    print("🔍 Detecting available cameras...")
    available = detect_available_cameras(args.max_camera_index)
    
    if not available:
        print("❌ ERROR: No cameras detected!")
        print("   Please check your camera connections")
        sys.exit(1)
    
    print(f"✓ Found {len(available)} camera(s): {available}")
    print()
    
    # Step 2: Get model camera requirements
    print("📋 Checking model camera requirements...")
    model_cameras = get_model_camera_names(args.model_id)
    print(f"   Model expects {len(model_cameras)} camera(s): {model_cameras}")
    print()
    
    # Step 3: Check for mismatches
    print("🔧 Analyzing camera mapping...")
    
    needs_mapping = False
    for i, model_cam in enumerate(model_cameras):
        if i < len(available):
            expected_id = i
            actual_id = available[i]
            
            if expected_id != actual_id:
                print(f"   ⚠️  {model_cam}: expects camera {expected_id}, but camera {actual_id} is available")
                needs_mapping = True
            else:
                print(f"   ✓ {model_cam}: camera {actual_id} matches")
        else:
            print(f"   ❌ {model_cam}: NO CAMERA AVAILABLE!")
            needs_mapping = True
    
    print()
    
    # Step 4: Generate mapping if needed
    if needs_mapping:
        print("=" * 60)
        print("⚠️  CAMERA MAPPING REQUIRED")
        print("=" * 60)
        print()
        
        mapping = generate_camera_mapping(available, model_cameras)
        
        if mapping:
            print("Add this camera mapping when starting AI control:")
            print()
            print("📋 For Phosphobot Web Interface:")
            print("   1. Go to AI Control page")
            print("   2. Click 'Show cameras mapping settings'")
            print("   3. Set the following mappings:")
            for model_cam, cam_id in mapping.items():
                print(f"      • {model_cam} → Camera {cam_id}")
            print()
            
            print("📋 For Python API:")
            print("   cameras_keys_mapping = " + json.dumps(mapping, indent=4))
            print()
            
            print("📋 For Command Line:")
            print(f"   --cameras_keys_mapping='{json.dumps(mapping)}'")
            print()
        else:
            print("✓ No explicit mapping needed (cameras match by index)")
            print()
    else:
        print("=" * 60)
        print("✅ ALL CAMERAS MATCH - NO MAPPING NEEDED")
        print("=" * 60)
        print()
        print("Your cameras are correctly aligned with the model expectations.")
        print("You can start AI control without any camera mapping.")
        print()
    
    # Step 5: Additional diagnostics
    print("=" * 60)
    print("Additional Information")
    print("=" * 60)
    print()
    print("Available cameras:")
    for cam_id in available:
        print(f"  • Camera {cam_id}: /dev/video{cam_id}")
    print()
    
    print("Model camera names:")
    for i, cam_name in enumerate(model_cameras):
        if i < len(available):
            print(f"  • {cam_name}: should use camera {available[i]}")
        else:
            print(f"  • {cam_name}: ❌ NO CAMERA AVAILABLE")
    print()
    
    print("=" * 60)
    print("For more help, see: CAMERA_MAPPING_FIX.md")
    print("=" * 60)


if __name__ == "__main__":
    main()

