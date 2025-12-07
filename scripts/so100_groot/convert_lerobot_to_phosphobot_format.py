#!/usr/bin/env python3
"""
Convert LeRobot v3.0 dataset to phosphobot-compatible v2.1 format.

This script performs three main conversions:
1. Reorganizes video folder structure from LeRobot format to phosphobot format
2. Fixes video codec metadata from 'h264' to 'avc1'
3. Converts metadata from v3.0 to v2.1 format

Usage:
    python convert_lerobot_to_phosphobot_format.py \\
        --source ~/.cache/huggingface/lerobot/rubbotix/cheese \\
        --target ~/.cache/huggingface/lerobot/rubbotix/cheese_phosphobot \\
        [--dry-run]
"""

import json
import shutil
from pathlib import Path
import argparse
from typing import Dict, Any


def reorganize_video_structure(source_path: Path, target_path: Path, dry_run: bool = False) -> bool:
    """
    Step 1: Reorganize video folder structure.
    
    Converts from LeRobot v3.0 format:
        videos/
          observation.images.scene/
            chunk-000/
              episode_*.mp4
          observation.images.wrist/
            chunk-000/
              episode_*.mp4
    
    To phosphobot v2.1 format:
        videos/
          chunk-000/
            observation.images.scene/
              episode_*.mp4
            observation.images.wrist/
              episode_*.mp4
    """
    print("\n" + "="*70)
    print("STEP 1: Reorganizing video folder structure")
    print("="*70)
    
    source_videos = source_path / "videos"
    target_videos = target_path / "videos"
    
    if not source_videos.exists():
        print(f"❌ Source videos directory not found: {source_videos}")
        return False
    
    # Find all camera folders in source
    camera_folders = [d for d in source_videos.iterdir() if d.is_dir()]
    
    if not camera_folders:
        print(f"❌ No camera folders found in {source_videos}")
        return False
    
    print(f"📁 Found {len(camera_folders)} camera folders:")
    for cam in camera_folders:
        print(f"   - {cam.name}")
    
    # Collect all chunks across all cameras
    chunks = set()
    for camera_folder in camera_folders:
        chunk_folders = [d for d in camera_folder.iterdir() if d.is_dir() and d.name.startswith('chunk-')]
        chunks.update(d.name for d in chunk_folders)
    
    print(f"📦 Found {len(chunks)} chunks: {sorted(chunks)}")
    
    if dry_run:
        print("\n🔍 DRY RUN MODE - Would create structure:")
        for chunk in sorted(chunks):
            print(f"   videos/{chunk}/")
            for camera_folder in sorted(camera_folders, key=lambda x: x.name):
                print(f"      {camera_folder.name}/")
        return True
    
    # Create target structure
    print(f"\n📂 Creating target directory: {target_videos}")
    target_videos.mkdir(parents=True, exist_ok=True)
    
    # Reorganize videos
    total_videos = 0
    for chunk in sorted(chunks):
        chunk_dir = target_videos / chunk
        chunk_dir.mkdir(exist_ok=True)
        
        for camera_folder in camera_folders:
            source_chunk_dir = camera_folder / chunk
            if not source_chunk_dir.exists():
                continue
            
            target_camera_dir = chunk_dir / camera_folder.name
            target_camera_dir.mkdir(exist_ok=True)
            
            # Copy all videos
            videos = list(source_chunk_dir.glob("episode_*.mp4"))
            for video in videos:
                target_video = target_camera_dir / video.name
                shutil.copy2(video, target_video)
                total_videos += 1
            
            print(f"   ✓ Copied {len(videos)} videos: {chunk}/{camera_folder.name}/")
    
    print(f"\n✅ Step 1 complete: Reorganized {total_videos} videos")
    return True


def copy_metadata_and_data(source_path: Path, target_path: Path, dry_run: bool = False) -> bool:
    """Copy metadata and data files from source to target."""
    print("\n" + "="*70)
    print("STEP 2: Copying metadata and data files")
    print("="*70)
    
    # Copy meta directory
    source_meta = source_path / "meta"
    target_meta = target_path / "meta"
    
    if not source_meta.exists():
        print(f"❌ Source meta directory not found: {source_meta}")
        return False
    
    if dry_run:
        print(f"🔍 DRY RUN MODE - Would copy:")
        print(f"   {source_meta} → {target_meta}")
        print(f"   {source_path / 'data'} → {target_path / 'data'}")
        return True
    
    print(f"📂 Copying metadata: {source_meta} → {target_meta}")
    if target_meta.exists():
        shutil.rmtree(target_meta)
    shutil.copytree(source_meta, target_meta)
    
    # Copy data directory
    source_data = source_path / "data"
    target_data = target_path / "data"
    
    if source_data.exists():
        print(f"📂 Copying data: {source_data} → {target_data}")
        if target_data.exists():
            shutil.rmtree(target_data)
        shutil.copytree(source_data, target_data)
    
    print("✅ Step 2 complete: Metadata and data copied")
    return True


def fix_metadata_for_phosphobot(target_path: Path, source_path: Path = None, dry_run: bool = False) -> bool:
    """
    Step 3: Fix metadata for phosphobot compatibility.

    Changes:
    1. Video codec: 'h264' → 'avc1'
    2. Codebase version: 'v3.0' → 'v2.1'
    3. Video path format: v3.0 → v2.1
    4. Add v2.1 fields: total_videos, total_chunks
    5. Remove v3.0-only fields
    """
    print("\n" + "="*70)
    print("STEP 3: Fixing metadata for phosphobot v2.1 compatibility")
    print("="*70)

    # In dry-run mode, read from source if target doesn't exist yet
    if dry_run and source_path:
        info_path = source_path / "meta" / "info.json"
    else:
        info_path = target_path / "meta" / "info.json"

    if not info_path.exists():
        print(f"❌ info.json not found: {info_path}")
        return False
    
    # Load the info.json
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    changes = []
    
    # 1. Fix codebase version
    current_version = info.get('codebase_version')
    if current_version == 'v3.0':
        changes.append(('codebase_version', current_version, 'v2.1'))
        if not dry_run:
            info['codebase_version'] = 'v2.1'
    
    # 2. Fix video_path format (v3.0 vs v2.1)
    current_video_path = info.get('video_path')
    v3_format = "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4"
    v2_format = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    
    if current_video_path == v3_format:
        changes.append(('video_path', 'v3.0 format', 'v2.1 format'))
        if not dry_run:
            info['video_path'] = v2_format
    
    # 3. Add total_videos and total_chunks if missing (v2.1 fields)
    if 'total_videos' not in info:
        # Count videos from features
        video_features = [k for k, v in info.get('features', {}).items() 
                         if isinstance(v, dict) and v.get('dtype') == 'video']
        total_videos = info.get('total_episodes', 0) * len(video_features)
        changes.append(('total_videos', 'missing', str(total_videos)))
        if not dry_run:
            info['total_videos'] = total_videos
    
    if 'total_chunks' not in info:
        changes.append(('total_chunks', 'missing', '1'))
        if not dry_run:
            info['total_chunks'] = 1
    
    # 4. Fix video codec and add missing video info fields
    features = info.get('features', {})
    for feature_name, feature_data in features.items():
        if isinstance(feature_data, dict) and feature_data.get('dtype') == 'video':
            video_info = feature_data.get('info', {})
            
            # Fix codec
            current_codec = video_info.get('video.codec')
            if current_codec == 'h264':
                changes.append((f'{feature_name}.video.codec', current_codec, 'avc1'))
                if not dry_run:
                    video_info['video.codec'] = 'avc1'
            
            # Add missing video info fields if not present
            shape = feature_data.get('shape', [])
            if len(shape) >= 3:
                height, width, channels = shape[0], shape[1], shape[2]
                
                if 'video.height' not in video_info:
                    changes.append((f'{feature_name}.video.height', 'missing', str(height)))
                    if not dry_run:
                        video_info['video.height'] = height
                
                if 'video.width' not in video_info:
                    changes.append((f'{feature_name}.video.width', 'missing', str(width)))
                    if not dry_run:
                        video_info['video.width'] = width
                
                if 'video.channels' not in video_info:
                    changes.append((f'{feature_name}.video.channels', 'missing', str(channels)))
                    if not dry_run:
                        video_info['video.channels'] = channels
    
    # 5. Remove v3.0-specific fields if present
    v3_only_fields = ['data_files_size_in_mb', 'video_files_size_in_mb']
    for field in v3_only_fields:
        if field in info:
            changes.append((field, 'present', 'removed'))
            if not dry_run:
                del info[field]
    
    if not changes:
        print("✅ No metadata changes needed")
        return True
    
    print(f"\n📝 Found {len(changes)} metadata changes needed:")
    for field, old_val, new_val in changes:
        print(f"   - {field}: {old_val} → {new_val}")
    
    if dry_run:
        print("\n🔍 DRY RUN MODE - No changes will be made")
        return True
    
    # Backup original file
    backup_path = info_path.with_suffix('.json.v3_backup')
    print(f"\n💾 Creating backup: {backup_path.name}")
    with open(backup_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    # Save updated info.json
    print(f"💾 Saving updated info.json...")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print("\n✅ Step 3 complete: Metadata fixed for phosphobot v2.1")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot v3.0 dataset to phosphobot-compatible v2.1 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be done
  python convert_lerobot_to_phosphobot_format.py \\
      --source ~/.cache/huggingface/lerobot/rubbotix/cheese \\
      --target ~/.cache/huggingface/lerobot/rubbotix/cheese_phosphobot \\
      --dry-run

  # Actually perform the conversion
  python convert_lerobot_to_phosphobot_format.py \\
      --source ~/.cache/huggingface/lerobot/rubbotix/cheese \\
      --target ~/.cache/huggingface/lerobot/rubbotix/cheese_phosphobot
        """
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to source LeRobot v3.0 dataset"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Path to target phosphobot v2.1 dataset (will be created)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    source_path = Path(args.source).expanduser().resolve()
    target_path = Path(args.target).expanduser().resolve()

    print("="*70)
    print("LeRobot v3.0 → Phosphobot v2.1 Dataset Converter")
    print("="*70)
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print("="*70)

    # Validate source
    if not source_path.exists():
        print(f"\n❌ Source path not found: {source_path}")
        return 1

    if not (source_path / "meta" / "info.json").exists():
        print(f"\n❌ Source is not a valid LeRobot dataset (missing meta/info.json)")
        return 1

    # Check if target already exists
    if target_path.exists() and not args.dry_run:
        response = input(f"\n⚠️  Target path already exists: {target_path}\nOverwrite? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Conversion cancelled")
            return 1
        print(f"🗑️  Removing existing target directory...")
        shutil.rmtree(target_path)

    # Step 1: Reorganize video structure
    success = reorganize_video_structure(source_path, target_path, dry_run=args.dry_run)
    if not success:
        print("\n❌ Step 1 failed: Video reorganization")
        return 1

    # Step 2: Copy metadata and data
    success = copy_metadata_and_data(source_path, target_path, dry_run=args.dry_run)
    if not success:
        print("\n❌ Step 2 failed: Metadata/data copy")
        return 1

    # Step 3: Fix metadata
    success = fix_metadata_for_phosphobot(target_path, source_path=source_path, dry_run=args.dry_run)
    if not success:
        print("\n❌ Step 3 failed: Metadata fixes")
        return 1

    # Summary
    print("\n" + "="*70)
    print("✅ CONVERSION COMPLETE!")
    print("="*70)

    if args.dry_run:
        print("\n🔍 This was a DRY RUN - no changes were made")
        print("Run without --dry-run to perform the actual conversion")
    else:
        print(f"\n📁 Phosphobot v2.1 dataset created at:")
        print(f"   {target_path}")
        print(f"\n📊 Dataset summary:")

        # Read info.json for summary
        info_path = target_path / "meta" / "info.json"
        with open(info_path, 'r') as f:
            info = json.load(f)

        print(f"   - Codebase version: {info.get('codebase_version')}")
        print(f"   - Total episodes: {info.get('total_episodes')}")
        print(f"   - Total frames: {info.get('total_frames')}")
        print(f"   - Total videos: {info.get('total_videos')}")
        print(f"   - Total chunks: {info.get('total_chunks')}")
        print(f"   - FPS: {info.get('fps')}")

        video_features = [k for k, v in info.get('features', {}).items()
                         if isinstance(v, dict) and v.get('dtype') == 'video']
        print(f"   - Camera streams: {', '.join(video_features)}")

        print(f"\n📤 Ready to upload to Hugging Face:")
        print(f"   python scripts/so100_groot/upload_to_huggingface.py \\")
        print(f"       --local-dir {target_path} \\")
        print(f"       --repo-id rubbotix/{target_path.name}")

    return 0


if __name__ == "__main__":
    exit(main())

