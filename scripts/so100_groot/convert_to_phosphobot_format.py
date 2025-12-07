#!/usr/bin/env python3
"""
Convert LeRobot dataset to phosphobot-compatible format.

Creates a new dataset with reorganized video structure.

Current structure (LeRobot default):
videos/
  observation.images.scene/
    chunk-000/
      *.mp4
  observation.images.wrist/
    chunk-000/
      *.mp4

Required structure (phosphobot):
videos/
  chunk-000/
    observation.images.scene/
      *.mp4
    observation.images.wrist/
      *.mp4
"""

import shutil
from pathlib import Path
import argparse


def convert_to_phosphobot_format(source_path: Path, target_path: Path, dry_run: bool = False):
    """Convert dataset to phosphobot-compatible format."""
    
    if not source_path.exists():
        print(f"❌ Source dataset not found: {source_path}")
        return False
    
    print(f"📁 Source dataset: {source_path}")
    print(f"📁 Target dataset: {target_path}")
    print()
    
    # Check if target already exists
    if target_path.exists() and not dry_run:
        response = input(f"⚠️  Target path already exists: {target_path}\nOverwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Aborted")
            return False
        shutil.rmtree(target_path)
    
    videos_source = source_path / "videos"
    
    if not videos_source.exists():
        print(f"❌ Videos folder not found: {videos_source}")
        return False
    
    # Find all camera folders
    camera_folders = [f for f in videos_source.iterdir() if f.is_dir() and not f.name.startswith('.')]
    
    if not camera_folders:
        print("❌ No camera folders found in videos/")
        return False
    
    print(f"Found {len(camera_folders)} camera folders:")
    for cam in camera_folders:
        print(f"  - {cam.name}")
    print()
    
    # Collect all chunks
    chunks = set()
    for camera_folder in camera_folders:
        chunk_folders = [f for f in camera_folder.iterdir() if f.is_dir() and f.name.startswith('chunk-')]
        for chunk in chunk_folders:
            chunks.add(chunk.name)
    
    if not chunks:
        print("❌ No chunk folders found")
        return False
    
    print(f"Found {len(chunks)} chunks: {sorted(chunks)}")
    print()
    
    if dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        print()
        print("Would create:")
        print(f"  {target_path}/")
        print(f"    data/ (copied)")
        print(f"    meta/ (copied)")
        print(f"    videos/")
        for chunk_name in sorted(chunks):
            print(f"      {chunk_name}/")
            for camera_folder in camera_folders:
                video_count = len(list((camera_folder / chunk_name).glob('*.mp4'))) if (camera_folder / chunk_name).exists() else 0
                print(f"        {camera_folder.name}/ ({video_count} videos)")
        return True
    
    # Create target directory
    print("📦 Creating target dataset...")
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Copy data and meta folders
    print("📋 Copying data folder...")
    if (source_path / "data").exists():
        shutil.copytree(source_path / "data", target_path / "data")
        print("  ✅ data/ copied")
    
    print("📋 Copying meta folder...")
    if (source_path / "meta").exists():
        shutil.copytree(source_path / "meta", target_path / "meta")
        print("  ✅ meta/ copied")
    
    # Copy other files (README, .gitattributes, etc.)
    for item in source_path.iterdir():
        if item.is_file():
            shutil.copy2(item, target_path / item.name)
            print(f"  ✅ {item.name} copied")
    
    print()
    print("🔄 Reorganizing videos to phosphobot format...")
    
    videos_target = target_path / "videos"
    videos_target.mkdir(exist_ok=True)
    
    for chunk_name in sorted(chunks):
        chunk_path = videos_target / chunk_name
        chunk_path.mkdir(exist_ok=True)
        print(f"  Creating {chunk_name}/")
        
        for camera_folder in camera_folders:
            source_chunk_path = camera_folder / chunk_name
            
            if not source_chunk_path.exists():
                print(f"    ⚠️  Skipping {camera_folder.name} (not found in this chunk)")
                continue
            
            target_camera_path = chunk_path / camera_folder.name
            
            # Copy the videos
            video_files = list(source_chunk_path.glob('*.mp4'))
            print(f"    Copying {camera_folder.name}/ ({len(video_files)} videos)")
            
            shutil.copytree(source_chunk_path, target_camera_path)
    
    print()
    print("✅ Conversion complete!")
    print()
    print(f"📁 New phosphobot-compatible dataset created at:")
    print(f"   {target_path}")
    print()
    print("New structure:")
    for chunk_name in sorted(chunks):
        chunk_path = videos_target / chunk_name
        print(f"  videos/{chunk_name}/")
        for camera_path in sorted(chunk_path.iterdir()):
            if camera_path.is_dir():
                video_count = len(list(camera_path.glob('*.mp4')))
                print(f"    {camera_path.name}/ ({video_count} videos)")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot dataset to phosphobot-compatible format"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to source dataset (e.g., ~/.cache/huggingface/lerobot/rubbotix/bread)"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Path to target dataset (e.g., ~/.cache/huggingface/lerobot/rubbotix/bread_phosphobot)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    args = parser.parse_args()
    
    source_path = Path(args.source).expanduser().resolve()
    target_path = Path(args.target).expanduser().resolve()
    
    success = convert_to_phosphobot_format(source_path, target_path, dry_run=args.dry_run)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

