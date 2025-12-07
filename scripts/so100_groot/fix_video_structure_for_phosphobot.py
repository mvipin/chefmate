#!/usr/bin/env python3
"""
Fix video folder structure for phosphobot compatibility.

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


def reorganize_videos(dataset_path: Path, dry_run: bool = False):
    """Reorganize video folder structure for phosphobot compatibility."""
    
    videos_path = dataset_path / "videos"
    
    if not videos_path.exists():
        print(f"❌ Videos folder not found: {videos_path}")
        return False
    
    print(f"📁 Processing dataset: {dataset_path}")
    print(f"📹 Videos folder: {videos_path}")
    print()
    
    # Find all camera folders (e.g., observation.images.scene, observation.images.wrist)
    camera_folders = [f for f in videos_path.iterdir() if f.is_dir() and not f.name.startswith('.')]
    
    if not camera_folders:
        print("❌ No camera folders found in videos/")
        return False
    
    print(f"Found {len(camera_folders)} camera folders:")
    for cam in camera_folders:
        print(f"  - {cam.name}")
    print()
    
    # Collect all chunks across all cameras
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
    
    # Create temporary folder for reorganization
    temp_videos = dataset_path / "videos_temp"
    
    if dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        print()
        print("Would create structure:")
        for chunk_name in sorted(chunks):
            print(f"  videos_temp/{chunk_name}/")
            for camera_folder in camera_folders:
                print(f"    {camera_folder.name}/")
        return True
    
    # Create new structure
    print("🔄 Reorganizing videos...")
    temp_videos.mkdir(exist_ok=True)
    
    for chunk_name in sorted(chunks):
        chunk_path = temp_videos / chunk_name
        chunk_path.mkdir(exist_ok=True)
        print(f"  Creating {chunk_name}/")
        
        for camera_folder in camera_folders:
            old_chunk_path = camera_folder / chunk_name
            
            if not old_chunk_path.exists():
                print(f"    ⚠️  Skipping {camera_folder.name} (not found in this chunk)")
                continue
            
            new_camera_path = chunk_path / camera_folder.name
            
            # Move the entire camera folder
            print(f"    Moving {camera_folder.name}/ ({len(list(old_chunk_path.glob('*.mp4')))} videos)")
            shutil.move(str(old_chunk_path), str(new_camera_path))
    
    print()
    print("🗑️  Removing old camera folders...")
    for camera_folder in camera_folders:
        if camera_folder.exists():
            # Check if empty
            remaining = list(camera_folder.iterdir())
            if not remaining:
                camera_folder.rmdir()
                print(f"  Removed empty folder: {camera_folder.name}")
            else:
                print(f"  ⚠️  Folder not empty: {camera_folder.name} (contains {remaining})")
    
    print()
    print("🔄 Replacing old videos folder with new structure...")
    
    # Backup old videos folder
    backup_path = dataset_path / "videos_old_structure"
    if videos_path.exists():
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.move(str(videos_path), str(backup_path))
        print(f"  Backed up old structure to: videos_old_structure/")
    
    # Move new structure to videos/
    shutil.move(str(temp_videos), str(videos_path))
    print(f"  ✅ New structure in place: videos/")
    
    print()
    print("✅ Reorganization complete!")
    print()
    print("New structure:")
    for chunk_name in sorted(chunks):
        chunk_path = videos_path / chunk_name
        print(f"  videos/{chunk_name}/")
        for camera_path in sorted(chunk_path.iterdir()):
            if camera_path.is_dir():
                video_count = len(list(camera_path.glob('*.mp4')))
                print(f"    {camera_path.name}/ ({video_count} videos)")
    
    print()
    print("💡 Old structure backed up to: videos_old_structure/")
    print("   You can delete it after verifying the new structure works.")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fix video folder structure for phosphobot compatibility"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset (e.g., ~/.cache/huggingface/lerobot/rubbotix/bread)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return 1
    
    success = reorganize_videos(dataset_path, dry_run=args.dry_run)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

