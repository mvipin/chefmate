#!/usr/bin/env python3
"""
Convert LeRobot v3.0 dataset to GR00T format.
- Split single parquet file into per-episode files
- Update info.json with correct paths
"""

import json
import pandas as pd
import shutil
import sys
from pathlib import Path


def convert_dataset(dataset_path):
    """
    Convert LeRobot v3.0 dataset to GR00T format.
    
    Args:
        dataset_path: Path to the dataset directory
    """
    dataset_path = Path(dataset_path)
    
    print(f"Converting dataset: {dataset_path}")
    print("=" * 60)
    
    # Read the combined data file
    data_file = dataset_path / "data" / "chunk-000" / "file-000.parquet"
    
    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}")
        sys.exit(1)
    
    print(f"\n1. Reading combined data file...")
    data_df = pd.read_parquet(data_file)
    print(f"   Total frames: {len(data_df)}")
    
    # Get unique episodes
    episodes = data_df['episode_index'].unique()
    print(f"   Episodes: {len(episodes)}")
    
    # Create per-episode files
    print(f"\n2. Splitting into per-episode files...")
    data_dir = dataset_path / "data" / "chunk-000"
    
    for episode_idx in sorted(episodes):
        episode_data = data_df[data_df['episode_index'] == episode_idx]
        episode_file = data_dir / f"episode_{episode_idx:06d}.parquet"
        
        episode_data.to_parquet(episode_file, index=False)
        print(f"   ✓ Episode {episode_idx}: {len(episode_data)} frames -> {episode_file.name}")
    
    # Remove the old combined file
    print(f"\n3. Removing old combined file...")
    data_file.unlink()
    print(f"   ✓ Removed {data_file.name}")
    
    # Update info.json
    print(f"\n4. Updating info.json...")
    info_file = dataset_path / "meta" / "info.json"
    
    with open(info_file, 'r') as f:
        info = json.load(f)
    
    # Update paths to GR00T format
    info['data_path'] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info['video_path'] = "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4"
    
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"   ✓ Updated data_path to: {info['data_path']}")
    print(f"   ✓ Updated video_path to: {info['video_path']}")
    
    # Now handle videos - split them per episode
    print(f"\n5. Splitting video files per episode...")
    
    # Read episodes metadata to get video timestamps
    episodes_parquet = dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_df = pd.read_parquet(episodes_parquet)
    
    # Process each camera
    videos_dir = dataset_path / "videos"
    for camera_dir in videos_dir.iterdir():
        if not camera_dir.is_dir():
            continue
        
        camera_name = camera_dir.name  # e.g., "observation.images.wrist"
        print(f"\n   Processing {camera_name}...")
        
        video_file = camera_dir / "chunk-000" / "file-000.mp4"
        
        if not video_file.exists():
            print(f"   Warning: Video file not found: {video_file}")
            continue
        
        # Use ffmpeg to split video by timestamps
        import subprocess
        
        for idx, row in episodes_df.iterrows():
            episode_idx = int(row['episode_index'])
            
            # Get timestamps for this camera
            from_ts_col = f"videos/{camera_name}/from_timestamp"
            to_ts_col = f"videos/{camera_name}/to_timestamp"
            
            if from_ts_col not in row or to_ts_col not in row:
                print(f"   Warning: Timestamp columns not found for {camera_name}")
                continue
            
            from_ts = float(row[from_ts_col])
            to_ts = float(row[to_ts_col])
            duration = to_ts - from_ts
            
            output_file = camera_dir / "chunk-000" / f"episode_{episode_idx:06d}.mp4"
            
            # Use ffmpeg to extract segment
            cmd = [
                'ffmpeg', '-y', '-loglevel', 'error',
                '-ss', str(from_ts),
                '-i', str(video_file),
                '-t', str(duration),
                '-c', 'copy',
                str(output_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"   ✓ Episode {episode_idx}: {output_file.name}")
            else:
                print(f"   ✗ Failed to create {output_file.name}: {result.stderr}")
        
        # Remove old combined video file
        print(f"   Removing old combined video: {video_file.name}")
        video_file.unlink()
    
    print(f"\n{'=' * 60}")
    print(f"✓ Conversion complete!")
    print(f"\nDataset is now in GR00T format:")
    print(f"  - {len(episodes)} episode parquet files")
    print(f"  - {len(episodes)} video files per camera")
    print(f"  - Updated info.json with correct paths")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_to_groot_format.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    convert_dataset(dataset_path)

