#!/usr/bin/env python3
"""
Convert AV1 videos to H.264 for torchcodec compatibility.
"""

import subprocess
import json
from pathlib import Path
from tqdm import tqdm
import shutil

def convert_video_to_h264(input_path: Path, output_path: Path):
    """Convert a video to H.264 codec."""
    cmd = [
        'ffmpeg',
        '-i', str(input_path),
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-y',  # Overwrite output file
        str(output_path)
    ]
    
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")

def update_info_json(dataset_path: Path):
    """Update info.json to reflect H.264 codec."""
    info_path = dataset_path / "meta" / "info.json"
    
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    # Update codec for all video features
    for key, feature in info['features'].items():
        if feature.get('dtype') == 'video':
            if 'info' in feature:
                feature['info']['video.codec'] = 'h264'
    
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✓ Updated info.json with H.264 codec")

def convert_dataset_videos(dataset_path: Path):
    """Convert all videos in a dataset from AV1 to H.264."""
    dataset_path = Path(dataset_path)
    videos_dir = dataset_path / "videos"
    
    if not videos_dir.exists():
        raise ValueError(f"Videos directory not found: {videos_dir}")
    
    print(f"Converting videos in: {dataset_path}")
    print("=" * 60)
    
    # Find all video directories
    video_keys = [d for d in videos_dir.iterdir() if d.is_dir()]
    
    for video_key_dir in video_keys:
        print(f"\nProcessing {video_key_dir.name}...")
        
        # Find all chunk directories
        chunk_dirs = sorted([d for d in video_key_dir.iterdir() if d.is_dir()])
        
        for chunk_dir in chunk_dirs:
            # Find all video files
            video_files = sorted(chunk_dir.glob("*.mp4"))
            
            if not video_files:
                continue
            
            print(f"  Converting {len(video_files)} videos in {chunk_dir.name}...")
            
            for video_file in tqdm(video_files, desc=f"  {chunk_dir.name}"):
                # Create temporary output file
                temp_output = video_file.with_suffix('.h264.mp4')
                
                try:
                    # Convert to H.264
                    convert_video_to_h264(video_file, temp_output)
                    
                    # Replace original with converted
                    shutil.move(str(temp_output), str(video_file))
                    
                except Exception as e:
                    print(f"    ✗ Failed to convert {video_file.name}: {e}")
                    if temp_output.exists():
                        temp_output.unlink()
                    raise
    
    # Update info.json
    update_info_json(dataset_path)
    
    print("\n" + "=" * 60)
    print("✓ All videos converted to H.264")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python convert_videos_to_h264.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = Path(sys.argv[1])
    convert_dataset_videos(dataset_path)

