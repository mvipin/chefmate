#!/usr/bin/env python3
"""
Fix metadata for phosphobot compatibility.

Changes:
1. 'h264' to 'avc1' in video codec
2. 'v3.0' to 'v2.1' in codebase_version
3. Updates video_path format for v2.1
"""

import json
from pathlib import Path
import argparse


def fix_metadata_for_phosphobot(dataset_path: Path, dry_run: bool = False):
    """Fix metadata to make it compatible with phosphobot (v2.1 format)."""

    info_path = dataset_path / "meta" / "info.json"

    if not info_path.exists():
        print(f"❌ info.json not found: {info_path}")
        return False

    print(f"📁 Dataset: {dataset_path}")
    print(f"📄 Reading: {info_path}")
    print()

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

    # 4. Fix video codec
    features = info.get('features', {})
    for feature_name, feature_data in features.items():
        if isinstance(feature_data, dict) and feature_data.get('dtype') == 'video':
            video_info = feature_data.get('info', {})
            current_codec = video_info.get('video.codec')

            if current_codec == 'h264':
                changes.append((f'{feature_name}.video.codec', current_codec, 'avc1'))
                if not dry_run:
                    video_info['video.codec'] = 'avc1'

    # 5. Remove v3.0-specific fields if present
    v3_only_fields = ['data_files_size_in_mb', 'video_files_size_in_mb']
    for field in v3_only_fields:
        if field in info:
            changes.append((field, 'present', 'removed'))
            if not dry_run:
                del info[field]

    if not changes:
        print("✅ No changes needed - metadata is already correct")
        return True

    print(f"Found {len(changes)} changes needed:")
    for field, old_val, new_val in changes:
        print(f"  - {field}: {old_val} → {new_val}")
    print()

    if dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        return True

    # Backup original file
    backup_path = info_path.with_suffix('.json.backup2')
    print(f"💾 Creating backup: {backup_path.name}")
    with open(backup_path, 'w') as f:
        json.dump(info, f, indent=2)

    # Save updated info.json
    print(f"💾 Saving updated info.json...")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    print()
    print("✅ Metadata fixed successfully for phosphobot compatibility!")
    print()
    print("Changes made:")
    for field, old_val, new_val in changes:
        print(f"  ✓ {field}: {old_val} → {new_val}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fix metadata for phosphobot compatibility (v2.1 format)"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset (e.g., ~/.cache/huggingface/lerobot/rubbotix/cheese_phosphobot)"
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

    success = fix_metadata_for_phosphobot(dataset_path, dry_run=args.dry_run)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

