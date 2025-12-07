#!/usr/bin/env python3
"""
Upload LeRobot dataset to Hugging Face Hub.

This script uploads your local LeRobot dataset to Hugging Face Hub
so you can:
1. Use the visualization tool
2. Share the dataset
3. Train models from the cloud

Usage:
    python upload_to_huggingface.py --repo-id sparkmt/so100-striped-block
"""

import argparse
import sys
from pathlib import Path

# Add lerobot to path
sys.path.insert(0, str(Path.home() / "lerobot" / "src"))

from huggingface_hub import HfApi, create_repo


def upload_dataset(local_dir: Path, repo_id: str, private: bool = False):
    """Upload dataset to Hugging Face Hub."""
    
    print("=" * 70)
    print("Upload LeRobot Dataset to Hugging Face Hub")
    print("=" * 70)
    print(f"Local directory: {local_dir}")
    print(f"Repository ID: {repo_id}")
    print(f"Private: {private}")
    print("=" * 70)
    print()
    
    # Check if local directory exists
    if not local_dir.exists():
        print(f"ERROR: Local directory not found: {local_dir}")
        sys.exit(1)
    
    # Check for required files (tasks can be .json or .parquet)
    required_files = ["meta/info.json", "meta/stats.json"]
    missing_files = []
    for file in required_files:
        if not (local_dir / file).exists():
            missing_files.append(file)

    # Check for tasks file (either format)
    has_tasks = (local_dir / "meta/tasks.json").exists() or (local_dir / "meta/tasks.parquet").exists()
    if not has_tasks:
        missing_files.append("meta/tasks.json or meta/tasks.parquet")

    if missing_files:
        print("ERROR: Missing required metadata files:")
        for file in missing_files:
            print(f"  - {file}")
        print()
        print("Make sure your dataset is in LeRobot v3.0 format.")
        sys.exit(1)

    print("✓ All required metadata files found")
    print()
    
    # Initialize Hugging Face API
    api = HfApi()
    
    # Check if user is logged in
    try:
        user_info = api.whoami()
        print(f"✓ Logged in as: {user_info['name']}")
        print()
    except Exception as e:
        print("ERROR: Not logged in to Hugging Face")
        print()
        print("Please run: huggingface-cli login")
        print()
        sys.exit(1)
    
    # Create repository if it doesn't exist
    print(f"Creating repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True
        )
        print(f"✓ Repository created/verified: https://huggingface.co/datasets/{repo_id}")
        print()
    except Exception as e:
        print(f"ERROR creating repository: {e}")
        sys.exit(1)
    
    # Upload files
    print("Uploading dataset files...")
    print("This may take several minutes depending on dataset size...")
    print()
    
    try:
        # Upload the entire directory
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload SO-100 striped block dataset"
        )
        
        print()
        print("=" * 70)
        print("✓ Dataset uploaded successfully!")
        print("=" * 70)
        print()
        print(f"View your dataset at:")
        print(f"  https://huggingface.co/datasets/{repo_id}")
        print()
        print(f"Visualize episodes:")
        print(f"  python -m lerobot.scripts.visualize_dataset \\")
        print(f"      --repo-id {repo_id} \\")
        print(f"      --episode-index 0")
        print()
        
    except Exception as e:
        print()
        print(f"ERROR uploading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Upload LeRobot dataset to Hugging Face Hub")
    parser.add_argument(
        "--local-dir",
        type=str,
        default=str(Path.home() / "lerobot_datasets" / "so100-striped-block"),
        help="Local directory containing the dataset (default: ~/lerobot_datasets/so100-striped-block)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="sparkmt/so100-striped-block",
        help="Hugging Face repository ID (default: sparkmt/so100-striped-block)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    
    args = parser.parse_args()
    
    local_dir = Path(args.local_dir)
    upload_dataset(local_dir, args.repo_id, args.private)


if __name__ == "__main__":
    main()

