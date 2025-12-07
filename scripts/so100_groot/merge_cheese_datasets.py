#!/usr/bin/env python3
"""
Merge two cheese datasets (cheese.1_25 and cheese.26_50) into a single dataset (cheese).
"""

import logging
from pathlib import Path
from lerobot.datasets.aggregate import aggregate_datasets

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Define the two source datasets
    repo_ids = [
        "rubbotix/cheese.1_25",
        "rubbotix/cheese.26_50",
    ]
    
    # Define the merged dataset name
    merged_repo_id = "rubbotix/cheese"
    
    # Define the local paths
    cache_dir = Path.home() / ".cache/huggingface/lerobot"
    roots = [
        cache_dir / "rubbotix/cheese.1_25",
        cache_dir / "rubbotix/cheese.26_50",
    ]
    merged_root = cache_dir / "rubbotix/cheese"
    
    # Check if source datasets exist
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"Dataset not found: {root}")
        print(f"✓ Found dataset: {root}")
    
    # Remove merged dataset if it already exists
    if merged_root.exists():
        print(f"⚠️  Merged dataset already exists at {merged_root}")
        response = input("Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            print("Aborting merge.")
            exit(0)
        import shutil
        shutil.rmtree(merged_root)
        print(f"✓ Removed existing dataset: {merged_root}")
    
    print(f"\nMerging datasets:")
    print(f"  Source 1: {repo_ids[0]}")
    print(f"  Source 2: {repo_ids[1]}")
    print(f"  Destination: {merged_repo_id}")
    print()
    
    # Aggregate the datasets
    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=merged_repo_id,
        roots=roots,
        aggr_root=merged_root,
    )
    
    print(f"\n✓ Successfully merged datasets into: {merged_root}")
    print(f"\nYou can now push the merged dataset to Hugging Face:")
    print(f"  python -m lerobot.datasets.push_dataset_to_hub --repo-id {merged_repo_id} --local-dir {merged_root}")

