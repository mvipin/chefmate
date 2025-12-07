#!/usr/bin/env python3
"""
Convert LeRobot v3.0 episodes parquet to GR00T episodes.jsonl format.
"""

import json
import pandas as pd
import sys
from pathlib import Path


def convert_episodes(dataset_path):
    """
    Convert episodes from parquet to jsonl format.
    
    Args:
        dataset_path: Path to the dataset directory
    """
    dataset_path = Path(dataset_path)
    
    # Read episodes parquet
    episodes_parquet = dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    
    if not episodes_parquet.exists():
        print(f"Error: Episodes file not found at {episodes_parquet}")
        sys.exit(1)
    
    print(f"Reading episodes from: {episodes_parquet}")
    episodes_df = pd.read_parquet(episodes_parquet)
    
    print(f"Found {len(episodes_df)} episodes")
    
    # Create episodes.jsonl
    episodes_jsonl = dataset_path / "meta" / "episodes.jsonl"
    
    print(f"Creating: {episodes_jsonl}")
    
    with open(episodes_jsonl, 'w') as f:
        for idx, row in episodes_df.iterrows():
            # Convert tasks to list of strings
            tasks = row['tasks']
            if hasattr(tasks, 'tolist'):  # numpy array
                tasks = tasks.tolist()
            elif not isinstance(tasks, list):
                tasks = [str(tasks)]

            episode_data = {
                "episode_index": int(row['episode_index']),
                "length": int(row['length']),
                "tasks": tasks,
            }
            f.write(json.dumps(episode_data) + '\n')
    
    print(f"✓ Created episodes.jsonl with {len(episodes_df)} episodes")
    
    # Verify the file
    with open(episodes_jsonl, 'r') as f:
        lines = f.readlines()
        print(f"✓ Verified {len(lines)} lines in episodes.jsonl")

        # Show first episode as example
        first_episode = json.loads(lines[0])
        print(f"\nExample episode:")
        print(f"  Episode: {first_episode['episode_index']}")
        print(f"  Length: {first_episode['length']} frames")
        print(f"  Task: {first_episode['tasks'][0]}")

    # Create tasks.jsonl
    print(f"\nCreating tasks.jsonl...")
    tasks_parquet = dataset_path / "meta" / "tasks.parquet"
    tasks_jsonl = dataset_path / "meta" / "tasks.jsonl"

    if tasks_parquet.exists():
        tasks_df = pd.read_parquet(tasks_parquet)
        print(f"Found {len(tasks_df)} tasks")

        with open(tasks_jsonl, 'w') as f:
            for task_name in tasks_df.index:
                task_data = {
                    "task": str(task_name),
                    "task_index": int(tasks_df.loc[task_name, 'task_index'])
                }
                f.write(json.dumps(task_data) + '\n')

        print(f"✓ Created tasks.jsonl with {len(tasks_df)} tasks")
    else:
        print(f"Warning: tasks.parquet not found, skipping tasks.jsonl creation")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_episodes_to_jsonl.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    convert_episodes(dataset_path)

