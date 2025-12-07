#!/bin/bash
# Visualize recorded episodes using LeRobot's visualization tool

# Activate lerobot environment
eval "$(conda shell.bash hook)"
conda activate lerobot

# Dataset path
DATASET_REPO="sparkmt/so100-striped-block"

# Default episode
EPISODE=${1:-0}

echo "========================================================================"
echo "LeRobot Episode Visualizer"
echo "========================================================================"
echo "Dataset: ${DATASET_REPO}"
echo "Episode: ${EPISODE}"
echo "========================================================================"
echo ""
echo "This will open a Rerun viewer window showing:"
echo "  - Camera feeds (scene and wrist)"
echo "  - Joint trajectories"
echo "  - Action values over time"
echo ""
echo "Usage:"
echo "  ./visualize_episodes.sh [episode_number]"
echo ""
echo "Examples:"
echo "  ./visualize_episodes.sh 0    # Visualize episode 0"
echo "  ./visualize_episodes.sh 5    # Visualize episode 5"
echo ""
echo "========================================================================"
echo ""

# Run visualization
python -m lerobot.scripts.visualize_dataset \
    --repo-id "${DATASET_REPO}" \
    --episode-index "${EPISODE}" \
    --mode local

echo ""
echo "✓ Visualization complete"

