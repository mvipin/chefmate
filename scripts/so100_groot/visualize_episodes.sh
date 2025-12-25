#!/bin/bash
# Visualize recorded episodes using LeRobot's visualization tool with Rerun.io
#
# Usage: ./visualize_episodes.sh [dataset_name] [episode_number]
# Examples:
#   ./visualize_episodes.sh                  # Visualize episode 0 of seq1
#   ./visualize_episodes.sh seq1 5          # Visualize episode 5 of seq1
#   ./visualize_episodes.sh cheese 10        # Visualize episode 10 of cheese

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Consolidated ChefMate directory structure
CHEFMATE_DIR="$HOME/chefmate"
DATASETS_DIR="${CHEFMATE_DIR}/datasets/lerobot"

# Configuration
DATASET_NAME="${1:-seq1}"
EPISODE="${2:-0}"
DATASET_REPO="rubbotix/${DATASET_NAME}"

# Activate lerobot environment
echo -e "${GREEN}Activating lerobot environment...${NC}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}LeRobot Episode Visualizer (Rerun.io)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Dataset: ${YELLOW}${DATASET_REPO}${NC}"
echo -e "  Path: ${YELLOW}${DATASETS_DIR}${NC}"
echo -e "  Episode: ${YELLOW}${EPISODE}${NC}"
echo ""

# Check if dataset exists
DATASET_PATH="${DATASETS_DIR}/${DATASET_REPO}"
if [ ! -d "$DATASET_PATH" ]; then
    echo -e "${RED}Error: Dataset not found at ${DATASET_PATH}${NC}"
    echo ""
    echo "Available datasets in ${DATASETS_DIR}:"
    ls -la "${DATASETS_DIR}/rubbotix/" 2>/dev/null || echo "  (none found)"
    echo ""
    echo "If your dataset is in the old location (~/.cache/huggingface/lerobot/),"
    echo "migrate it with:"
    echo "  cp -r ~/.cache/huggingface/lerobot/rubbotix/${DATASET_NAME} ${DATASETS_DIR}/rubbotix/"
    exit 1
fi

echo -e "This will open a Rerun viewer window showing:"
echo "  - Camera feeds (front and wrist)"
echo "  - Joint trajectories (observation.state)"
echo "  - Action values over time"
echo ""
echo -e "${YELLOW}Usage:${NC}"
echo "  ./visualize_episodes.sh [dataset_name] [episode_number]"
echo ""
echo -e "${YELLOW}Examples:${NC}"
echo "  ./visualize_episodes.sh                  # Visualize episode 0 of seq1"
echo "  ./visualize_episodes.sh seq1 5          # Visualize episode 5 of seq1"
echo "  ./visualize_episodes.sh cheese 10        # Visualize episode 10 of cheese"
echo ""
echo -e "${GREEN}========================================${NC}"
echo ""

# Run visualization with Rerun
# Note: lerobot expects --root to be the FULL path including repo_id
echo -e "${GREEN}Launching Rerun visualization...${NC}"
python -m lerobot.scripts.visualize_dataset \
    --repo-id "${DATASET_REPO}" \
    --root "${DATASET_PATH}" \
    --episode-index "${EPISODE}" \
    --mode local

echo ""
echo -e "${GREEN}✓ Visualization complete${NC}"

