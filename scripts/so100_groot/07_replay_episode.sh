#!/bin/bash
# SO-100 Episode Replay Script
# Replays recorded actions from a dataset episode on the physical robot
# WITHOUT using the neural network - pure open-loop action playback
#
# Usage: ./07_replay_episode.sh [dataset_name] [episode_number]
# Examples:
#   ./07_replay_episode.sh seq1 0      # Replay episode 0 from seq1 dataset
#   ./07_replay_episode.sh cheese 5     # Replay episode 5 from cheese dataset

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

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SO-100 Episode Replay${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Dataset: ${YELLOW}${DATASET_REPO}${NC}"
echo -e "  Episode: ${YELLOW}${EPISODE}${NC}"
echo -e "  Path: ${YELLOW}${DATASETS_DIR}${NC}"
echo ""

# Check if lerobot environment exists
if ! conda env list | grep -q "lerobot"; then
    echo -e "${RED}Error: lerobot conda environment not found${NC}"
    exit 1
fi

# Activate lerobot environment
echo -e "${GREEN}Activating lerobot environment...${NC}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

# Check dataset exists
DATASET_PATH="${DATASETS_DIR}/${DATASET_REPO}"
if [ ! -d "$DATASET_PATH" ]; then
    echo -e "${RED}Error: Dataset not found at ${DATASET_PATH}${NC}"
    echo ""
    echo "Available datasets:"
    ls -la "${DATASETS_DIR}/rubbotix/" 2>/dev/null || echo "  (none found)"
    exit 1
fi

# Check device permissions
echo -e "${GREEN}Checking device permissions...${NC}"
if [ ! -w /dev/follower ]; then
    echo -e "${YELLOW}Granting device permissions...${NC}"
    sudo chmod 666 /dev/ttyACM*
fi

# Verify devices exist
echo -e "${GREEN}Verifying device mappings...${NC}"
if [ ! -e /dev/follower ]; then
    echo -e "${RED}Error: Device /dev/follower not found${NC}"
    exit 1
fi
echo -e "  ✓ /dev/follower -> $(readlink -f /dev/follower)"
echo ""

# Safety warning
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}SAFETY CHECKLIST${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""
echo "This will replay the recorded actions on the physical robot."
echo ""
echo "Before starting, ensure:"
echo "  1. Robot workspace is clear and matches the recording setup"
echo "  2. Emergency stop is accessible"
echo "  3. Objects are in similar positions as during recording"
echo "  4. You are ready to monitor robot behavior"
echo ""
echo -e "${RED}WARNING: Robot will move autonomously!${NC}"
echo -e "${RED}Keep hand near emergency stop button!${NC}"
echo ""
read -p "Press Enter when ready to replay (Ctrl+C to cancel)..."
echo ""

# Run episode replay
echo -e "${GREEN}Replaying episode ${EPISODE}...${NC}"
echo -e "${BLUE}Robot will execute actions from the recorded dataset${NC}"
echo ""

# Note: lerobot expects --dataset.root to be the FULL path including repo_id
lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=/dev/follower \
    --robot.id=so101_follower \
    --dataset.repo_id="${DATASET_REPO}" \
    --dataset.root="${DATASET_PATH}" \
    --dataset.episode="${EPISODE}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Episode replay complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Usage:${NC}"
echo "  ./07_replay_episode.sh [dataset_name] [episode_number]"
echo ""
echo -e "${BLUE}Examples:${NC}"
echo "  ./07_replay_episode.sh seq1 0      # Replay episode 0"
echo "  ./07_replay_episode.sh cheese 5     # Replay episode 5 from cheese"
echo ""

