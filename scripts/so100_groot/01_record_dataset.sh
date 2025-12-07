#!/bin/bash
# SO-100 GR00T Dataset Recording Script
# Phase 1: Record demonstration dataset via teleoperation
#
# Usage: ./01_record_dataset.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATASET_NAME="bread"
NUM_EPISODES=50
TASK_DESCRIPTION="Pick slice of bread and place it in the white plate"
EPISODE_TIME=30
RESET_TIME=10
USE_TELEOPERATION=true  # Set to false if recording without leader arm

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SO-100 GR00T Dataset Recording${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Dataset Name: ${YELLOW}${DATASET_NAME}${NC}"
echo -e "  Episodes: ${YELLOW}${NUM_EPISODES}${NC}"
echo -e "  Task: ${YELLOW}${TASK_DESCRIPTION}${NC}"
echo -e "  Episode Time: ${YELLOW}${EPISODE_TIME}s${NC}"
echo -e "  Reset Time: ${YELLOW}${RESET_TIME}s${NC}"
echo -e "  Recording Mode: ${YELLOW}Teleoperation (Leader Arm)${NC}"
echo ""

# Check if lerobot environment exists
if ! conda env list | grep -q "lerobot"; then
    echo -e "${RED}Error: lerobot conda environment not found${NC}"
    echo "Please create it first with: conda create -n lerobot python=3.10"
    exit 1
fi

# Activate lerobot environment
echo -e "${GREEN}Activating lerobot environment...${NC}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

# Check device permissions
echo -e "${GREEN}Checking device permissions...${NC}"
if [ ! -w /dev/follower ] || [ ! -w /dev/leader ]; then
    echo -e "${YELLOW}Granting device permissions...${NC}"
    sudo chmod 666 /dev/ttyACM*
fi

# Verify devices exist
echo -e "${GREEN}Verifying device mappings...${NC}"
for device in /dev/follower /dev/leader /dev/wrist /dev/scene; do
    if [ ! -e "$device" ]; then
        echo -e "${RED}Error: Device $device not found${NC}"
        echo "Please check your udev rules and device connections"
        exit 1
    fi
    echo -e "  ✓ $device -> $(readlink -f $device)"
done
echo ""

# Check calibration
echo -e "${GREEN}Checking calibration files...${NC}"
CALIB_DIR="$HOME/.cache/huggingface/lerobot/calibration"
if [ ! -d "$CALIB_DIR" ] || [ -z "$(ls -A $CALIB_DIR 2>/dev/null)" ]; then
    echo -e "${YELLOW}Warning: Calibration files not found${NC}"
    echo "You may need to run calibration first:"
    echo "  lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/follower --robot.id=so100_groot_arm"
    echo "  lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/leader --teleop.id=so100_groot_leader"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "  ✓ Calibration files found"
fi
echo ""

# Find cameras
echo -e "${GREEN}Detecting cameras...${NC}"
echo "Running camera detection (this may take a moment)..."
lerobot-find-cameras opencv 2>&1 | grep -E "Camera #|Name:|Id:" || true
echo ""

# Prepare workspace
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}WORKSPACE PREPARATION${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""
echo "Before starting recording, please ensure:"
echo "  1. Robot workspace is clear and well-lit"
echo "  2. Striped block is prepared and in starting position"
echo "  3. White plate is positioned consistently"
echo "  4. Both cameras have clear view of workspace"
echo "  5. Leader arm is connected and calibrated"
echo "  6. Leader arm is in comfortable starting position"
echo "  7. You can comfortably reach and manipulate the leader arm"
echo ""
echo -e "${BLUE}Teleoperation Mode:${NC}"
echo "  • You will control the follower arm using the leader arm"
echo "  • Move the leader arm smoothly and deliberately"
echo "  • The follower arm will mirror your movements"
echo "  • Practice a few movements before recording if needed"
echo ""
read -p "Press Enter when ready to start recording..."
echo ""

# Display recording instructions
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}TELEOPERATION RECORDING INSTRUCTIONS${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Keyboard controls during recording:"
echo "  → (Right Arrow): Skip to next episode"
echo "  ← (Left Arrow): Re-record current episode"
echo "  ESC: Stop recording and save dataset"
echo ""
echo "Tips for good teleoperation demonstrations:"
echo "  • Move the leader arm smoothly and deliberately"
echo "  • Avoid jerky or sudden movements"
echo "  • Keep similar timing and motion patterns across episodes"
echo "  • Ensure successful task completion in each episode"
echo "  • Maintain consistent starting positions for objects"
echo "  • The follower arm will mirror your leader arm movements"
echo "  • Take your time - quality is more important than speed"
echo ""
echo -e "${YELLOW}Important:${NC}"
echo "  • You are controlling the follower arm via the leader arm"
echo "  • Watch the follower arm and cameras, not just the leader"
echo "  • If a demonstration fails, use ← to re-record it"
echo ""
read -p "Press Enter to begin recording..."
echo ""

# Start recording
echo -e "${GREEN}Starting dataset recording...${NC}"
echo -e "${YELLOW}Recording ${NUM_EPISODES} episodes...${NC}"
echo ""

# Record dataset with scene and wrist cameras at 640x480 (smooth recording)
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/follower \
    --robot.id=so101_follower \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: /dev/wrist, width: 640, height: 480, fps: 30}, scene: {type: opencv, index_or_path: /dev/scene, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/leader \
    --teleop.id=so101_leader \
    --display_data=false \
    --dataset.repo_id="rubbotix/${DATASET_NAME}" \
    --dataset.num_episodes="${NUM_EPISODES}" \
    --dataset.single_task="${TASK_DESCRIPTION}" \
    --dataset.push_to_hub=false \
    --dataset.episode_time_s="${EPISODE_TIME}" \
    --dataset.reset_time_s="${RESET_TIME}"

# Check if recording was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Recording completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    
    # Dataset location
    DATASET_PATH="$HOME/.cache/huggingface/lerobot/rubbotix/${DATASET_NAME}"
    echo "Dataset saved to:"
    echo "  ${DATASET_PATH}"
    echo ""
    
    # Show dataset info
    if [ -d "$DATASET_PATH" ]; then
        echo "Dataset structure:"
        tree -L 2 "$DATASET_PATH" 2>/dev/null || ls -R "$DATASET_PATH"
        echo ""
        
        # Count episodes
        if [ -d "$DATASET_PATH/videos/chunk-000" ]; then
            EPISODE_COUNT=$(ls -1 "$DATASET_PATH/videos/chunk-000" | wc -l)
            echo "Episodes recorded: ${EPISODE_COUNT}"
        fi
    fi
    
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "  1. Review dataset quality"
    echo "  2. Run: ./02_prepare_dataset.sh"
    echo ""
else
    echo ""
    echo -e "${RED}Recording failed or was interrupted${NC}"
    echo "Check the error messages above"
    exit 1
fi

