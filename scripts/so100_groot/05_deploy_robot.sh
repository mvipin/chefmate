#!/bin/bash
# SO-100 GR00T Robot Deployment Script
# Phase 4b: Deploy trained model on physical robot
#
# Usage: ./05_deploy_robot.sh
#
# NOTE: Run 04_start_inference_server.sh in another terminal first!

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration (matching your previous successful deployment)
#TASK_INSTRUCTION="pick up the yellow cheese and put it into the white plate"
TASK_INSTRUCTION="Do not pick up the yellow cheese. Stay at base"
POLICY_HOST="localhost"
POLICY_PORT=8000

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SO-100 GR00T Robot Deployment${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Task: ${YELLOW}${TASK_INSTRUCTION}${NC}"
echo -e "  Policy Server: ${YELLOW}${POLICY_HOST}:${POLICY_PORT}${NC}"
echo ""

# Check if gr00t environment exists
if ! conda env list | grep -q "gr00t"; then
    echo -e "${RED}Error: gr00t conda environment not found${NC}"
    exit 1
fi

# Activate gr00t environment
echo -e "${GREEN}Activating gr00t environment...${NC}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gr00t

# Check if inference server is running (ZMQ server on TCP port)
echo -e "${GREEN}Checking inference server...${NC}"
if ! netstat -tuln 2>/dev/null | grep -q ":${POLICY_PORT}.*LISTEN" && ! ss -tuln 2>/dev/null | grep -q ":${POLICY_PORT}.*LISTEN"; then
    echo -e "${RED}Error: No server listening on port ${POLICY_PORT}${NC}"
    echo ""
    echo "Please start the inference server first:"
    echo "  ./04_start_inference_server.sh"
    echo ""
    echo "In a separate terminal window!"
    exit 1
fi
echo -e "  ✓ Inference server is listening on port ${POLICY_PORT}"
echo ""

# Check device permissions
echo -e "${GREEN}Checking device permissions...${NC}"
if [ ! -w /dev/follower ]; then
    echo -e "${YELLOW}Granting device permissions...${NC}"
    sudo chmod 666 /dev/ttyACM*
fi

# Verify devices exist
echo -e "${GREEN}Verifying device mappings...${NC}"
for device in /dev/follower /dev/wrist /dev/scene; do
    if [ ! -e "$device" ]; then
        echo -e "${RED}Error: Device $device not found${NC}"
        exit 1
    fi
    echo -e "  ✓ $device -> $(readlink -f $device)"
done
echo ""

# Safety check
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}SAFETY CHECKLIST${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""
echo "Before deploying, ensure:"
echo "  1. ✓ Inference server is running (checked above)"
echo "  2. Robot workspace is clear"
echo "  3. Emergency stop is accessible"
echo "  4. Striped block is in starting position"
echo "  5. White plate is positioned correctly"
echo "  6. Both cameras have clear view"
echo "  7. You are ready to monitor robot behavior"
echo ""
echo -e "${RED}WARNING: Robot will move autonomously!${NC}"
echo -e "${RED}Keep hand near emergency stop button!${NC}"
echo ""
read -p "Press Enter when ready to deploy (Ctrl+C to cancel)..."
echo ""

# Create log file
LOG_DIR="$HOME/so100-groot-checkpoints/deployment_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/deployment_$(date +%Y%m%d_%H%M%S).log"
echo "Deployment log: ${LOG_FILE}"
echo ""

# Change to Isaac-GR00T examples directory
cd "$HOME/Isaac-GR00T/examples/SO-100"

# Display deployment info
echo -e "${GREEN}Starting robot deployment...${NC}"
echo -e "${BLUE}Robot will execute: ${TASK_INSTRUCTION}${NC}"
echo ""
echo "Monitoring:"
echo "  • Watch robot motion carefully"
echo "  • Press Ctrl+C to stop at any time"
echo "  • Check log file for detailed information"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the robot${NC}"
echo ""

# Run deployment (using your proven configuration with camera indices 0 and 2)
# NOTE: use_degrees=false (default) because the GR00T model outputs actions in RANGE_M100_100 format (-100 to +100)
python eval_lerobot.py \
    --robot.type=so101_follower \
    --robot.port=/dev/follower \
    --robot.id=so101_follower \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, scene: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --policy_host="${POLICY_HOST}" \
    --policy_port="${POLICY_PORT}" \
    --lang_instruction="${TASK_INSTRUCTION}" \
    2>&1 | tee "$LOG_FILE"

# Check deployment result
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Deployment completed${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Log saved to: ${LOG_FILE}"
    echo ""
else
    echo ""
    echo -e "${RED}Deployment stopped or failed${NC}"
    echo "Check log file: ${LOG_FILE}"
    exit 1
fi

