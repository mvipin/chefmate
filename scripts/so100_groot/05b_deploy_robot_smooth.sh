#!/bin/bash
# SO-100 GR00T Robot Deployment Script with Action Smoothing
# This version uses action filtering to reduce jitter and shaking
#
# Usage: ./05b_deploy_robot_smooth.sh [SMOOTHING_LEVEL]
#
# SMOOTHING_LEVEL options:
#   light    - Minimal smoothing (alpha=0.7, max_vel=10.0, sleep=0.03)
#   medium   - Balanced smoothing (alpha=0.4, max_vel=5.0, sleep=0.05) [DEFAULT]
#   heavy    - Maximum smoothing (alpha=0.2, max_vel=3.0, sleep=0.08)
#   custom   - Prompts for custom parameters
#
# NOTE: Run 04_start_inference_server.sh in another terminal first!

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TASK_INSTRUCTION="pick up the yellow cheese and put it into the white plate"
POLICY_HOST="localhost"
POLICY_PORT=8000

# Default smoothing level
SMOOTHING_LEVEL="${1:-medium}"

# Set smoothing parameters based on level
case "$SMOOTHING_LEVEL" in
    light)
        SMOOTHING_ALPHA=0.7
        MAX_VELOCITY=10.0
        ACTION_SLEEP=0.03
        ;;
    medium)
        SMOOTHING_ALPHA=0.4
        MAX_VELOCITY=5.0
        ACTION_SLEEP=0.05
        ;;
    heavy)
        SMOOTHING_ALPHA=0.2
        MAX_VELOCITY=3.0
        ACTION_SLEEP=0.08
        ;;
    custom)
        echo -e "${YELLOW}Custom smoothing parameters:${NC}"
        read -p "Smoothing alpha (0.1-0.9, lower=smoother): " SMOOTHING_ALPHA
        read -p "Max velocity (1.0-10.0, lower=smoother): " MAX_VELOCITY
        read -p "Action sleep (0.02-0.1, higher=smoother): " ACTION_SLEEP
        ;;
    *)
        echo -e "${RED}Invalid smoothing level: $SMOOTHING_LEVEL${NC}"
        echo "Valid options: light, medium, heavy, custom"
        exit 1
        ;;
esac

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SO-100 GR00T Robot Deployment (SMOOTHED)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Task: ${YELLOW}${TASK_INSTRUCTION}${NC}"
echo -e "  Policy Server: ${YELLOW}${POLICY_HOST}:${POLICY_PORT}${NC}"
echo ""
echo -e "${BLUE}Smoothing Configuration (${SMOOTHING_LEVEL}):${NC}"
echo -e "  Smoothing Alpha: ${YELLOW}${SMOOTHING_ALPHA}${NC} (lower = smoother)"
echo -e "  Max Velocity: ${YELLOW}${MAX_VELOCITY}${NC} (lower = smoother)"
echo -e "  Action Sleep: ${YELLOW}${ACTION_SLEEP}s${NC} (higher = smoother)"
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

# Check if inference server is running
echo -e "${GREEN}Checking inference server...${NC}"
if ! netstat -tuln 2>/dev/null | grep -q ":${POLICY_PORT}.*LISTEN" && ! ss -tuln 2>/dev/null | grep -q ":${POLICY_PORT}.*LISTEN"; then
    echo -e "${RED}Error: No server listening on port ${POLICY_PORT}${NC}"
    echo ""
    echo "Please start the inference server first:"
    echo "  ./04_start_inference_server.sh"
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
echo "  4. Yellow cheese is in starting position"
echo "  5. White plate is positioned correctly"
echo "  6. Both cameras have clear view"
echo "  7. You are ready to monitor robot behavior"
echo ""
echo -e "${BLUE}INFO: This version uses action smoothing to reduce jitter${NC}"
echo -e "${RED}WARNING: Robot will move autonomously!${NC}"
echo -e "${RED}Keep hand near emergency stop button!${NC}"
echo ""
read -p "Press Enter when ready to deploy (Ctrl+C to cancel)..."
echo ""

# Create log file
LOG_DIR="$HOME/so100-groot-checkpoints/deployment_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/deployment_smooth_${SMOOTHING_LEVEL}_$(date +%Y%m%d_%H%M%S).log"
echo "Deployment log: ${LOG_FILE}"
echo ""

# Display deployment info
echo -e "${GREEN}Starting robot deployment with smoothing...${NC}"
echo -e "${BLUE}Robot will execute: ${TASK_INSTRUCTION}${NC}"
echo ""
echo "Monitoring:"
echo "  • Watch for smoother motion compared to unfiltered version"
echo "  • Press Ctrl+C to stop at any time"
echo "  • Check log file for detailed information"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the robot${NC}"
echo ""

# Run deployment with smoothing
cd ~/lerobot
python scripts/so100_groot/eval_lerobot_smooth.py \
    --robot.type=so101_follower \
    --robot.port=/dev/follower \
    --robot.id=so101_follower \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, scene: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --policy_host="${POLICY_HOST}" \
    --policy_port="${POLICY_PORT}" \
    --lang_instruction="${TASK_INSTRUCTION}" \
    --enable_smoothing=true \
    --smoothing_alpha="${SMOOTHING_ALPHA}" \
    --enable_velocity_limit=true \
    --max_velocity="${MAX_VELOCITY}" \
    --action_sleep="${ACTION_SLEEP}" \
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

