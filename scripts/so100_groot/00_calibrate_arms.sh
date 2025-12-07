#!/bin/bash

# SO-100 Arms Calibration Script
# Calibrates both leader and follower arms

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
LEADER_PORT="/dev/leader"
FOLLOWER_PORT="/dev/follower"
CALIBRATION_DIR="$HOME/.cache/huggingface/lerobot/calibration"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                                ║${NC}"
echo -e "${BLUE}║           SO-100 ARMS CALIBRATION                              ║${NC}"
echo -e "${BLUE}║                                                                ║${NC}"
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo ""

# Activate lerobot environment
echo -e "${YELLOW}[1/4] Activating lerobot environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate lerobot
echo -e "${GREEN}✓ Environment activated${NC}"
echo ""

# Check device connections
echo -e "${YELLOW}[2/4] Checking device connections...${NC}"
if [ ! -e "$LEADER_PORT" ]; then
    echo -e "${RED}✗ Leader arm not found at $LEADER_PORT${NC}"
    echo "Available devices:"
    ls -la /dev/ttyACM* 2>/dev/null || echo "No ttyACM devices found"
    exit 1
fi

if [ ! -e "$FOLLOWER_PORT" ]; then
    echo -e "${RED}✗ Follower arm not found at $FOLLOWER_PORT${NC}"
    echo "Available devices:"
    ls -la /dev/ttyACM* 2>/dev/null || echo "No ttyACM devices found"
    exit 1
fi

echo -e "${GREEN}✓ Leader arm found: $LEADER_PORT${NC}"
echo -e "${GREEN}✓ Follower arm found: $FOLLOWER_PORT${NC}"
echo ""

# Check permissions
echo -e "${YELLOW}[3/4] Checking device permissions...${NC}"
if [ ! -r "$LEADER_PORT" ] || [ ! -w "$LEADER_PORT" ]; then
    echo -e "${YELLOW}⚠ Setting permissions for leader arm...${NC}"
    sudo chmod 666 "$LEADER_PORT"
fi

if [ ! -r "$FOLLOWER_PORT" ] || [ ! -w "$FOLLOWER_PORT" ]; then
    echo -e "${YELLOW}⚠ Setting permissions for follower arm...${NC}"
    sudo chmod 666 "$FOLLOWER_PORT"
fi

echo -e "${GREEN}✓ Permissions OK${NC}"
echo ""

# Check existing calibration
echo -e "${YELLOW}[4/4] Checking existing calibration...${NC}"
if [ -d "$CALIBRATION_DIR" ]; then
    echo -e "${GREEN}✓ Calibration directory exists${NC}"
    if [ "$(ls -A $CALIBRATION_DIR 2>/dev/null)" ]; then
        echo "Existing calibration files:"
        ls -lh "$CALIBRATION_DIR"
        echo ""
        read -p "Do you want to recalibrate? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Keeping existing calibration. Exiting."
            exit 0
        fi
    fi
else
    echo -e "${YELLOW}⚠ Calibration directory not found, will create${NC}"
    mkdir -p "$CALIBRATION_DIR"
fi
echo ""

# Calibrate follower arm
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}CALIBRATING FOLLOWER ARM${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Instructions:"
echo "  1. Manually move the FOLLOWER arm to the MIDDLE of its range"
echo "  2. Position all joints at approximately 50% of their travel"
echo "  3. Ensure the arm is in a comfortable, neutral position"
echo "  4. Press ENTER when ready"
echo ""
read -p "Press ENTER to start follower calibration..."
echo ""

lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.id=so101_follower

echo ""
echo -e "${GREEN}✓ Follower arm calibrated${NC}"
echo ""

# Calibrate leader arm
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}CALIBRATING LEADER ARM${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Instructions:"
echo "  1. Manually move the LEADER arm to the MIDDLE of its range"
echo "  2. Position all joints at approximately 50% of their travel"
echo "  3. Ensure the arm is in a comfortable, neutral position"
echo "  4. Press ENTER when ready"
echo ""
read -p "Press ENTER to start leader calibration..."
echo ""

lerobot-calibrate \
    --robot.type=so101_leader \
    --robot.port="$LEADER_PORT" \
    --robot.id=so101_leader

echo ""
echo -e "${GREEN}✓ Leader arm calibrated${NC}"
echo ""

# Show calibration results
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}CALIBRATION COMPLETE!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Calibration files saved to:"
echo "  $CALIBRATION_DIR"
echo ""
ls -lh "$CALIBRATION_DIR"
echo ""
echo "Next steps:"
echo "  1. Test teleoperation: ./00_test_teleoperation.sh"
echo "  2. Start recording: ./01_record_dataset.sh"
echo ""

