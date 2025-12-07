#!/bin/bash

# SO-100 Teleoperation Test Script
# Tests leader-follower arm connection before recording

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

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                                ║${NC}"
echo -e "${BLUE}║           SO-100 TELEOPERATION TEST                            ║${NC}"
echo -e "${BLUE}║                                                                ║${NC}"
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo ""

# Activate lerobot environment
echo -e "${YELLOW}[1/5] Activating lerobot environment...${NC}"
eval "$(conda shell.bash hook)"
conda activate lerobot
echo -e "${GREEN}✓ Environment activated${NC}"
echo ""

# Check device connections
echo -e "${YELLOW}[2/5] Checking device connections...${NC}"
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
echo -e "${YELLOW}[3/5] Checking device permissions...${NC}"
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

# Check calibration
echo -e "${YELLOW}[4/5] Checking calibration...${NC}"
CALIBRATION_DIR="$HOME/.cache/huggingface/lerobot/calibration"
if [ -d "$CALIBRATION_DIR" ]; then
    echo -e "${GREEN}✓ Calibration directory found${NC}"
    echo "Calibration files:"
    ls -lh "$CALIBRATION_DIR" 2>/dev/null || echo "  (empty)"
else
    echo -e "${YELLOW}⚠ Calibration directory not found${NC}"
    echo "  Will need to calibrate during test"
fi
echo ""

# Test teleoperation
echo -e "${YELLOW}[5/5] Starting teleoperation test...${NC}"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}TELEOPERATION TEST INSTRUCTIONS${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "This will start a teleoperation session where:"
echo "  • You move the LEADER arm"
echo "  • The FOLLOWER arm mirrors your movements"
echo ""
echo "What to test:"
echo "  1. Move leader arm slowly - verify follower mirrors correctly"
echo "  2. Test all joints - shoulder, elbow, wrist, gripper"
echo "  3. Check for any lag or jerkiness"
echo "  4. Verify gripper opens/closes properly"
echo "  5. Test full range of motion"
echo ""
echo -e "${YELLOW}Safety:${NC}"
echo "  • Keep clear of follower arm workspace"
echo "  • Have emergency stop ready"
echo "  • Start with slow, small movements"
echo ""
echo -e "${GREEN}To exit: Press Ctrl+C${NC}"
echo ""
read -p "Press ENTER to start teleoperation test..."
echo ""

# Start teleoperation
echo -e "${GREEN}Starting teleoperation...${NC}"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}TELEOPERATION ACTIVE - Move the leader arm${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Run teleoperation command
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.id=so101_follower \
    --teleop.type=so101_leader \
    --teleop.port="$LEADER_PORT" \
    --teleop.id=so101_leader

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Teleoperation test completed!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Next steps:"
echo "  1. If teleoperation worked well → Run ./01_record_dataset.sh"
echo "  2. If there were issues → Check calibration or device connections"
echo ""

