#!/bin/bash
# SO-100 GR00T Inference Server Script
# Phase 4a: Start inference server for model deployment
#
# Usage: ./04_start_inference_server.sh [checkpoint_step]
# Example: ./04_start_inference_server.sh 1000

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATASET_NAME="cheese_bread_multitask"
#DATASET_NAME="cheese"
CHECKPOINT_DIR="$HOME/so100-groot-checkpoints/${DATASET_NAME}"
#CHECKPOINT_STEP="${1:-3000}"  # Default to step 3000 (latest checkpoint)
CHECKPOINT_STEP="${1:-10000}"  # Default to step 3000 (latest checkpoint)
PORT=8000

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SO-100 GR00T Inference Server${NC}"
echo -e "${GREEN}========================================${NC}"
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

# Find checkpoint
CHECKPOINT_PATH="${CHECKPOINT_DIR}/checkpoint-${CHECKPOINT_STEP}"

# If specific checkpoint not found, try to find latest
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo -e "${YELLOW}Checkpoint ${CHECKPOINT_STEP} not found, searching for latest...${NC}"
    CHECKPOINT_PATH=$(ls -td "${CHECKPOINT_DIR}"/checkpoint-* 2>/dev/null | head -1)
    
    if [ -z "$CHECKPOINT_PATH" ]; then
        echo -e "${RED}Error: No checkpoints found in ${CHECKPOINT_DIR}${NC}"
        echo "Please run 03_train_model.sh first"
        exit 1
    fi
    
    CHECKPOINT_STEP=$(basename "$CHECKPOINT_PATH" | sed 's/checkpoint-//')
    echo -e "${GREEN}Using latest checkpoint: ${CHECKPOINT_STEP}${NC}"
fi

echo -e "${BLUE}Configuration:${NC}"
echo -e "  Checkpoint: ${YELLOW}${CHECKPOINT_PATH}${NC}"
echo -e "  Port: ${YELLOW}${PORT}${NC}"
echo ""

# Check if port is already in use
if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Port ${PORT} is already in use${NC}"
    echo "Existing process:"
    lsof -Pi :${PORT} -sTCP:LISTEN
    echo ""
    read -p "Kill existing process and continue? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Killing existing process..."
        lsof -ti:${PORT} | xargs kill -9 2>/dev/null || true
        sleep 2
    else
        exit 1
    fi
fi

# Check GPU
echo -e "${GREEN}Checking GPU availability...${NC}"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
echo ""

# Change to Isaac-GR00T directory
cd "$HOME/Isaac-GR00T"

# Display server info
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}SERVER INFORMATION${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""
echo "The inference server will:"
echo "  • Load the trained model from checkpoint ${CHECKPOINT_STEP}"
echo "  • Listen on http://localhost:${PORT}"
echo "  • Serve predictions for robot control"
echo ""
echo "Expected VRAM usage: ~6-8GB"
echo "Expected latency: ~50-80ms per inference"
echo ""
echo "To test the server (in another terminal):"
echo "  curl http://localhost:${PORT}/health"
echo ""
echo "To deploy on robot (in another terminal):"
echo "  ./05_deploy_robot.sh"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""
read -p "Press Enter to start inference server..."
echo ""

# Start inference server
echo -e "${GREEN}Starting inference server...${NC}"
echo -e "${BLUE}Server will run in foreground. Keep this terminal open.${NC}"
echo ""

python scripts/inference_service.py \
    --model-path "${CHECKPOINT_PATH}" \
    --server \
    --port ${PORT} \
    --embodiment_tag new_embodiment \
    --data_config so100_dualcam

# This will only execute if server stops
echo ""
echo -e "${YELLOW}Inference server stopped${NC}"

