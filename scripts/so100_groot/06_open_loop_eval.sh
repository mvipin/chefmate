#!/bin/bash
# SO-100 GR00T Open-Loop Evaluation Script
# Evaluates trained model by comparing predicted actions against ground truth dataset
#
# Usage: ./06_open_loop_eval.sh [checkpoint_name] [dataset_name] [num_trajectories] [steps] [checkpoint_step]
# Examples:
#   ./06_open_loop_eval.sh                                    # Evaluate seq checkpoint (latest) on seq2 dataset
#   ./06_open_loop_eval.sh seq seq2 1 500 latest              # Explicitly use latest checkpoint
#   ./06_open_loop_eval.sh seq seq2 1 500 1000                # Use checkpoint-1000
#   ./06_open_loop_eval.sh cheese_bread_multitask cheese 5 500 2000  # Use checkpoint-2000

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Consolidated ChefMate directory structure
CHEFMATE_DIR="$HOME/chefmate"
GROOT_DATASETS_DIR="${CHEFMATE_DIR}/datasets/groot_format"
CHECKPOINTS_DIR="${CHEFMATE_DIR}/checkpoints"
ISAAC_GROOT_DIR="$HOME/Isaac-GR00T"

# Configuration
CHECKPOINT_NAME="${1:-seq}"
DATASET_NAME="${2:-seq2}"
NUM_TRAJECTORIES="${3:-1}"
STEPS="${4:-500}"
CHECKPOINT_STEP="${5:-latest}"

# Find checkpoint directory
CHECKPOINT_DIR="${CHECKPOINTS_DIR}/${CHECKPOINT_NAME}"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo -e "${RED}Error: Checkpoint directory not found at ${CHECKPOINT_DIR}${NC}"
    echo ""
    echo "Available checkpoints:"
    ls -la "${CHECKPOINTS_DIR}/" 2>/dev/null || echo "  (none found)"
    exit 1
fi

# Find checkpoint based on step parameter
if [ "$CHECKPOINT_STEP" = "latest" ]; then
    # Find latest checkpoint step
    SELECTED_CHECKPOINT=$(ls -td "${CHECKPOINT_DIR}"/checkpoint-* 2>/dev/null | head -1)
    if [ -z "$SELECTED_CHECKPOINT" ]; then
        echo -e "${RED}Error: No checkpoint-* directories found in ${CHECKPOINT_DIR}${NC}"
        exit 1
    fi
else
    # Use specified checkpoint step
    SELECTED_CHECKPOINT="${CHECKPOINT_DIR}/checkpoint-${CHECKPOINT_STEP}"
    if [ ! -d "$SELECTED_CHECKPOINT" ]; then
        echo -e "${RED}Error: Checkpoint not found at ${SELECTED_CHECKPOINT}${NC}"
        echo ""
        echo "Available checkpoint steps in ${CHECKPOINT_NAME}:"
        ls -d "${CHECKPOINT_DIR}"/checkpoint-* 2>/dev/null | xargs -n1 basename || echo "  (none found)"
        exit 1
    fi
fi

# Dataset path
DATASET_PATH="${GROOT_DATASETS_DIR}/${DATASET_NAME}"
if [ ! -d "$DATASET_PATH" ]; then
    echo -e "${RED}Error: Dataset not found at ${DATASET_PATH}${NC}"
    echo ""
    echo "Available datasets:"
    ls -la "${GROOT_DATASETS_DIR}/" 2>/dev/null || echo "  (none found)"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SO-100 GR00T Open-Loop Evaluation${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Checkpoint: ${YELLOW}${SELECTED_CHECKPOINT}${NC}"
echo -e "  Dataset: ${YELLOW}${DATASET_PATH}${NC}"
echo -e "  Trajectories: ${YELLOW}${NUM_TRAJECTORIES}${NC}"
echo -e "  Steps per trajectory: ${YELLOW}${STEPS}${NC}"
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

# Check GPU
echo -e "${GREEN}Checking GPU availability...${NC}"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader
echo ""

# Display evaluation info
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}OPEN-LOOP EVALUATION INFORMATION${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""
echo "Open-loop evaluation compares model predictions against ground truth:"
echo "  • For each step, the model receives the observation from the dataset"
echo "  • The model predicts the next action(s)"
echo "  • MSE is computed between predicted and actual actions"
echo "  • Lower MSE indicates better action prediction accuracy"
echo ""
echo "With --plot flag, you'll see:"
echo "  • Predicted vs. actual action trajectories"
echo "  • Camera images from the dataset"
echo "  • Action error over time"
echo ""
read -p "Press Enter to start open-loop evaluation..."
echo ""

# Change to Isaac-GR00T directory
cd "$ISAAC_GROOT_DIR"

# Run open-loop evaluation
echo -e "${GREEN}Running open-loop evaluation...${NC}"
python scripts/eval_policy.py \
    --model-path "${SELECTED_CHECKPOINT}" \
    --dataset-path "${DATASET_PATH}" \
    --data-config so100_dualcam \
    --embodiment-tag new_embodiment \
    --modality-keys single_arm gripper \
    --video-backend torchvision_av \
    --trajs "${NUM_TRAJECTORIES}" \
    --steps "${STEPS}" \
    --plot

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Open-loop evaluation complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Interpretation:${NC}"
echo "  • Lower MSE = Better model performance"
echo "  • Typical good MSE: < 0.01 for position, < 0.1 for velocity"
echo "  • Check plots for systematic errors (drift, oscillation)"
echo ""

