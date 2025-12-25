#!/bin/bash
# SO-100 GR00T Dataset Preparation Script
# Phase 2: Prepare LeRobot dataset for GR00T training
#
# Usage: ./02_prepare_dataset.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATASET_NAME="seq1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Consolidated ChefMate directory structure
CHEFMATE_DIR="$HOME/chefmate"
DATASETS_DIR="${CHEFMATE_DIR}/datasets/lerobot"
GROOT_DATASETS_DIR="${CHEFMATE_DIR}/datasets/groot_format"
ISAAC_GROOT_DIR="$HOME/Isaac-GR00T"

# Source and destination paths
SOURCE_LEROBOT="${DATASETS_DIR}/rubbotix/${DATASET_NAME}"
DEST_GROOT="${GROOT_DATASETS_DIR}/${DATASET_NAME}"

# Ensure directories exist
mkdir -p "${GROOT_DATASETS_DIR}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SO-100 Dataset Preparation for GR00T${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Dataset Name: ${YELLOW}${DATASET_NAME}${NC}"
echo -e "  Source: ${YELLOW}${SOURCE_LEROBOT}${NC}"
echo -e "  Destination: ${YELLOW}${DEST_GROOT}${NC}"
echo ""

# Check if gr00t environment exists
if ! conda env list | grep -q "gr00t"; then
    echo -e "${RED}Error: gr00t conda environment not found${NC}"
    echo "Please create it first"
    exit 1
fi

# Activate gr00t environment
echo -e "${GREEN}Activating gr00t environment...${NC}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gr00t

# Check source dataset (now in ~/chefmate/datasets/lerobot/)
if [ ! -d "$SOURCE_LEROBOT" ]; then
    echo -e "${RED}Error: Source dataset not found at ${SOURCE_LEROBOT}${NC}"
    echo ""
    echo "Please run 01_record_dataset.sh first, or if your dataset is in the"
    echo "old location (~/.cache/huggingface/lerobot/), migrate it with:"
    echo "  cp -r ~/.cache/huggingface/lerobot/rubbotix/${DATASET_NAME} ${DATASETS_DIR}/rubbotix/"
    exit 1
fi

echo -e "${GREEN}Source dataset found${NC}"
echo ""

# Validate source dataset structure
echo -e "${GREEN}Validating source dataset structure...${NC}"
for dir in data meta videos; do
    if [ ! -d "$SOURCE_LEROBOT/$dir" ]; then
        echo -e "${RED}Error: Missing directory: $dir${NC}"
        exit 1
    fi
    echo -e "  ✓ $dir/"
done
echo ""

# Create destination directory
echo -e "${GREEN}Creating destination directory...${NC}"
mkdir -p "$DEST_GROOT"

# Copy dataset
echo -e "${GREEN}Copying dataset to GR00T format directory...${NC}"
echo "  From: ${SOURCE_LEROBOT}"
echo "  To: ${DEST_GROOT}"
echo ""

# Use rsync for efficient copying with progress
# Note: --no-perms --no-owner --no-group flags prevent harmless permission warnings
if command -v rsync &> /dev/null; then
    rsync -ah --info=progress2 --no-perms --no-owner --no-group "$SOURCE_LEROBOT/" "$DEST_GROOT/"
else
    echo -e "${YELLOW}rsync not found, using cp (slower)${NC}"
    cp -r "$SOURCE_LEROBOT/"* "$DEST_GROOT/"
fi

echo ""
echo -e "${GREEN}Dataset copied successfully${NC}"
echo ""

# Remove LeRobot stats.json - GR00T will regenerate with correct format (includes q01/q99)
if [ -f "$DEST_GROOT/meta/stats.json" ]; then
    echo -e "${YELLOW}Removing LeRobot stats.json (GR00T will regenerate with correct format)${NC}"
    rm "$DEST_GROOT/meta/stats.json"
fi

# Convert episodes to jsonl format
echo -e "${GREEN}Converting episodes to GR00T format...${NC}"
python "${SCRIPT_DIR}/convert_episodes_to_jsonl.py" "$DEST_GROOT"

if [ ! -f "$DEST_GROOT/meta/episodes.jsonl" ]; then
    echo -e "${RED}Failed to create episodes.jsonl${NC}"
    exit 1
fi
echo ""

# Convert dataset structure to GR00T format (per-episode files)
echo -e "${GREEN}Converting dataset structure to GR00T format...${NC}"
python "${SCRIPT_DIR}/convert_to_groot_format.py" "$DEST_GROOT"
echo ""

# Create custom modality.json with front and wrist cameras
echo -e "${GREEN}Creating custom modality.json...${NC}"
MODALITY_FILE="${DEST_GROOT}/meta/modality.json"

cat > "$MODALITY_FILE" << 'EOF'
{
    "state": {
        "single_arm": {
            "start": 0,
            "end": 5
        },
        "gripper": {
            "start": 5,
            "end": 6
        }
    },
    "action": {
        "single_arm": {
            "start": 0,
            "end": 5
        },
        "gripper": {
            "start": 5,
            "end": 6
        }
    },
    "video": {
        "front": {
            "original_key": "observation.images.front"
        },
        "wrist": {
            "original_key": "observation.images.wrist"
        }
    },
    "annotation": {
        "human.task_description": {
            "original_key": "task_index"
        }
    }
}
EOF

echo -e "${GREEN}modality.json created with front and wrist camera keys${NC}"
echo ""

# Validate modality file
echo -e "${GREEN}Validating modality.json...${NC}"
if python -c "import json; json.load(open('$MODALITY_FILE'))" 2>/dev/null; then
    echo -e "  ✓ Valid JSON format"
else
    echo -e "${RED}Error: Invalid JSON in modality.json${NC}"
    exit 1
fi
echo ""

# Convert videos to H.264 for torchcodec compatibility
echo -e "${GREEN}Converting videos to H.264 format...${NC}"
cd "$SCRIPT_DIR"
python convert_videos_to_h264.py "$DEST_GROOT"
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Video conversion failed${NC}"
    exit 1
fi
echo ""

# Validate dataset with GR00T
echo -e "${GREEN}Validating dataset with GR00T...${NC}"
cd "$ISAAC_GROOT_DIR"

echo "Testing dataset loading..."
python scripts/load_dataset.py \
    --dataset-path "${DEST_GROOT}" \
    2>&1 | tee /tmp/dataset_validation.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Dataset preparation completed!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Dataset location:"
    echo "  ${DEST_GROOT}"
    echo ""
    echo "Dataset structure:"
    tree -L 2 "$DEST_GROOT" 2>/dev/null || ls -R "$DEST_GROOT"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "  1. Review dataset: python scripts/load_dataset.py --dataset-path ${DEST_GROOT} --plot-state-action"
    echo "  2. Start training: ./03_train_model.sh"
    echo ""
else
    echo ""
    echo -e "${RED}Dataset validation failed${NC}"
    echo "Check the log at: /tmp/dataset_validation.log"
    exit 1
fi

