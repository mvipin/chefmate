#!/bin/bash
# SO-100 GR00T Model Training Script
# Phase 3: Fine-tune GR00T N1.5 model on recorded dataset
#
# Usage: ./03_train_model.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration for Multi-Task Training (Cheese + Bread)
# Using LeRobotMixtureDataset for automatic balancing
# Training from scratch on both datasets together
DATASET_NAMES=("cheese" "bread")  # Multiple datasets
OUTPUT_DIR="$HOME/so100-groot-checkpoints/cheese_bread_multitask"
BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=8
MAX_STEPS=10000  # Full training on combined dataset
SAVE_STEPS=500
LEARNING_RATE=0.0001  # Standard learning rate for training from scratch
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0.1
DATALOADER_NUM_WORKERS=8

# Resume from checkpoint (set to "true" to resume from latest checkpoint in output dir)
RESUME_TRAINING="false"  # Set to "false" to start fresh, "true" to resume from latest checkpoint

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SO-100 GR00T Multi-Task Training${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Datasets: ${YELLOW}${DATASET_NAMES[@]}${NC}"
echo -e "  Training: ${YELLOW}From scratch on combined dataset${NC}"
echo -e "  Output Dir: ${YELLOW}${OUTPUT_DIR}${NC}"
echo -e "  Batch Size: ${YELLOW}${BATCH_SIZE}${NC}"
echo -e "  Gradient Accumulation: ${YELLOW}${GRADIENT_ACCUMULATION_STEPS}${NC}"
echo -e "  Effective Batch Size: ${YELLOW}$((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))${NC}"
echo -e "  Max Steps: ${YELLOW}${MAX_STEPS}${NC}"
echo -e "  Save Every: ${YELLOW}${SAVE_STEPS} steps${NC}"
echo -e "  Learning Rate: ${YELLOW}${LEARNING_RATE}${NC}"
echo -e "  LoRA Rank: ${YELLOW}${LORA_RANK}${NC}"
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

# Check datasets exist
echo -e "${GREEN}Checking datasets...${NC}"
for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    DATASET_PATH="$HOME/Isaac-GR00T/demo_data/${DATASET_NAME}"
    if [ ! -d "$DATASET_PATH" ]; then
        echo -e "${RED}Error: Dataset not found at ${DATASET_PATH}${NC}"
        echo "Please run 02_prepare_dataset.sh first"
        exit 1
    fi
    echo -e "  ✓ ${DATASET_NAME} found"
done
echo ""

# Check GPU
echo -e "${GREEN}Checking GPU availability...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. Is CUDA installed?${NC}"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Create output directory
echo -e "${GREEN}Creating output directory...${NC}"
mkdir -p "$OUTPUT_DIR"

# Change to Isaac-GR00T directory
cd "$HOME/Isaac-GR00T"

# Display training info
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}MULTI-TASK TRAINING INFORMATION${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""
echo "Training Strategy: LeRobotMixtureDataset with automatic balancing"
echo "Training from scratch on combined cheese + bread dataset"
echo "Total episodes: 100 (50 cheese + 50 bread)"
echo ""
echo "Estimated training time: ~3-4 hours for 10000 steps"
echo "Expected VRAM usage: ~7-8GB (with LLM + Vision fine-tuning)"
echo "Training speed: ~2.8 iterations/second"
echo ""
echo -e "${GREEN}⚠️  LANGUAGE CONDITIONING ENABLED:${NC}"
echo "  - LLM fine-tuning: ✅ Enabled (learns task-specific language)"
echo "  - Vision fine-tuning: ✅ Enabled (learns visual object recognition)"
echo "  - This configuration enables proper multitask language conditioning"
echo ""
echo "Monitor GPU usage in another terminal with:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Training logs will be saved to:"
echo "  ${OUTPUT_DIR}/tensorboard/"
echo ""
echo "View logs with:"
echo "  tensorboard --logdir ${OUTPUT_DIR}/tensorboard/"
echo ""
read -p "Press Enter to start multi-task training..."
echo ""

# Start training with your proven configuration
echo -e "${GREEN}Starting model training...${NC}"
echo -e "${YELLOW}Training for ${MAX_STEPS} steps...${NC}"
echo ""

# Set environment variables for memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONWARNINGS="ignore::UserWarning"
export TF_CPP_MIN_LOG_LEVEL=2

# Build dataset paths for multi-task training
DATASET_PATHS=""
for DATASET_NAME in "${DATASET_NAMES[@]}"; do
    DATASET_PATHS="$DATASET_PATHS ./demo_data/${DATASET_NAME}/"
done

# Build training command for multi-task training
TRAIN_CMD="python scripts/gr00t_finetune.py \
    --dataset-path ${DATASET_PATHS} \
    --num-gpus 1 \
    --output-dir $OUTPUT_DIR \
    --max-steps ${MAX_STEPS} \
    --data-config so100_dualcam \
    --video-backend torchvision_av \
    --batch-size ${BATCH_SIZE} \
    --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
    --dataloader-num-workers ${DATALOADER_NUM_WORKERS} \
    --save-steps ${SAVE_STEPS} \
    --learning-rate ${LEARNING_RATE} \
    --report-to tensorboard \
    --lora-rank ${LORA_RANK} \
    --lora-alpha ${LORA_ALPHA} \
    --lora-dropout ${LORA_DROPOUT} \
    --tune-llm \
    --tune-visual \
    --balance-dataset-weights \
    --balance-trajectory-weights"

# Add resume flag if specified
if [ "$RESUME_TRAINING" = "true" ]; then
    # Check if any checkpoints exist
    if ls "$OUTPUT_DIR"/checkpoint-* 1> /dev/null 2>&1; then
        LATEST_CHECKPOINT=$(ls -td "$OUTPUT_DIR"/checkpoint-* | head -1)
        echo -e "${GREEN}Resuming from latest checkpoint: ${LATEST_CHECKPOINT}${NC}"
        TRAIN_CMD="$TRAIN_CMD --resume"
    else
        echo -e "${YELLOW}No checkpoints found. Starting training from scratch...${NC}"
    fi
else
    echo -e "${YELLOW}Starting training from scratch...${NC}"
fi

echo ""
echo -e "${BLUE}Training command:${NC}"
echo "$TRAIN_CMD"
echo ""

# Run training
eval $TRAIN_CMD

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    
    # Show checkpoint info
    echo "Checkpoints saved to:"
    echo "  ${OUTPUT_DIR}"
    echo ""
    
    if [ -d "$OUTPUT_DIR" ]; then
        echo "Available checkpoints:"
        ls -lh "$OUTPUT_DIR" | grep "checkpoint-" || echo "  (listing checkpoints...)"
        echo ""
        
        # Find latest checkpoint
        LATEST_CHECKPOINT=$(ls -td "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | head -1)
        if [ -n "$LATEST_CHECKPOINT" ]; then
            echo "Latest checkpoint:"
            echo "  ${LATEST_CHECKPOINT}"
        fi
    fi
    
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "  1. Review training logs: tensorboard --logdir ${OUTPUT_DIR}/tensorboard/"
    echo "  2. Start inference server: ./04_start_inference_server.sh"
    echo "  3. Deploy on robot: ./05_deploy_robot.sh"
    echo ""
else
    echo ""
    echo -e "${RED}Training failed${NC}"
    echo "Check the error messages above"
    echo ""
    echo "Common issues:"
    echo "  - GPU out of memory: Reduce batch_size in this script"
    echo "  - Dataset loading error: Check dataset preparation"
    echo "  - CUDA error: Check CUDA installation"
    exit 1
fi

