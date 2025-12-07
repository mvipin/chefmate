#!/bin/bash

# Test VRAM requirements for different training configurations
# This script helps determine which configuration fits on your GPU

set -e

echo "=========================================="
echo "GR00T Training VRAM Requirements Test"
echo "=========================================="
echo ""

# Configuration
DATASET_PATH="$HOME/Isaac-GR00T/demo_data/cheese/"
OUTPUT_DIR="/tmp/gr00t_vram_test"
MAX_STEPS=10  # Just test initialization and a few steps
BATCH_SIZE=4  # Small batch for testing

# Activate environment
echo "Activating gr00t environment..."
eval "$(conda shell.bash hook)"
conda activate gr00t
cd ~/Isaac-GR00T

# Function to test configuration
test_config() {
    local config_name=$1
    local tune_llm=$2
    local tune_visual=$3
    local lora_rank=$4
    
    echo ""
    echo "=========================================="
    echo "Testing: $config_name"
    echo "  tune_llm: $tune_llm"
    echo "  tune_visual: $tune_visual"
    echo "  lora_rank: $lora_rank"
    echo "=========================================="
    
    # Clear output directory
    rm -rf "$OUTPUT_DIR"
    
    # Build command
    CMD="python scripts/gr00t_finetune.py \
        --dataset-path $DATASET_PATH \
        --num-gpus 1 \
        --output-dir $OUTPUT_DIR \
        --max-steps $MAX_STEPS \
        --data-config so100_dualcam \
        --video-backend torchvision_av \
        --batch-size $BATCH_SIZE \
        --gradient-accumulation-steps 1 \
        --dataloader-num-workers 4 \
        --save-steps 1000 \
        --learning-rate 0.0001 \
        --report-to tensorboard \
        --lora-rank $lora_rank \
        --lora-alpha $((lora_rank * 2)) \
        --lora-dropout 0.1"
    
    # Add tune flags
    if [ "$tune_llm" = "True" ]; then
        CMD="$CMD --tune-llm"
    fi
    
    if [ "$tune_visual" = "True" ]; then
        CMD="$CMD --tune-visual"
    fi
    
    echo ""
    echo "Command: $CMD"
    echo ""
    
    # Run and capture result
    if eval "$CMD" 2>&1 | tee /tmp/gr00t_vram_test.log; then
        # Extract VRAM usage from logs
        VRAM=$(grep "GPU memory before training" /tmp/gr00t_vram_test.log | tail -1 | awk '{print $NF}')
        echo ""
        echo "✅ SUCCESS: $config_name"
        echo "   VRAM Usage: $VRAM"
        echo ""
        
        # Also check nvidia-smi
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
        
        return 0
    else
        echo ""
        echo "❌ FAILED: $config_name"
        echo "   Likely OOM (Out of Memory)"
        echo ""
        
        # Check if it was OOM
        if grep -q "CUDA out of memory" /tmp/gr00t_vram_test.log; then
            echo "   Confirmed: CUDA Out of Memory"
        fi
        
        return 1
    fi
}

# Test configurations in order of increasing VRAM usage

echo ""
echo "Starting VRAM requirement tests..."
echo "This will test 5 configurations to find what fits on your GPU"
echo ""
read -p "Press Enter to continue..."

# Config 1: Baseline (frozen backbone, diffusion model trainable)
test_config "Baseline (Frozen Backbone)" "False" "False" 32
BASELINE_OK=$?

# Config 2: LLM only, LoRA rank 16
test_config "LLM Only (LoRA 16)" "True" "False" 16
LLM_16_OK=$?

# Config 3: LLM only, LoRA rank 32
test_config "LLM Only (LoRA 32)" "True" "False" 32
LLM_32_OK=$?

# Config 4: LLM + Vision, LoRA rank 16
test_config "LLM + Vision (LoRA 16)" "True" "True" 16
BOTH_16_OK=$?

# Config 5: LLM + Vision, LoRA rank 32
test_config "LLM + Vision (LoRA 32)" "True" "True" 32
BOTH_32_OK=$?

# Summary
echo ""
echo "=========================================="
echo "VRAM Requirements Test Summary"
echo "=========================================="
echo ""

if [ $BASELINE_OK -eq 0 ]; then
    echo "✅ Baseline (Frozen Backbone, LoRA 32): FITS"
else
    echo "❌ Baseline (Frozen Backbone, LoRA 32): OOM"
fi

if [ $LLM_16_OK -eq 0 ]; then
    echo "✅ LLM Only (LoRA 16): FITS"
else
    echo "❌ LLM Only (LoRA 16): OOM"
fi

if [ $LLM_32_OK -eq 0 ]; then
    echo "✅ LLM Only (LoRA 32): FITS"
else
    echo "❌ LLM Only (LoRA 32): OOM"
fi

if [ $BOTH_16_OK -eq 0 ]; then
    echo "✅ LLM + Vision (LoRA 16): FITS"
else
    echo "❌ LLM + Vision (LoRA 16): OOM"
fi

if [ $BOTH_32_OK -eq 0 ]; then
    echo "✅ LLM + Vision (LoRA 32): FITS"
else
    echo "❌ LLM + Vision (LoRA 32): OOM"
fi

echo ""
echo "=========================================="
echo "Recommendations"
echo "=========================================="
echo ""

if [ $BOTH_32_OK -eq 0 ]; then
    echo "🎉 Your GPU can handle the BEST configuration:"
    echo "   --tune-llm --tune-visual --lora-rank 32"
    echo ""
    echo "This will give you the best language conditioning performance."
elif [ $BOTH_16_OK -eq 0 ]; then
    echo "👍 Your GPU can handle LLM + Vision with reduced LoRA:"
    echo "   --tune-llm --tune-visual --lora-rank 16"
    echo ""
    echo "This is a good compromise between performance and VRAM."
elif [ $LLM_32_OK -eq 0 ]; then
    echo "⚠️  Your GPU can handle LLM only:"
    echo "   --tune-llm --lora-rank 32"
    echo ""
    echo "This will enable language conditioning but not visual grounding."
    echo "Should work for multitask learning with distinct language instructions."
elif [ $LLM_16_OK -eq 0 ]; then
    echo "⚠️  Your GPU can handle LLM only with reduced LoRA:"
    echo "   --tune-llm --lora-rank 16"
    echo ""
    echo "This is the minimum configuration for language conditioning."
else
    echo "❌ Your GPU cannot handle any configuration with backbone training."
    echo ""
    echo "Options:"
    echo "1. Reduce batch size further (try --batch-size 2 or 1)"
    echo "2. Use gradient checkpointing (if available)"
    echo "3. Train separate models for each task"
    echo "4. Upgrade GPU or use cloud training"
fi

echo ""
echo "Test complete!"
echo ""

# Cleanup
rm -rf "$OUTPUT_DIR"
rm -f /tmp/gr00t_vram_test.log

