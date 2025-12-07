# Multi-Task Training Setup: Cheese + Bread

## Overview
This document describes the multi-task training configuration for training GR00T N1.5 on both cheese and bread pick-and-place tasks simultaneously.

## Training Strategy: Option A - LeRobotMixtureDataset

### Why This Approach?
- **No physical merging needed** - Datasets remain separate
- **Native support** - GR00T's training script natively supports multiple datasets
- **Automatic balancing** - Provides weighted sampling across datasets
- **Prevents catastrophic forgetting** - Lower learning rate preserves cheese task performance

## Configuration Details

### Datasets
- **cheese**: 50 episodes (14,212 frames) - "Pick slice of cheese and place it in the white plate"
- **bread**: 50 episodes (13,483 frames) - "Pick slice of bread and place it in the white plate"
- **Total**: 100 episodes, ~27,695 frames

### Training Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Base Model** | `checkpoint-6000` (cheese) | Starting from trained cheese model |
| **Learning Rate** | `0.00005` | **50% reduction** to prevent catastrophic forgetting |
| **Max Steps** | `8000` | Reduced from 10000 (continuing from checkpoint) |
| **Batch Size** | `16` | Same as original training |
| **Gradient Accumulation** | `8` | Effective batch size = 128 |
| **LoRA Rank** | `32` | Same as original (memory constraint) |
| **LoRA Alpha** | `64` | Same as original |
| **Save Steps** | `500` | Checkpoint every 500 steps |

### Key Features
- `--balance-dataset-weights true` - Equal sampling from both datasets
- `--balance-trajectory-weights true` - Balanced trajectory sampling
- `--base-model-path` - Continue from cheese checkpoint-6000

## File Locations

### Datasets
```
~/Isaac-GR00T/demo_data/
├── cheese/          # Original cheese dataset
│   ├── data/
│   ├── meta/
│   └── videos/
└── bread/           # New bread dataset
    ├── data/
    ├── meta/
    └── videos/
```

### Checkpoints
```
~/so100-groot-checkpoints/
├── cheese/                      # Original single-task training
│   ├── checkpoint-6000/         # Base model for multi-task
│   └── checkpoint-10000/
└── cheese_bread_multitask/      # New multi-task training output
    ├── checkpoint-500/
    ├── checkpoint-1000/
    └── ...
```

## Running Multi-Task Training

### Prerequisites
1. ✅ Cheese dataset prepared and trained (checkpoint-6000 exists)
2. ✅ Bread dataset prepared and converted to GR00T format
3. ✅ Both datasets uploaded to Hugging Face Hub
   - https://huggingface.co/datasets/rubbotix/cheese
   - https://huggingface.co/datasets/rubbotix/bread

### Start Training
```bash
cd ~/lerobot
./scripts/so100_groot/03_train_model.sh
```

### Monitor Training
```bash
# Terminal 1: Watch GPU usage
watch -n 1 nvidia-smi

# Terminal 2: View TensorBoard
tensorboard --logdir ~/so100-groot-checkpoints/cheese_bread_multitask/tensorboard/
```

## Expected Results

### Training Metrics
- **Initial Loss**: ~0.025 (starting from trained checkpoint)
- **Target Loss**: ~0.015-0.020 (improved multi-task performance)
- **Training Time**: ~2.5-3.5 hours for 8000 steps
- **VRAM Usage**: ~12-14GB

### What to Monitor
1. **Loss convergence** - Should decrease gradually without spikes
2. **Both tasks** - Model should maintain cheese performance while learning bread
3. **No catastrophic forgetting** - Cheese task accuracy should remain high

## Deployment After Training

### Update Deployment Script
After training completes, update the deployment script to use the new checkpoint:

```bash
# In scripts/so100_groot/05b_deploy_robot_smooth.sh
CHECKPOINT_PATH="$HOME/so100-groot-checkpoints/cheese_bread_multitask/checkpoint-8000"
```

### Test Both Tasks
1. Test cheese pick-and-place
2. Test bread pick-and-place
3. Verify smooth motion for both tasks

## Troubleshooting

### Issue: Catastrophic Forgetting
**Symptoms**: Cheese task performance degrades during training
**Solution**: 
- Reduce learning rate further (try 0.000025)
- Increase dataset balancing weight for cheese

### Issue: Slow Convergence
**Symptoms**: Loss not decreasing after 2000 steps
**Solution**:
- Increase learning rate slightly (try 0.000075)
- Train for more steps (10000 instead of 8000)

### Issue: Out of Memory
**Symptoms**: CUDA out of memory error
**Solution**:
- Reduce batch size to 12
- Reduce dataloader workers to 4

## Next Steps After Training

1. **Evaluate Performance**
   - Test on physical robot with both tasks
   - Measure success rate for each task
   - Compare with single-task models

2. **Add More Tasks**
   - Record new datasets (lettuce, tomato, etc.)
   - Continue multi-task training from latest checkpoint
   - Build complete sandwich assembly pipeline

3. **Optimize Deployment**
   - Fine-tune action smoothing parameters
   - Adjust action horizon if needed
   - Test different temporal ensembling strategies

## Training Command Reference

Full command executed by the script:
```bash
python scripts/gr00t_finetune.py \
    --dataset-path ./demo_data/cheese/ ./demo_data/bread/ \
    --num-gpus 1 \
    --output-dir ~/so100-groot-checkpoints/cheese_bread_multitask \
    --max-steps 8000 \
    --data-config so100_dualcam \
    --video-backend torchvision_av \
    --batch-size 16 \
    --gradient-accumulation-steps 8 \
    --dataloader-num-workers 8 \
    --save-steps 500 \
    --learning-rate 0.00005 \
    --lora-rank 32 \
    --lora-alpha 64 \
    --lora-dropout 0.1 \
    --no-tune_diffusion_model \
    --balance-dataset-weights true \
    --balance-trajectory-weights true \
    --base-model-path ~/so100-groot-checkpoints/cheese/checkpoint-6000 \
    --report-to tensorboard
```

## References
- Original cheese training: `~/so100-groot-checkpoints/cheese/`
- Bread dataset: https://huggingface.co/datasets/rubbotix/bread
- Cheese dataset: https://huggingface.co/datasets/rubbotix/cheese

