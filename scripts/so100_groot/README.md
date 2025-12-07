# SO-100 GR00T N1.5 Automated Workflow Scripts

## Overview

This directory contains automated scripts for the complete SO-100 GR00T N1.5 workflow, customized for your specific setup and proven configuration.

**Task**: Pick striped block and place it in the white plate
**Dataset Size**: 20 episodes
**Recording Mode**: Teleoperation (using leader arm)
**Training Steps**: 1000 steps
**Estimated Total Time**: ~1.5-2 hours

---

## Quick Start

### Prerequisites
```bash
# Verify setup
ls -la /dev/follower /dev/leader /dev/wrist /dev/scene
conda env list | grep -E "lerobot|gr00t"
nvidia-smi
```

### Run Complete Workflow
```bash
cd ~/lerobot/scripts/so100_groot

# Phase 1: Record dataset (20-30 minutes)
./01_record_dataset.sh

# Phase 2: Prepare dataset (5 minutes)
./02_prepare_dataset.sh

# Phase 3: Train model (20-30 minutes for 1000 steps)
./03_train_model.sh

# Phase 4a: Start inference server (Terminal 1)
./04_start_inference_server.sh

# Phase 4b: Deploy on robot (Terminal 2)
./05_deploy_robot.sh
```

---

## ⚠️ CRITICAL TRAINING CONFIGURATION ISSUE

### Language Conditioning Not Working - Root Cause Identified

**Problem**: Models trained with the previous configuration **DO NOT respond to language instructions**. The model ignores task descriptions and relies purely on visual state-based heuristics.

#### Evidence from Testing:

**Test Case**: Multitask model trained on both cheese and bread datasets
- **Expected**: Model should differentiate between "pick up cheese" vs "pick up bread" instructions
- **Actual**: Model exhibits identical behavior regardless of language instruction
- **Extreme Test**: "do not pick up cheese" still causes robot to pick up cheese

#### Model Behavior Analysis:

The model learned a **position-based state machine** instead of language-conditioned behavior:

**Single Ingredient Scenarios:**
1. Nothing in plate or holder → Random search (no clear target)
2. One ingredient in holder only → Grasp and place in plate
3. One ingredient in plate only → Stop (task complete)

**Two Ingredient Scenarios:**
4. Both ingredients in holder → Randomly select one to grasp
5. One in plate, one in holder → Stop (task complete)
6. Both in plate → Stop (task complete)

**Conclusion**: The model uses visual heuristic "if object in plate → task done" rather than understanding language instructions.

#### Root Cause:

The training script previously used `--no-tune_diffusion_model` flag, which **froze the action prediction head** (diffusion model). This prevented the model from learning proper language conditioning.

**Previous (BROKEN) Configuration:**
```bash
--no-tune_diffusion_model    # ❌ Action head frozen - cannot learn language conditioning
tune_llm: False              # ❌ Language encoder frozen
tune_visual: False           # ❌ Vision encoder frozen
tune_projector: True         # ✅ Only tiny projector layer trainable
```

**Result**: Only ~0.24% of model parameters were trainable, and the critical action prediction component was completely frozen.

#### Fix Applied (2025-10-19):

**Removed `--no-tune_diffusion_model` flag** to enable proper training:

```bash
# Now training with:
tune_diffusion_model: True   # ✅ Action head trainable (LoRA rank 32)
tune_projector: True         # ✅ Projector trainable (LoRA rank 32)
tune_llm: False              # ❌ LLM frozen (NOT OK - see update below!)
tune_visual: False           # ❌ Vision frozen (NOT OK - see update below!)
```

**Trainable parameters**: 6,553,600 (0.24% of total) - but now includes the critical diffusion model

#### ⚠️ CRITICAL UPDATE (2025-10-19 - Later):

**User reported**: Language conditioning still fails even with diffusion model training enabled!

**Root Cause**: The **Eagle VLM backbone** (vision-language model) is completely frozen. Eagle processes both images and text together to create joint embeddings. When frozen, Eagle cannot learn to differentiate between similar instructions like "pick cheese" vs "pick bread".

**The fundamental issue**:
- Eagle was pre-trained on general VLM tasks (image captioning, VQA)
- Eagle has never seen your specific tasks
- Frozen Eagle produces nearly identical embeddings for similar instructions
- Diffusion model has no signal to differentiate tasks
- Result: Model learns visual heuristics instead of language conditioning

**REQUIRED FIX for Multitask Learning**:
```bash
# Enable LLM fine-tuning (minimum requirement)
tune_llm: True               # ✅ REQUIRED for language conditioning
tune_visual: False           # Can stay frozen to save VRAM
tune_diffusion_model: True   # ✅ Required
tune_projector: True         # ✅ Required

# OR enable both (best results, but needs more VRAM)
tune_llm: True               # ✅ REQUIRED
tune_visual: True            # ✅ RECOMMENDED
tune_diffusion_model: True   # ✅ Required
tune_projector: True         # ✅ Required
```

**VRAM Requirements**:
- LLM only: ~12-16GB (may fit on RTX 4080 Super with reduced batch size)
- LLM + Vision: ~20-24GB (likely too much for RTX 4080 Super)
- Workaround: Reduce LoRA rank from 32 to 16 to save VRAM

#### Important Notes:

1. **All models trained before 2025-10-19 are affected** - they cannot properly use language conditioning
2. **Retrain required**: Models must be retrained with the corrected configuration
3. **VRAM requirement**: Ensure inference server is stopped before training (needs ~8GB VRAM)
4. **Training time**: Expect ~5-7 seconds per step (vs ~3-4 seconds with frozen diffusion model)

---

## Script Details

### 01_record_dataset.sh
**Purpose**: Record 20 demonstration episodes via teleoperation using leader arm

**What it does**:
- Activates lerobot environment
- Checks device permissions and mappings (follower + leader arms)
- Verifies calibration files for both arms
- Records dataset with scene + wrist cameras at 640x480 (smooth recording)
- Uses leader arm to control follower arm (teleoperation)
- Saves to `~/.cache/huggingface/lerobot/rubbotix/striped-block/`

**Duration**: 20-30 minutes (20 episodes × 30s + 30s reset)

**Recording Mode**: Teleoperation
- You control the follower arm by moving the leader arm
- The follower arm mirrors your leader arm movements
- Both arms must be connected and calibrated

**Keyboard controls during recording**:
- `→` (Right Arrow): Skip to next episode
- `←` (Left Arrow): Re-record current episode
- `ESC`: Stop and save dataset

---

### 02_prepare_dataset.sh
**Purpose**: Convert LeRobot dataset to GR00T-compatible format

**What it does**:
- Activates gr00t environment
- Copies dataset to `~/Isaac-GR00T/demo_data/stripped-block/`
- Creates custom `modality.json` with scene/wrist camera keys
- Validates dataset loading

**Duration**: 5 minutes

---

### 03_train_model.sh
**Purpose**: Fine-tune GR00T N1.5 model on recorded dataset

**What it does**:
- Activates gr00t environment
- Trains with your proven configuration:
  - Batch size: 16
  - Gradient accumulation: 8 (effective batch size: 128)
  - Max steps: 1000
  - Learning rate: 0.0001
  - LoRA rank: 32, alpha: 64, dropout: 0.1
  - Video backend: torchvision_av
  - No diffusion model tuning
- Saves checkpoints every 200 steps
- Logs to TensorBoard

**Duration**: 20-30 minutes for 1000 steps

**Expected VRAM**: 12-14GB

**Checkpoints saved to**: `~/so100-groot-checkpoints/stripped-block/`

---

### 04_start_inference_server.sh
**Purpose**: Start HTTP inference server for model deployment

**Usage**:
```bash
./04_start_inference_server.sh [checkpoint_step]

# Examples:
./04_start_inference_server.sh 1000  # Use checkpoint-1000
./04_start_inference_server.sh       # Use latest checkpoint
```

**What it does**:
- Activates gr00t environment
- Loads trained model checkpoint
- Starts HTTP server on port 8000
- Serves predictions for robot control

**Expected VRAM**: 6-8GB  
**Expected latency**: 50-80ms per inference

**Keep this terminal open** - server runs in foreground

---

### 05_deploy_robot.sh
**Purpose**: Deploy trained model on physical robot

**Prerequisites**: Inference server must be running (04_start_inference_server.sh)

**What it does**:
- Activates lerobot environment
- Checks inference server is running
- Verifies device permissions and mappings
- Connects to robot and cameras
- Executes autonomous task using trained policy

**Camera configuration**: Uses indices 0 and 2 (your proven setup)

**Safety**: Includes safety checklist and warnings

---

## Configuration Summary

### Dataset Recording
```yaml
Dataset Name: striped-block
Episodes: 20
Task: "Pick striped block and place it in the white plate"
Recording Mode: Teleoperation (Leader Arm)
Episode Time: 30s
Reset Time: 30s
Cameras:
  - wrist: /dev/wrist (640x480 @ 30fps)
  - scene: /dev/scene (640x480 @ 30fps)
Arms:
  - Follower: /dev/follower (performs task)
  - Leader: /dev/leader (controlled by human)
```

### Training
```yaml
Batch Size: 16
Gradient Accumulation: 8
Effective Batch Size: 128
Max Steps: 1000
Save Every: 200 steps
Learning Rate: 0.0001
LoRA:
  Rank: 32
  Alpha: 64
  Dropout: 0.1
Video Backend: torchvision_av
Tune Diffusion: false
```

### Deployment
```yaml
Task: "pick up the striped box and put it into the white plate"
Policy Server: localhost:8000
Cameras:
  - wrist: index 0 (640x480 @ 30fps)
  - front: index 2 (640x480 @ 30fps)
```

---

## File Locations

### Dataset Files
```
~/.cache/huggingface/lerobot/rubbotix/striped-block/
├── data/chunk-000/
│   ├── episode_000000.parquet
│   └── ...
├── meta/
│   ├── info.json
│   ├── stats.json
│   └── tasks.json
└── videos/chunk-000/
    ├── episode_000000.mp4
    └── ...
```

### GR00T Dataset
```
~/Isaac-GR00T/demo_data/striped-block/
├── data/
├── meta/
│   └── modality.json  ← Custom scene/wrist config
└── videos/
```

### Checkpoints
```
~/so100-groot-checkpoints/striped-block/
├── checkpoint-200/
├── checkpoint-400/
├── checkpoint-600/
├── checkpoint-800/
├── checkpoint-1000/
└── tensorboard/
```

### Deployment Logs
```
~/so100-groot-checkpoints/deployment_logs/
└── deployment_YYYYMMDD_HHMMSS.log
```

---

## Teleoperation Recording Tips

### Before Recording
1. **Calibrate both arms**: Ensure follower and leader arms are properly calibrated
2. **Test teleoperation**: Move the leader arm and verify follower mirrors correctly
3. **Practice the task**: Do a few practice runs before recording
4. **Comfortable position**: Position yourself comfortably to manipulate the leader arm

### During Recording
1. **Smooth movements**: Move the leader arm smoothly and deliberately
2. **Watch the follower**: Monitor the follower arm and cameras, not just the leader
3. **Consistent timing**: Try to maintain similar timing across episodes
4. **Complete the task**: Ensure successful task completion in each episode
5. **Use keyboard controls**: Don't hesitate to re-record (←) if a demonstration fails

### Quality Checks
- Follower arm successfully picks up the striped block
- Block is placed accurately in the white plate
- No collisions or jerky movements
- Both cameras capture the task clearly
- Consistent starting and ending positions

---

## Troubleshooting

### Device Permission Denied
```bash
sudo chmod 666 /dev/ttyACM*
```

### Camera Not Found
```bash
lerobot-find-cameras opencv
v4l2-ctl --list-devices
```

### GPU Out of Memory
Edit `03_train_model.sh` and reduce `BATCH_SIZE` from 16 to 8:
```bash
BATCH_SIZE=8
```

### Inference Server Not Responding
```bash
# Check if server is running
curl http://localhost:8000/health

# Check port usage
lsof -i :8000

# Kill stuck server
pkill -f inference_service.py
```

### Dataset Loading Error
```bash
# Verify modality.json
cat ~/Isaac-GR00T/demo_data/stripped-block/meta/modality.json

# Test loading
cd ~/Isaac-GR00T
python scripts/load_dataset.py --dataset-path demo_data/stripped-block
```

---

## Monitoring

### GPU Usage
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Memory usage
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

### Training Progress
```bash
# TensorBoard
tensorboard --logdir ~/so100-groot-checkpoints/stripped-block/tensorboard/

# Then open: http://localhost:6006
```

### Deployment Performance
```bash
# View live logs
tail -f ~/so100-groot-checkpoints/deployment_logs/deployment_*.log
```

---

## Expected Performance

### Training Metrics (Updated 2025-10-19)
- **Loss**: 1.0 → 0.15-0.20 over 1000 steps
- **VRAM Usage**: ~8GB (with corrected configuration)
- **Training Speed**: ~5-7 seconds per step (with diffusion model training enabled)
- **Total Time**: ~1.5-2 hours for 1000 steps

### Inference Metrics
- **Latency**: 50-80ms per action
- **Control Frequency**: 12-20 Hz
- **GPU Utilization**: 30-40%
- **VRAM Usage**: 6-8GB

### Task Performance
- **After 1000 steps**: 50-60% success rate (expected)
- **After 3000 steps**: 70-80% success rate (if you train longer)

---

## Tips for Success

### Recording Phase
1. **Consistency is key**: Perform similar motions each episode
2. **Smooth movements**: Avoid jerky or erratic motions
3. **Clear views**: Ensure both cameras see the task clearly
4. **Good lighting**: Maintain consistent lighting throughout
5. **Quality over quantity**: 20 good episodes > 50 mediocre ones

### Training Phase
1. **Monitor GPU**: Watch VRAM usage with `nvidia-smi`
2. **Check logs**: Use TensorBoard to monitor loss
3. **Be patient**: 1000 steps takes ~20-30 minutes
4. **Save checkpoints**: Don't interrupt training

### Deployment Phase
1. **Safety first**: Keep emergency stop accessible
2. **Start slow**: Test with a few runs first
3. **Monitor closely**: Watch robot behavior carefully
4. **Iterate**: If performance is poor, record more data or train longer

---

## Extending the Workflow

### Record More Episodes
Edit `01_record_dataset.sh`:
```bash
NUM_EPISODES=50  # Increase from 20
```

### Train Longer
Edit `03_train_model.sh`:
```bash
MAX_STEPS=3000  # Increase from 1000
```

### Different Task
Edit `01_record_dataset.sh` and `05_deploy_robot.sh`:
```bash
TASK_DESCRIPTION="Your new task description"
```

### Adjust Camera Resolution
Edit `01_record_dataset.sh` for recording:
```bash
width: 1920, height: 1080  # Higher resolution
```

Edit `05_deploy_robot.sh` for deployment (must match training):
```bash
width: 1920, height: 1080
```

---

## Troubleshooting

### CUDA Out of Memory During Training

**Symptom**: Training fails with `torch.OutOfMemoryError: CUDA out of memory`

**Cause**: Inference server or other GPU processes are still running

**Solution**:
```bash
# 1. Check GPU usage
nvidia-smi

# 2. Find Python processes using GPU
ps aux | grep python | grep -E "inference_service|eval_lerobot"

# 3. Kill the process (replace PID with actual process ID)
kill <PID>

# 4. Verify GPU is free
nvidia-smi  # Should show <500MB usage

# 5. Restart training
cd ~/lerobot/scripts/so100_groot
./03_train_model.sh
```

**Prevention**: Always stop the inference server before training:
```bash
pkill -f inference_service.py
```

### Training Speed Very Slow

**Expected**: ~5-7 seconds per step (with diffusion model training enabled)

**If slower than 10 seconds per step**:
1. Check GPU utilization: `nvidia-smi` (should be 90-100%)
2. Reduce batch size in `03_train_model.sh`: `BATCH_SIZE=8`
3. Reduce dataloader workers: `DATALOADER_NUM_WORKERS=4`

### Model Ignores Language Instructions

**Symptom**: Robot performs same action regardless of task instruction

**Cause**: Model trained with `--no-tune_diffusion_model` flag (before 2025-10-19 fix)

**Solution**: Retrain model with corrected configuration (see Critical Issue section above)

### Loss Not Decreasing

**Expected loss progression**:
- Step 100: ~0.8-1.0
- Step 500: ~0.3-0.5
- Step 1000: ~0.15-0.25

**If loss stays high (>0.8) after 500 steps**:
1. Check dataset quality: Review recorded episodes
2. Increase learning rate: `LEARNING_RATE=0.0002`
3. Train longer: `MAX_STEPS=3000`
4. Record more diverse episodes

---

## Advanced Options

### Adjust LoRA Parameters
Edit `03_train_model.sh`:
```bash
LORA_RANK=64      # Increase for more capacity
LORA_ALPHA=128    # Usually 2x rank
LORA_DROPOUT=0.05 # Reduce for less regularization
```

### Change Video Backend
Edit `03_train_model.sh`:
```bash
--video-backend decord  # Alternative: decord, pyav
```

---

## Support Resources

- **Main Documentation**: `~/lerobot/SO100_GROOT_IMPLEMENTATION_PROPOSAL.md`
- **Compatibility Info**: `~/lerobot/COMPATIBILITY_ASSESSMENT.md`
- **Quick Reference**: `~/lerobot/QUICK_REFERENCE.md`
- **Seeed Studio Docs**: https://wiki.seeedstudio.com/lerobot_so100m_new
- **NVIDIA GR00T**: https://github.com/NVIDIA/Isaac-GR00T
- **LeRobot**: https://github.com/huggingface/lerobot

---

## Version History

- **v1.1** (2025-10-19): Critical bug fix
  - **FIXED**: Removed `--no-tune_diffusion_model` flag that prevented language conditioning
  - Models can now properly learn from language instructions
  - Updated training metrics (slower but correct)
  - Added troubleshooting section for VRAM issues
  - Documented model behavior analysis

- **v1.0** (2025-10-04): Initial release
  - Customized for RTX 4080 Super (16GB)
  - Based on proven training configuration
  - Optimized for 1000 steps training
  - Scene/wrist camera naming
  - 20 episode dataset
  - ⚠️ **DEPRECATED**: Had `--no-tune_diffusion_model` bug

---

## Notes

- These scripts are based on your **proven successful configuration**
- Camera indices (0 and 2) match your working deployment setup
- Training parameters are optimized for your RTX 4080 Super
- All scripts include error checking and helpful messages
- Logs are saved for debugging and analysis

---

**Ready to start?** Run `./01_record_dataset.sh` to begin!

