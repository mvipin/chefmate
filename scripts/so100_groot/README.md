# SO-100 GR00T N1.5 Automated Workflow Scripts

## Overview

This directory contains automated scripts for the complete SO-100 GR00T N1.5 workflow, customized for your specific setup and proven configuration.

**Task**: Multi-ingredient sandwich assembly
**Recording Mode**: Teleoperation (using leader arm)
**Training**: LoRA fine-tuning with GR00T N1.5
**Estimated Total Time**: ~2-3 hours for full pipeline

---

## 📁 Consolidated Directory Structure

All data is organized with datasets on high-speed NVMe storage:

```
~/chefmate/
├── datasets -> /mnt/nvme_data/chefmate_datasets/  # Symlink to NVMe
├── checkpoints/               # Trained model checkpoints
├── logs/
│   ├── training/              # Training logs
│   └── deployment/            # Deployment logs
└── scripts/so100_groot/       # This workflow scripts directory

/mnt/nvme_data/chefmate_datasets/   # NVMe storage (795GB, fast I/O)
├── lerobot/rubbotix/              # Raw LeRobot datasets (recording)
└── groot_format/                  # GR00T-formatted datasets (training)
```

**Benefits of NVMe Storage:**
- **Fast I/O**: NVMe SSD for fast dataset read/write during training
- **Large Capacity**: 795GB available for multiple datasets
- **Script Compatibility**: Symlink ensures all paths work unchanged

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
cd ~/chefmate/scripts/so100_groot

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

# ===== Additional Tools =====

# Visualize recorded episodes with Rerun.io
./visualize_episodes.sh [dataset_name] [episode_number]

# Open-loop evaluation (compare predictions vs ground truth)
./06_open_loop_eval.sh [checkpoint_name] [dataset_name]

# Replay episode on robot (pure action playback)
./07_replay_episode.sh [dataset_name] [episode_number]

# Migrate data from old locations
./migrate_existing_data.sh [--dry-run]
```

---

## Script Details

### 01_record_dataset.sh
**Purpose**: Record 20 demonstration episodes via teleoperation using leader arm

**What it does**:
- Activates lerobot environment
- Checks device permissions and mappings (follower + leader arms)
- Verifies calibration files for both arms
- Records dataset with front + wrist cameras at 640x480 (smooth recording)
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
- Creates custom `modality.json` with front/wrist camera keys
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
  - front: /dev/scene (640x480 @ 30fps)
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

## File Locations (Consolidated in ~/chefmate/)

### LeRobot Dataset Files
```
~/chefmate/datasets/lerobot/rubbotix/{dataset_name}/
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

### GR00T Format Dataset
```
~/chefmate/datasets/groot_format/{dataset_name}/
├── data/
├── meta/
│   ├── modality.json  ← Custom front/wrist config
│   └── episodes.jsonl
└── videos/
```

### Checkpoints
```
~/chefmate/checkpoints/{dataset_name}/
├── checkpoint-500/
├── checkpoint-1000/
├── ...
└── tensorboard/
```

### Logs
```
~/chefmate/logs/
├── training/
└── deployment/
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

### Training Metrics
- **Loss**: 1.0 → 0.15-0.20 over 1000 steps
- **VRAM Usage**: 12-14GB
- **Training Speed**: ~1.5-2 seconds per step
- **Total Time**: 20-30 minutes

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

## Advanced Options

### Enable Diffusion Model Tuning
Edit `03_train_model.sh`, remove this line:
```bash
--no-tune_diffusion_model \
```

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

## 🔍 Rerun.io Integration

[Rerun](https://rerun.io) is a visualization tool for multimodal data, integrated into LeRobot for dataset inspection and debugging.

### What is Rerun.io?

Rerun is an open-source toolkit for logging, visualizing, and debugging temporal multimodal data. In robotics, it's used to:

- **Visualize camera feeds** side-by-side with synchronized timestamps
- **Plot joint trajectories** and action values over time
- **Debug robot behavior** by scrubbing through episodes frame-by-frame
- **Analyze data quality** before training

### How LeRobot Uses Rerun

LeRobot's `lerobot-dataset-viz` command uses Rerun to render:
1. **Camera images** from `observation.images.front` and `observation.images.wrist`
2. **State trajectories** from `observation.state` (6 joint values)
3. **Action values** from `action` (6 joint commands)
4. **Episode metadata** (timestamps, frame indices)

### Using Rerun for ChefMate

```bash
# Activate lerobot environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate lerobot

# Basic usage - visualize episode 0 of test1
./visualize_episodes.sh test1 0

# Visualize specific dataset and episode
./visualize_episodes.sh cheese 10

# Direct command with custom options
lerobot-dataset-viz \
    --repo-id rubbotix/test1 \
    --root ~/chefmate/datasets/lerobot \
    --episode-index 0 \
    --mode local
```

### Rerun Viewer Controls

Once the viewer opens:
- **Timeline scrubbing**: Drag the timeline to navigate through frames
- **Pan/Zoom**: Mouse drag and scroll in camera views
- **Space bar**: Play/pause automatic playback
- **Entity panels**: Show/hide specific data streams
- **Log filtering**: Filter by entity path or type

### Remote Visualization (SSH)

For viewing datasets on a remote machine:

```bash
# On remote machine (server)
lerobot-dataset-viz \
    --repo-id rubbotix/test1 \
    --root ~/chefmate/datasets/lerobot \
    --episode-index 0 \
    --mode distant \
    --ws-port 9087

# On local machine (with SSH tunnel)
ssh -L 9087:localhost:9087 username@remote-host
rerun ws://localhost:9087
```

### Saving Visualizations

```bash
# Save .rrd file for later viewing
lerobot-dataset-viz \
    --repo-id rubbotix/test1 \
    --root ~/chefmate/datasets/lerobot \
    --episode-index 0 \
    --save 1 \
    --output-dir ~/chefmate/assets/visualizations

# View saved recording
rerun ~/chefmate/assets/visualizations/rubbotix_test1_episode_0.rrd
```

### Interpreting Rerun Visualizations

| Panel | What to Look For |
|-------|------------------|
| **Camera feeds** | Clear view of workspace, no occlusion, consistent lighting |
| **State trajectory** | Smooth joint movements, no sudden jumps or spikes |
| **Action values** | Actions match states with slight offset (command vs actual) |
| **Timeline** | Consistent frame rate (30 fps), no dropped frames |

### Debugging with Rerun

Common issues visible in Rerun:
1. **Camera dropout**: Black frames indicate camera disconnect
2. **Jittery motion**: Spiky state values suggest vibration or noise
3. **Action mismatch**: Large gaps between action and state suggest lag
4. **Synchronization**: Misaligned camera timestamps

---

## Support Resources

- **ChefMate Documentation**: `~/chefmate/README.md`
- **Seeed Studio Docs**: https://wiki.seeedstudio.com/lerobot_so100m_new
- **NVIDIA GR00T**: https://github.com/NVIDIA/Isaac-GR00T
- **LeRobot**: https://github.com/huggingface/lerobot
- **Rerun.io**: https://rerun.io/docs

---

## Version History

- **v1.0** (2025-10-04): Initial release
  - Customized for RTX 4080 Super (16GB)
  - Based on proven training configuration
  - Optimized for 1000 steps training
  - Scene/wrist camera naming
  - 20 episode dataset

---

## Notes

- These scripts are based on your **proven successful configuration**
- Camera indices (0 and 2) match your working deployment setup
- Training parameters are optimized for your RTX 4080 Super
- All scripts include error checking and helpful messages
- Logs are saved for debugging and analysis

---

**Ready to start?** Run `./01_record_dataset.sh` to begin!

