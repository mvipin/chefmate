# Dataset Preparation Complete ✅

## Summary

Successfully prepared the **cheese** dataset (50 episodes, 14,212 frames) for GR00T training.

---

## What Was Done

### 1. Dataset Conversion
- ✅ Copied LeRobot dataset from `~/.cache/huggingface/lerobot/rubbotix/cheese`
- ✅ Converted to GR00T format at `~/Isaac-GR00T/demo_data/cheese`
- ✅ Split combined parquet files into per-episode files (50 files)
- ✅ Split combined video files into per-episode videos (50 videos × 2 cameras = 100 videos)
- ✅ Created `episodes.jsonl` with episode metadata
- ✅ Created `tasks.jsonl` with task descriptions
- ✅ Created `modality.json` with camera and action mappings
- ✅ Updated `info.json` with correct paths

### 2. Video Format Conversion
- ✅ Converted all videos from **AV1** to **H.264** codec
  - Reason: `torchcodec` (used by GR00T) doesn't support AV1 well
  - Tool: `scripts/so100_groot/convert_videos_to_h264.py`
  - Result: All 100 videos successfully converted

### 3. Dataset Validation
- ✅ Dataset loads successfully in GR00T
- ✅ Video frames decode correctly
- ✅ State and action data accessible
- ✅ Task descriptions properly formatted

---

## Dataset Structure

```
~/Isaac-GR00T/demo_data/cheese/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ... (50 episode files)
├── meta/
│   ├── episodes.jsonl          # Episode metadata (50 episodes)
│   ├── tasks.jsonl             # Task descriptions (1 task)
│   ├── modality.json           # Camera and action mappings
│   ├── info.json               # Dataset configuration
│   └── stats.safetensors       # Dataset statistics
└── videos/
    ├── observation.images.wrist/
    │   └── chunk-000/
    │       ├── episode_000000.mp4
    │       └── ... (50 videos, H.264 codec)
    └── observation.images.scene/
        └── chunk-000/
            ├── episode_000000.mp4
            └── ... (50 videos, H.264 codec)
```

---

## Dataset Details

- **Total Episodes**: 50
- **Total Frames**: 14,212
- **Task**: "Pick slice of yellow cheese and place it in the white plate"
- **Cameras**: 2 (wrist + scene)
- **Video Format**: H.264, 640×480, 30 fps
- **Action Space**: 6D (5 joint positions + 1 gripper)
- **State Space**: 6D (5 joint positions + 1 gripper position)

---

## Modality Mapping

The dataset uses the following modality mappings (defined in `modality.json`):

### State
- `state.single_arm`: Joint positions [0:5]
- `state.gripper`: Gripper position [5:6]

### Action
- `action.single_arm`: Joint actions [0:5]
- `action.gripper`: Gripper action [5:6]

### Video
- `video.scene`: Scene camera (original key: `observation.images.scene`)
- `video.wrist`: Wrist camera (original key: `observation.images.wrist`)

### Annotation
- `annotation.human.task_description`: Task description (from `task_index`)

---

## Validation Results

```
✓ Dataset initialized with EmbodimentTag.GR1
✓ Video frames: (1, 480, 640, 3) uint8
✓ State: (1, 5) float64 + (1, 1) float64
✓ Action: (1, 5) float64 + (1, 1) float64
✓ Task description: "Pick slice of yellow cheese and place it in the white plate"
```

---

## Next Steps

### 1. Review Dataset
```bash
cd ~/Isaac-GR00T
conda activate gr00t
python scripts/load_dataset.py --dataset-path demo_data/cheese --plot-state-action
```

### 2. Start Training
```bash
cd ~/lerobot
./scripts/so100_groot/03_train_model.sh
```

Or manually:
```bash
cd ~/Isaac-GR00T
conda activate gr00t
python scripts/gr00t_finetune.py \
    --dataset-path demo_data/cheese \
    --num-gpus 1 \
    --output-dir outputs/cheese_finetune
```

---

## Tools Created

1. **`convert_episodes_to_jsonl.py`**: Converts episode metadata to JSONL format
2. **`convert_to_groot_format.py`**: Converts LeRobot dataset structure to GR00T format
3. **`convert_videos_to_h264.py`**: Converts AV1 videos to H.264 for torchcodec compatibility

---

## Troubleshooting

### Issue: "No valid stream found in input file"
**Solution**: Videos need to be in H.264 format. Run:
```bash
python scripts/so100_groot/convert_videos_to_h264.py ~/Isaac-GR00T/demo_data/cheese
```

### Issue: "Failed to load dataset statistics"
**Solution**: This is normal on first load. GR00T will calculate and cache statistics automatically.

### Issue: Dataset not loading
**Solution**: Check that all required files exist:
- `meta/episodes.jsonl`
- `meta/tasks.jsonl`
- `meta/modality.json`
- `meta/info.json`
- `data/chunk-000/episode_*.parquet`
- `videos/*/chunk-000/episode_*.mp4`

---

## Environment

- **GR00T Environment**: `/mnt/nvme_data/conda/envs/gr00t`
- **Python**: 3.10.18
- **PyTorch**: 2.5.1+cu124
- **Flash Attention**: 2.8.3
- **GPU**: NVIDIA GeForce RTX 4080 SUPER (16GB VRAM)

---

## Files Modified

1. `scripts/so100_groot/02_prepare_dataset.sh` - Added video conversion step
2. Created `scripts/so100_groot/convert_videos_to_h264.py` - Video conversion tool

---

**Status**: ✅ Ready for Training

**Date**: 2025-10-18

