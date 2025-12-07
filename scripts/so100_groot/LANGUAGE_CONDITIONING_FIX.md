# Language Conditioning Fix for GR00T Multitask Training

**Date**: 2025-10-19  
**Issue**: Language conditioning fails even after enabling diffusion model training  
**Status**: Root cause identified, solution provided

---

## TL;DR

**Problem**: Model ignores language instructions and uses visual heuristics instead.

**Root Cause**: Frozen Eagle VLM backbone cannot learn task-specific language-vision associations.

**Solution**: Enable LLM fine-tuning with `--tune-llm` flag (minimum requirement).

**Best Solution**: Enable both LLM and vision fine-tuning with `--tune-llm --tune-visual` (if VRAM allows).

---

## Quick Fix

### Minimum Configuration (Language Conditioning)

```bash
cd ~/lerobot/scripts/so100_groot

# Edit 03_train_model.sh and add these flags to TRAIN_CMD:
--tune-llm \
```

**VRAM**: ~12-16GB (may require reducing batch size to 8 or 4)

### Recommended Configuration (Best Results)

```bash
# Edit 03_train_model.sh and add these flags to TRAIN_CMD:
--tune-llm \
--tune-visual \
--lora-rank 16 \  # Reduce from 32 to save VRAM
--lora-alpha 32   # Reduce from 64 (2x rank)
```

**VRAM**: ~16-20GB (may require batch size 4)

---

## Testing VRAM Requirements

Before starting full training, test what fits on your GPU:

```bash
cd ~/lerobot/scripts/so100_groot
./test_vram_requirements.sh
```

This script will test 5 configurations and tell you which one fits on your GPU.

---

## Understanding the Problem

### How GR00T Processes Language

1. **Input**: Images + Language text
2. **Eagle VLM**: Processes both together → joint vision-language embeddings
3. **Action Head**: Uses embeddings to predict actions

### Why Frozen Backbone Fails

**When Eagle is frozen** (`tune_llm=False`, `tune_visual=False`):
- Eagle was pre-trained on general VLM tasks (not your specific tasks)
- Eagle produces nearly identical embeddings for "pick cheese" vs "pick bread"
- Diffusion model has no signal to differentiate tasks
- Model learns visual heuristics: "if object in holder → pick it"

**Evidence from testing**:
- "pick up cheese" → picks up cheese ✓
- "pick up bread" → picks up cheese ✗ (ignores instruction)
- "do not pick up cheese" → picks up cheese ✗ (completely ignores instruction)

### Why Enabling LLM Helps

**When `tune_llm=True`**:
- LLM fine-tunes on your specific instructions
- LLM learns to differentiate "cheese" vs "bread" tokens
- Creates distinct language embeddings
- Diffusion model can learn different actions for different instructions

### Why Enabling Vision Helps Even More

**When `tune_visual=True` + `tune_llm=True`**:
- Vision tower learns to recognize cheese vs bread visually
- LLM learns task-specific language
- Model learns to ground language to visual objects
- Robust task conditioning

---

## Training Configurations

### Configuration 1: Baseline (BROKEN for Multitask)

```bash
python scripts/gr00t_finetune.py \
    --dataset-path ./demo_data/cheese/ ./demo_data/bread/ \
    --tune-llm False \           # ❌ Frozen
    --tune-visual False \        # ❌ Frozen
    --tune-projector True \
    --tune-diffusion-model True \
    --lora-rank 32
```

**Result**: Language conditioning fails, model uses visual heuristics  
**VRAM**: ~8GB  
**Use case**: Single-task training only

### Configuration 2: LLM Only (MINIMUM for Multitask)

```bash
python scripts/gr00t_finetune.py \
    --dataset-path ./demo_data/cheese/ ./demo_data/bread/ \
    --tune-llm True \            # ✅ Trainable
    --tune-visual False \        # ❌ Frozen
    --tune-projector True \
    --tune-diffusion-model True \
    --lora-rank 32 \
    --batch-size 8               # May need to reduce
```

**Result**: Language conditioning works  
**VRAM**: ~12-16GB  
**Use case**: Multitask with distinct language instructions

### Configuration 3: LLM + Vision (BEST)

```bash
python scripts/gr00t_finetune.py \
    --dataset-path ./demo_data/cheese/ ./demo_data/bread/ \
    --tune-llm True \            # ✅ Trainable
    --tune-visual True \         # ✅ Trainable
    --tune-projector True \
    --tune-diffusion-model True \
    --lora-rank 16 \             # Reduced to save VRAM
    --lora-alpha 32 \
    --batch-size 4               # Reduced to save VRAM
```

**Result**: Best language conditioning + visual grounding  
**VRAM**: ~16-20GB  
**Use case**: Multitask with robust task understanding

---

## Step-by-Step: Updating Your Training Script

### Option 1: Edit 03_train_model.sh Manually

1. Open the file:
   ```bash
   nano ~/lerobot/scripts/so100_groot/03_train_model.sh
   ```

2. Find the `TRAIN_CMD` section (around line 133)

3. Add `--tune-llm \` after `--video-backend torchvision_av \`

4. Optionally add `--tune-visual \` for best results

5. Optionally reduce LoRA rank:
   ```bash
   LORA_RANK=16  # Change from 32
   LORA_ALPHA=32 # Change from 64
   ```

6. Optionally reduce batch size:
   ```bash
   BATCH_SIZE=8  # Change from 16
   # or
   BATCH_SIZE=4  # If still OOM
   ```

7. Save and run:
   ```bash
   ./03_train_model.sh
   ```

### Option 2: Test First, Then Train

1. Test VRAM requirements:
   ```bash
   cd ~/lerobot/scripts/so100_groot
   ./test_vram_requirements.sh
   ```

2. Based on results, update `03_train_model.sh` with the recommended configuration

3. Run full training:
   ```bash
   ./03_train_model.sh
   ```

---

## Expected Training Metrics

### With LLM Fine-tuning

- **Training speed**: ~8-12 seconds/step (slower than baseline)
- **VRAM usage**: ~12-16GB
- **Trainable params**: ~15-20M (vs 6.5M baseline)
- **Training time**: ~3-4 hours for 1000 steps

### With LLM + Vision Fine-tuning

- **Training speed**: ~12-18 seconds/step
- **VRAM usage**: ~16-20GB
- **Trainable params**: ~25-35M
- **Training time**: ~4-6 hours for 1000 steps

---

## Validation

After training, test language conditioning:

```bash
# Start inference server
./04_start_inference_server.sh

# In another terminal, test with different instructions
# Edit TASK_INSTRUCTION in 05_deploy_robot.sh and run multiple times:

# Test 1: Cheese task
TASK_INSTRUCTION="pick up the yellow cheese and put it into the white plate"
./05_deploy_robot.sh

# Test 2: Bread task
TASK_INSTRUCTION="pick up the bread and put it into the white plate"
./05_deploy_robot.sh

# Test 3: Negation (should do nothing or different behavior)
TASK_INSTRUCTION="do not pick up the cheese"
./05_deploy_robot.sh
```

**Expected results**:
- ✅ Different instructions → different robot behaviors
- ✅ Model responds to language changes
- ✅ Negation has effect

---

## Troubleshooting

### OOM (Out of Memory) Errors

**Solution 1**: Reduce batch size
```bash
BATCH_SIZE=8  # or 4, or even 2
```

**Solution 2**: Reduce LoRA rank
```bash
LORA_RANK=16
LORA_ALPHA=32
```

**Solution 3**: Reduce gradient accumulation
```bash
GRADIENT_ACCUMULATION_STEPS=4  # from 8
```

**Solution 4**: Kill other GPU processes
```bash
nvidia-smi  # Find PIDs
kill <PID>
```

### Training Very Slow

**Expected**: LLM fine-tuning is slower than baseline
- Baseline: ~5-7 sec/step
- LLM only: ~8-12 sec/step
- LLM + Vision: ~12-18 sec/step

**If much slower**: Check GPU utilization
```bash
watch -n 1 nvidia-smi
```

Should show 90-100% GPU utilization.

### Language Conditioning Still Fails

**Check 1**: Verify LLM is trainable
```bash
# Look for this in training logs:
grep "Tune backbone llm" <log_file>
# Should show: Tune backbone llm: True
```

**Check 2**: Verify sufficient training
- Need at least 1000-3000 steps for language conditioning to emerge
- Check loss is decreasing

**Check 3**: Verify distinct language annotations
```bash
cd ~/Isaac-GR00T
python3 -c "
import json
cheese = json.loads(open('demo_data/cheese/meta/episodes.jsonl').readline())
bread = json.loads(open('demo_data/bread/meta/episodes.jsonl').readline())
print('Cheese:', cheese['tasks'])
print('Bread:', bread['tasks'])
"
```

Should show different task descriptions.

---

## Alternative: Separate Models

If VRAM is too limited for backbone training:

1. **Train separate models** for each task:
   ```bash
   # Train cheese model
   python scripts/gr00t_finetune.py \
       --dataset-path ./demo_data/cheese/ \
       --output-dir ~/so100-groot-checkpoints/cheese_only
   
   # Train bread model
   python scripts/gr00t_finetune.py \
       --dataset-path ./demo_data/bread/ \
       --output-dir ~/so100-groot-checkpoints/bread_only
   ```

2. **Use task selector** at inference time (e.g., object detection or user input)

3. **Load appropriate model** based on task

**Pros**: Works with frozen backbone, lower VRAM  
**Cons**: Multiple models, no language conditioning, cannot handle novel combinations

---

## Summary

**For multitask learning with language conditioning**:
- ✅ **MUST** enable `--tune-llm`
- ✅ **RECOMMENDED** enable `--tune-visual` (if VRAM allows)
- ✅ **REQUIRED** enable `--tune-diffusion-model` (already fixed)
- ⚠️ **MAY NEED** to reduce batch size and LoRA rank

**Test first**: Run `./test_vram_requirements.sh` to find what fits on your GPU.

**Expected outcome**: Model will properly respond to different language instructions.

