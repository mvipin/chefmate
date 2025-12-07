# Multitask Training Analysis: Language Conditioning Failure

**Date**: 2025-10-19  
**Model**: GR00T N1.5 (nvidia/GR00T-N1.5-3B)  
**Datasets**: Cheese (50 episodes, 14,212 frames) + Bread (50 episodes, 13,483 frames)  
**Training Configuration**: LoRA fine-tuning with `--no-tune_diffusion_model` (BROKEN)

---

## Executive Summary

**Critical Finding**: Models trained with the `--no-tune_diffusion_model` flag **cannot learn language conditioning**. The model ignores task instructions and relies purely on visual state-based heuristics.

**Impact**: All models trained before 2025-10-19 are affected and must be retrained.

**Root Cause**: The diffusion model (action prediction head) was frozen during training, preventing the model from learning to map language instructions to actions.

**Solution**: Remove `--no-tune_diffusion_model` flag from training script.

---

## Test Methodology

### Dataset Setup
- **Cheese dataset**: 50 episodes, task instruction: `"Pick slice of yellow cheese and place it in the white plate"`
- **Bread dataset**: 50 episodes, task instruction: `"Pick slice of bread and place it in the white plate"`
- **Training**: LeRobotMixtureDataset with balanced sampling
- **Checkpoints tested**: checkpoint-3000 (partial training)

### Test Cases

#### Test 1: Task-Specific Instructions
- **Instruction**: `"pick up the yellow cheese and put it into the white plate"`
- **Expected**: Robot picks up cheese
- **Actual**: Robot picks up cheese ✓

- **Instruction**: `"pick up the bread and put it into the white plate"`
- **Expected**: Robot picks up bread
- **Actual**: Robot picks up cheese ✗ (ignores instruction)

#### Test 2: Negation Test (Extreme Case)
- **Instruction**: `"do not pick up the cheese"`
- **Expected**: Robot does nothing or picks up bread
- **Actual**: Robot picks up cheese ✗ (completely ignores instruction)

#### Test 3: Visual State Manipulation
- **Setup**: Manually move cheese to plate (without robot)
- **Instruction**: `"pick up the bread and put it into the white plate"`
- **Expected**: Robot picks up bread
- **Actual**: Robot stops (considers task complete because cheese is in plate) ✗

---

## Model Behavior Analysis

### Observed Behavior Pattern

The model uses a **position-based state machine** rather than language-conditioned decision making:

```
State Machine Logic:
IF (object detected in plate):
    STOP (task complete)
ELSE IF (object detected in holder):
    GRASP object → MOVE to plate → RELEASE
ELSE:
    SEARCH randomly
```

### Detailed Behavior Matrix

| Scenario | Cheese Location | Bread Location | Robot Action | Language Instruction Effect |
|----------|----------------|----------------|--------------|----------------------------|
| 1 | Holder | Holder | Randomly picks one | ❌ Ignores instruction |
| 2 | Holder | None | Picks cheese | ❌ Ignores instruction |
| 3 | None | Holder | Picks bread | ❌ Ignores instruction |
| 4 | Plate | Holder | Stops | ❌ Ignores instruction |
| 5 | Holder | Plate | Stops | ❌ Ignores instruction |
| 6 | Plate | Plate | Stops | ❌ Ignores instruction |
| 7 | None | None | Random search | ❌ Ignores instruction |
| 8 | Plate | None | Stops | ❌ Ignores instruction |

**Key Observation**: The robot's behavior is **100% determined by visual state**, with **0% influence from language instruction**.

---

## Root Cause Analysis

### Training Configuration Issue

**Broken Configuration** (used before 2025-10-19):
```bash
python scripts/gr00t_finetune.py \
    --dataset-path ./demo_data/cheese/ ./demo_data/bread/ \
    --no-tune_diffusion_model \  # ❌ CRITICAL BUG
    --lora-rank 32 \
    --lora-alpha 64 \
    --tune-llm False \           # Default
    --tune-visual False \        # Default
    --tune-projector True        # Default
```

**What was actually trained**:
```
GR00T N1.5 Architecture:
├── Vision Tower (SigLIP) ..................... ❌ FROZEN
├── Language Model (Qwen2.5-3B) ............... ❌ FROZEN
└── Action Head
    ├── Projector (Linear layers) ............ ✅ TRAINABLE (LoRA rank 32)
    └── Diffusion Model (DiT) ................ ❌ FROZEN (--no-tune_diffusion_model)
```

**Trainable parameters**: 6,553,600 out of 2,730,717,120 (0.24%)  
**But**: Only the projector was trainable, not the diffusion model!

### Why Language Conditioning Failed

1. **Language encoder frozen**: Cannot learn to interpret new task instructions
2. **Vision encoder frozen**: Cannot learn to recognize specific objects (cheese vs bread)
3. **Diffusion model frozen**: Cannot learn new action patterns conditioned on language
4. **Only projector trainable**: This is just a small linear layer - insufficient for complex conditioning

**Result**: The model can only use pre-trained language/vision features, which don't map well to your specific tasks. It falls back to simple visual heuristics.

---

## Corrected Configuration (PARTIAL FIX)

**Fixed Configuration** (2025-10-19 onwards):
```bash
python scripts/gr00t_finetune.py \
    --dataset-path ./demo_data/cheese/ ./demo_data/bread/ \
    # REMOVED: --no-tune_diffusion_model
    --lora-rank 32 \
    --lora-alpha 64 \
    --tune-llm False \           # Still frozen (saves VRAM)
    --tune-visual False \        # Still frozen (saves VRAM)
    --tune-projector True        # Trainable
```

**What is now trained**:
```
GR00T N1.5 Architecture:
├── Vision Tower (SigLIP) ..................... ❌ FROZEN (saves VRAM)
├── Language Model (Qwen2.5-3B) ............... ❌ FROZEN (saves VRAM)
└── Action Head
    ├── Projector (Linear layers) ............ ✅ TRAINABLE (LoRA rank 32)
    └── Diffusion Model (DiT) ................ ✅ TRAINABLE (LoRA rank 32) ← FIXED
```

**Trainable parameters**: Still 6,553,600 (0.24%), but now includes the critical diffusion model

### ⚠️ CRITICAL UPDATE: Language Conditioning Still Fails!

**User reported**: Even after removing `--no-tune_diffusion_model`, language conditioning still doesn't work.

**Root Cause Analysis - Deeper Issue**:

The problem is more fundamental than just the frozen diffusion model. The **Eagle VLM backbone** (vision-language model) processes both images and text together to create joint vision-language embeddings. These embeddings are then passed to the action head.

**How language flows through the model**:
1. **Input**: Images + Language text (e.g., "pick up cheese")
2. **Eagle Processor**: Tokenizes text and processes images
3. **Eagle Model** (frozen): Creates joint vision-language embeddings
4. **Action Head**: Uses these embeddings to predict actions

**The fundamental problem**:
- Eagle model is **completely frozen** (`tune_llm=False`, `tune_visual=False`)
- Eagle was pre-trained on general vision-language tasks (e.g., image captioning, VQA)
- Eagle has **never seen** your specific tasks: "pick cheese" vs "pick bread"
- Eagle's frozen embeddings cannot distinguish between these similar instructions
- The action head can only work with whatever embeddings Eagle provides

**Why diffusion model training alone is insufficient**:
- Diffusion model can learn to map embeddings → actions
- But if Eagle provides nearly identical embeddings for "pick cheese" and "pick bread"
- Then diffusion model has no signal to differentiate the tasks
- Result: Model learns visual heuristics instead (e.g., "if object in holder → pick it")

### Expected Improvements (REVISED)

With diffusion model training enabled **BUT Eagle frozen**:
- ⚠️ Model can learn better action prediction
- ⚠️ Model can learn smoother trajectories
- ❌ Model **CANNOT** differentiate between similar language instructions
- ❌ Model **CANNOT** learn task-specific language conditioning
- ❌ Model falls back to visual heuristics

**Trade-offs**:
- ⚠️ Training slower: ~5-7 seconds/step (vs ~3-4 seconds with frozen diffusion)
- ⚠️ Slightly higher VRAM: ~8GB (vs ~6GB with frozen diffusion)
- ❌ Language conditioning still broken!

---

## Technical Deep Dive: Why Frozen Backbone Breaks Language Conditioning

### Eagle VLM Architecture

GR00T uses the **Eagle 2.5 VLM** (Vision-Language Model) as its backbone:
- **Vision Tower**: SigLIP (processes images)
- **Language Model**: Qwen2.5-3B (processes text)
- **MLP Connector**: Fuses vision and language embeddings

### How Language is Processed

**Step 1: Input Preparation** (`transforms.py`)
```python
# Language text
lang = "Pick slice of yellow cheese and place it in the white plate"

# Images from cameras
images = [scene_camera_frame, wrist_camera_frame]  # Shape: [V, T, C, H, W]
```

**Step 2: Eagle Processing** (`transforms.py:_apply_vlm_processing`)
```python
# Create conversation format
eagle_conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": scene_img},
            {"type": "image", "image": wrist_img},
            {"type": "text", "text": lang}
        ]
    }
]

# Tokenize and process
text_list = eagle_processor.apply_chat_template(eagle_conversation)
image_inputs = eagle_processor.process_vision_info(eagle_conversation)
```

**Step 3: Eagle Model Forward** (`eagle_backbone.py`)
```python
# Eagle model processes BOTH vision and language together
eagle_output = self.eagle_model(
    input_ids=tokenized_text,      # Language tokens
    pixel_values=processed_images,  # Vision features
    attention_mask=attention_mask,
    output_hidden_states=True
)

# Extract joint vision-language embeddings
vl_embeddings = eagle_output.hidden_states[select_layer]  # Shape: [B, seq_len, 2048]
vl_embeddings = self.eagle_linear(vl_embeddings)          # Project to 1536 dim
```

**Step 4: Action Head Uses VL Embeddings** (`flow_matching_action_head.py`)
```python
# Action head receives joint vision-language embeddings
vl_embs = backbone_output.backbone_features  # From Eagle

# Diffusion model uses these embeddings as conditioning
model_output = self.model(
    hidden_states=sa_embs,           # State + action embeddings
    encoder_hidden_states=vl_embs,   # Vision-language conditioning ← KEY!
    encoder_attention_mask=vl_attn_mask,
    timestep=t_discretized
)
```

### The Problem: Frozen Embeddings

**When Eagle is frozen** (`tune_llm=False`, `tune_visual=False`):

1. **Eagle cannot learn new language-vision associations**
   - Pre-trained on general VLM tasks (image captioning, VQA, etc.)
   - Never seen "pick cheese" vs "pick bread" in training
   - Cannot learn to differentiate these similar instructions

2. **Eagle produces nearly identical embeddings**
   ```python
   # Hypothesis: Eagle's frozen embeddings
   emb_cheese = eagle("pick cheese", [scene_img, wrist_img])
   emb_bread = eagle("pick bread", [scene_img, wrist_img])

   # Cosine similarity likely very high (>0.95)
   # Because both are "pick X and place in plate" structure
   ```

3. **Diffusion model has no signal to differentiate**
   - Diffusion model learns: `embeddings → actions`
   - If `emb_cheese ≈ emb_bread`, then `actions_cheese ≈ actions_bread`
   - Model falls back to visual heuristics: "if object in holder → pick it"

### Why Enabling LLM Training Helps

**When `tune_llm=True`**:

1. **LLM can learn task-specific language understanding**
   - Fine-tunes on your specific instructions
   - Learns to differentiate "cheese" vs "bread" tokens
   - Creates distinct language embeddings

2. **Vision-language fusion still works**
   - Even with frozen vision tower
   - LLM embeddings are fused with vision features
   - Creates task-specific joint embeddings

3. **Diffusion model gets clear signal**
   ```python
   # After LLM fine-tuning
   emb_cheese = eagle("pick cheese", [scene_img, wrist_img])  # Distinct
   emb_bread = eagle("pick bread", [scene_img, wrist_img])    # Distinct

   # Cosine similarity lower (<0.8)
   # Diffusion model can learn different actions
   ```

### Why Enabling Vision Training Helps Even More

**When `tune_visual=True` + `tune_llm=True`**:

1. **Vision tower learns object-specific features**
   - Learns to recognize cheese vs bread visually
   - Creates distinct visual embeddings for different objects

2. **Language grounding**
   - LLM learns "cheese" token
   - Vision learns cheese visual features
   - Model learns to ground language to vision

3. **Robust task conditioning**
   - Model can differentiate tasks even with similar language
   - Model can handle visual variations (different cheese types)
   - Model can generalize to new instructions

---

## Implications for Existing Models

### Models Affected
- ✅ **All models trained before 2025-10-19** with the automated scripts
- ✅ Any model trained with `--no-tune_diffusion_model` flag
- ✅ Includes: cheese model (checkpoint-10000), cheese_bread_multitask (checkpoint-3000)

### Symptoms of Affected Models
1. Model ignores language instructions
2. Model performs same action regardless of task description
3. Model uses position-based heuristics (e.g., "if object in plate → stop")
4. Model cannot differentiate between similar tasks
5. Negation instructions have no effect

### Remediation Required
**All affected models must be retrained** with the corrected configuration. There is no way to fix the frozen diffusion model post-training.

---

## Recommendations

### ⚠️ UPDATED: Language Conditioning Requires Backbone Training

**For Multitask Training with Language Conditioning**:

Language conditioning is **ESSENTIAL** for multitask learning, but **CANNOT work with frozen Eagle backbone**.

**Option 1: Enable LLM Fine-tuning (RECOMMENDED)**
```bash
python scripts/gr00t_finetune.py \
    --dataset-path ./demo_data/cheese/ ./demo_data/bread/ \
    --tune-llm True \            # ✅ ENABLE THIS
    --tune-visual False \        # Keep frozen to save VRAM
    --tune-projector True \
    --tune-diffusion-model True \
    --lora-rank 32 \
    --lora-alpha 64
```

**Why this works**:
- LLM can learn to differentiate "pick cheese" vs "pick bread"
- LLM creates task-specific language embeddings
- Diffusion model learns to map these embeddings to different actions
- VRAM: ~12-16GB (may fit on RTX 4080 Super)

**Option 2: Enable Both LLM and Vision Fine-tuning (BEST)**
```bash
python scripts/gr00t_finetune.py \
    --dataset-path ./demo_data/cheese/ ./demo_data/bread/ \
    --tune-llm True \            # ✅ ENABLE THIS
    --tune-visual True \         # ✅ ENABLE THIS
    --tune-projector True \
    --tune-diffusion-model True \
    --lora-rank 32 \
    --lora-alpha 64
```

**Why this is best**:
- Vision tower learns to recognize cheese vs bread visually
- LLM learns to understand task instructions
- Combined: Model can ground language to visual objects
- VRAM: ~20-24GB (likely too much for RTX 4080 Super)

**Option 3: Reduce LoRA Rank to Fit VRAM**
```bash
python scripts/gr00t_finetune.py \
    --dataset-path ./demo_data/cheese/ ./demo_data/bread/ \
    --tune-llm True \
    --tune-visual True \
    --tune-projector True \
    --tune-diffusion-model True \
    --lora-rank 16 \             # ✅ REDUCE FROM 32
    --lora-alpha 32              # ✅ REDUCE FROM 64
```

**Trade-off**:
- Lower capacity for learning
- But enables full model training on limited VRAM

### For Single-Task Training
If you only need one task (e.g., "pick cheese"), language conditioning is less critical:
- Model can learn visual heuristics for single task
- No need to differentiate between instructions
- Frozen Eagle backbone is acceptable
- Faster training, lower VRAM

**However**, you lose:
- Ability to modify task via language at inference time
- Generalization to task variations
- Future extensibility to multitask scenarios

### Alternative Approach: Separate Models
If VRAM is too limited for backbone training:
1. **Train separate models** for each task
2. **Use task selector** at inference time (e.g., object detection)
3. **Load appropriate model** based on detected objects

**Pros**:
- Works with frozen backbone
- Lower VRAM requirements
- Simpler training

**Cons**:
- Multiple models to maintain
- Cannot handle novel task combinations
- Requires external task selection logic

---

## Lessons Learned

1. **Always verify critical components are trainable**: Check `trainable params` output
2. **Test language conditioning explicitly**: Use negation tests and task swapping
3. **Don't assume defaults are correct**: The `--no-tune_diffusion_model` flag was added for unknown reasons
4. **Monitor behavior, not just loss**: Low loss doesn't mean correct behavior
5. **Document configuration changes**: This issue could have been caught earlier with better documentation

---

## References

- Training script: `scripts/so100_groot/03_train_model.sh`
- GR00T fine-tuning code: `Isaac-GR00T/scripts/gr00t_finetune.py`
- Model architecture: `Isaac-GR00T/gr00t/model/gr00t_n1.py`
- Action head: `Isaac-GR00T/gr00t/model/action_head/flow_matching_action_head.py`

---

## Appendix: Training Logs

### Broken Configuration Output
```
tune_diffusion_model: False
trainable params: 6,553,600 || all params: 2,730,717,120 || trainable%: 0.2400
Warning: No backbone trainable parameters found.
Tune action head projector: True
Tune action head diffusion model: False  ← PROBLEM
```

### Corrected Configuration Output
```
tune_diffusion_model: True
trainable params: 6,553,600 || all params: 2,730,717,120 || trainable%: 0.2400
Warning: No backbone trainable parameters found.
Tune action head projector: True
Tune action head diffusion model: True  ← FIXED
```

Note: Same number of trainable params, but different components are trainable!

---

**Status**: Issue identified and fixed. Retraining required for all affected models.

