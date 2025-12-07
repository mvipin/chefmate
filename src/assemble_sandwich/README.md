# Sandwich Assembly Task

This directory contains the implementation of sandwich assembly tasks for LeIsaac, providing generalized ingredient manipulation with language prompt support for VLA training.

## Overview

The sandwich assembly task provides a generalized environment for ingredient manipulation (bread, patty, cheese) using language prompts to differentiate between ingredient types. This design is optimized for Vision-Language-Action (VLA) model training with GR00T N1.5, allowing a single environment to handle multiple ingredient types through natural language descriptions.

## Available Environments

### 1. Generalized Ingredient Manipulation (⭐ Recommended for VLA Training)
- **Environment ID**: `LeIsaac-SO101-AssembleSandwich-Mimic-v0`
- **Description**: Language prompt-based ingredient manipulation
- **Use Case**: VLA training with GR00T N1.5
- **Key Features**: Single environment for all ingredient types with language prompt differentiation

### 2. Basic Sandwich Assembly
- **Environment ID**: `LeIsaac-SO101-AssembleSandwich-v0`
- **Description**: Basic single-arm sandwich assembly (non-MimicGen)
- **Use Case**: Simple manipulation tasks without MimicGen

### 3. Bi-Arm Sandwich Assembly
- **Environment ID**: `LeIsaac-SO101-AssembleSandwich-BiArm-v0`
- **Description**: Two-arm version of the sandwich assembly
- **Use Case**: Bi-manual manipulation training

## Key Features

### Language Prompt Support
The generalized environment (`AssembleSandwich-Mimic-v0`) supports language prompt differentiation:
- **Bread**: "Grasp bread slice and place on plate"
- **Patty**: "Grasp patty and place on plate"
- **Cheese**: "Grasp cheese slice and place on plate"

### Generic Object References
- `ingredient` - Maps to any ingredient type (bread/patty/cheese)
- `plate` - Destination container
- `grasp_ingredient` / `place_ingredient` - Generic termination signals

### MimicGen Configuration
- Optimized parameters for ingredient manipulation
- Higher trial counts for better generalization
- Lower noise for precise manipulation
- More interpolation steps for smoother motion

## Usage Examples

### Basic Environment Usage
```python
import gymnasium as gym

# Recommended: Generalized ingredient environment for VLA training
env = gym.make("LeIsaac-SO101-AssembleSandwich-Mimic-v0")

# Basic assembly environment (non-MimicGen)
env = gym.make("LeIsaac-SO101-AssembleSandwich-v0")
```

### MimicGen Data Generation
```bash
# Generate data for generalized ingredient manipulation
python scripts/mimic/collect_demonstrations.py \
    --task LeIsaac-SO101-AssembleSandwich-Mimic-v0 \
    --num_demos 50 \
    --output_file ./datasets/sandwich_ingredient_demos.hdf5

# Focus on the generalized ingredient environment for VLA training
# This single environment handles all ingredient types via language prompts
```

### Language Prompt Integration
For GR00T N1.5 training, use the conversion script with custom language prompts:
```bash
# Convert bread ingredient dataset
python scripts/convert/isaaclab2lerobot.py \
    --repo_id your_username/bread_dataset \
    --task "Grasp bread slice and place on plate"

# Convert patty ingredient dataset
python scripts/convert/isaaclab2lerobot.py \
    --repo_id your_username/patty_dataset \
    --task "Grasp patty and place on plate"

# Convert cheese ingredient dataset
python scripts/convert/isaaclab2lerobot.py \
    --repo_id your_username/cheese_dataset \
    --task "Grasp cheese slice and place on plate"
```

## Asset Dependencies

### USD Scene Files
- **Primary**: `assets/scenes/kitchen_with_sandwich/scene.usd`

### Object Names in USD
- `bread_slice_1` - First bread slice
- `bread_slice_2` - Second bread slice  
- `patty` - Meat patty
- `cheese_slice` - Cheese slice
- `plate` - Destination container

## MimicGen Data Augmentation Workflow

This section provides a comprehensive end-to-end pipeline for collecting, processing, and augmenting demonstration data for the sandwich ingredient task using MimicGen.

### Overview

The MimicGen workflow transforms a small number of human demonstrations into a large dataset of synthetic demonstrations through data augmentation. This process involves several key steps: teleoperation collection, verification, action space conversion, subtask annotation, synthetic generation, and optional format conversion for VLA training.

### Task Selection Guide

**Important**: The sandwich assembly task has two separate task registrations for different purposes:

1. **`LeIsaac-SO101-AssembleSandwich-v0`** (Regular Task)
   - **Use for**: Teleoperation, data collection, replay
   - **Action space**: Joint position control (6D: 5 arm joints + 1 gripper)
   - **Environment**: Standard `ManagerBasedRLEnv`

2. **`LeIsaac-SO101-AssembleSandwich-Mimic-v0`** (Mimic Task)
   - **Use for**: MimicGen data generation only (Steps 4-5)
   - **Action space**: IK pose control (8D: 7 pose + 1 gripper)
   - **Environment**: `ManagerBasedRLLeIsaacMimicEnv` with MimicGen support

**Workflow Summary**:
- Steps 1-3 (Collection, Replay, IK Conversion): Use **regular task** (`-v0`)
- Steps 4-5 (Annotation, Generation): Use **Mimic task** (`-Mimic-v0`)
- Step 6 (Joint Conversion): Use **Mimic task** (`-Mimic-v0`)

This pattern follows the standard approach used by other tasks in the LeIsaac framework (e.g., `pick_orange`, `lift_cube`).

### Prerequisites

- SO101Leader teleoperation device properly configured
- Isaac Sim environment set up with LeIsaac extension
- CUDA-compatible GPU for simulation and processing

### Step 1: Collect Human Demonstrations via Teleoperation

Use the SO101Leader device to collect high-quality human demonstrations for ingredient manipulation.

**Important**: Use the regular task (`LeIsaac-SO101-AssembleSandwich-v0`) for teleoperation, not the Mimic task. The Mimic task is specifically designed for MimicGen data generation.

```bash
# Collect bread slice demonstrations
/home/vipin/IsaacSim/_build/linux-x86_64/release/python.sh scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-AssembleSandwich-v0 \
    --teleop_device=so101leader \
    --port=/dev/leader \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --record \
    --dataset_file=./datasets/bread_ingredient_demos.hdf5

# Collect patty demonstrations
/home/vipin/IsaacSim/_build/linux-x86_64/release/python.sh scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-AssembleSandwich-v0 \
    --teleop_device=so101leader \
    --port=/dev/leader \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --record \
    --dataset_file=./datasets/patty_ingredient_demos.hdf5

# Collect cheese slice demonstrations
/home/vipin/IsaacSim/_build/linux-x86_64/release/python.sh scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-SO101-AssembleSandwich-v0 \
    --teleop_device=so101leader \
    --port=/dev/leader \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --record \
    --dataset_file=./datasets/cheese_ingredient_demos.hdf5
```

**Parameters Explained:**
- `--task`: The environment to use for data collection
- `--teleop_device`: SO101Leader device for teleoperation
- `--port`: Device port (typically `/dev/leader`)
- `--num_envs`: Number of parallel environments (1 for teleoperation)
- `--device`: CUDA device for GPU acceleration
- `--enable_cameras`: Record camera observations for VLA training
- `--record`: Enable demonstration recording
- `--dataset_file`: Output HDF5 file path

**Collection Tips:**
- Collect 5-10 high-quality demonstrations per ingredient type
- Focus on smooth, consistent motions
- Ensure clear grasp and place actions
- Use descriptive filenames to distinguish ingredient types

### Step 2: Replay and Verify Demonstrations

Verify that collected demonstrations can be replayed correctly before proceeding with processing.

**Note**: Use the same task that was used for recording (`LeIsaac-SO101-AssembleSandwich-v0`).

```bash
# Replay bread demonstrations
~/IsaacLab/_isaac_sim/python.sh scripts/environments/teleoperation/replay.py \
    --task=LeIsaac-SO101-AssembleSandwich-v0 \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --replay_mode=action \
    --dataset_file=./datasets/bread_ingredient_demos.hdf5

# Replay patty demonstrations
~/IsaacLab/_isaac_sim/python.sh scripts/environments/teleoperation/replay.py \
    --task=LeIsaac-SO101-AssembleSandwich-v0 \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --replay_mode=action \
    --dataset_file=./datasets/patty_ingredient_demos.hdf5

# Replay cheese demonstrations
~/IsaacLab/_isaac_sim/python.sh scripts/environments/teleoperation/replay.py \
    --task=LeIsaac-SO101-AssembleSandwich-v0 \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --replay_mode=action \
    --dataset_file=./datasets/cheese_ingredient_demos.hdf5
```

**Parameters Explained:**
- `--replay_mode=action`: Replay using recorded actions (most reliable)
- Other parameters same as collection step

**Verification Checklist:**
- Demonstrations replay without errors
- Robot motions appear smooth and natural
- Objects are successfully grasped and placed
- Camera observations are recorded correctly

### Step 3: Convert End-Effector Actions to IK

Convert Cartesian end-effector actions to inverse kinematics for MimicGen processing.

```bash
# Process bread demonstrations
/home/vipin/IsaacSim/_build/linux-x86_64/release/python.sh scripts/mimic/eef_action_process.py \
    --input_file=./datasets/bread_ingredient_demos.hdf5 \
    --output_file=./datasets/processed_bread_ingredient.hdf5 \
    --to_ik \
    --device=cuda \
    --headless

# Process patty demonstrations
/home/vipin/IsaacSim/_build/linux-x86_64/release/python.sh scripts/mimic/eef_action_process.py \
    --input_file=./datasets/patty_ingredient_demos.hdf5 \
    --output_file=./datasets/processed_patty_ingredient.hdf5 \
    --to_ik \
    --device=cuda \
    --headless

# Process cheese demonstrations
/home/vipin/IsaacSim/_build/linux-x86_64/release/python.sh scripts/mimic/eef_action_process.py \
    --input_file=./datasets/cheese_ingredient_demos.hdf5 \
    --output_file=./datasets/processed_cheese_ingredient.hdf5 \
    --to_ik \
    --device=cuda \
    --headless
```

**Parameters Explained:**
- `--to_ik`: Convert end-effector actions to inverse kinematics
- `--headless`: Run without GUI for faster processing
- `--device=cuda`: Use GPU acceleration

**Purpose:** IK conversion enables MimicGen to manipulate demonstrations in joint space while preserving end-effector trajectories.

### Step 4: Annotate Demonstrations with Subtask Signals

Add MimicGen subtask termination signals to enable proper segmentation during data generation.

```bash
# Annotate bread demonstrations
/home/vipin/IsaacSim/_build/linux-x86_64/release/python.sh scripts/mimic/annotate_demos.py \
    --task=LeIsaac-SO101-AssembleSandwich-Mimic-v0 \
    --input_file=./datasets/processed_bread_ingredient.hdf5 \
    --output_file=./datasets/annotated_bread_ingredient.hdf5 \
    --device=cuda \
    --enable_cameras \
    --auto

# Annotate patty demonstrations
/home/vipin/IsaacSim/_build/linux-x86_64/release/python.sh scripts/mimic/annotate_demos.py \
    --task=LeIsaac-SO101-AssembleSandwich-Mimic-v0 \
    --input_file=./datasets/processed_patty_ingredient.hdf5 \
    --output_file=./datasets/annotated_patty_ingredient.hdf5 \
    --device=cuda \
    --enable_cameras \
    --auto

# Annotate cheese demonstrations
/home/vipin/IsaacSim/_build/linux-x86_64/release/python.sh scripts/mimic/annotate_demos.py \
    --task=LeIsaac-SO101-AssembleSandwich-Mimic-v0 \
    --input_file=./datasets/processed_cheese_ingredient.hdf5 \
    --output_file=./datasets/annotated_cheese_ingredient.hdf5 \
    --device=cuda \
    --enable_cameras \
    --auto
```

**Parameters Explained:**
- `--auto`: Use automatic annotation based on environment signals
- Alternative: Remove `--auto` for manual annotation interface

**Subtask Signals for Sandwich Ingredient Environment:**
- `grasp_ingredient`: Triggered when ingredient is successfully grasped
- `place_ingredient`: Triggered when ingredient is placed on plate (final subtask)

**Automatic Detection:** The `grasp_ingredient` signal is automatically detected using:
- **Distance threshold**: Ingredient within 0.05m of end-effector
- **Gripper closure**: Gripper joint position below 0.60 threshold
- **Multi-ingredient support**: Works with bread_slice_1, bread_slice_2, patty, or cheese_slice

**Purpose:** Subtask annotations enable MimicGen to understand task structure and generate variations that respect the logical sequence of actions.

### Step 5: Generate Augmented Demonstrations

Use MimicGen to create synthetic demonstrations from the annotated human demonstrations.

```bash
# Generate augmented bread demonstrations
/home/vipin/IsaacSim/_build/linux-x86_64/release/python.sh scripts/mimic/generate_dataset.py \
    --task=LeIsaac-SO101-AssembleSandwich-Mimic-v0 \
    --input_file=./datasets/annotated_bread_ingredient.hdf5 \
    --output_file=./datasets/generated_bread_ingredient.hdf5 \
    --ingredient_type=bread_slice_1 \
    --generation_num_trials=20 \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras

# Generate augmented patty demonstrations
/home/vipin/IsaacSim/_build/linux-x86_64/release/python.sh scripts/mimic/generate_dataset.py \
    --task=LeIsaac-SO101-AssembleSandwich-Mimic-v0 \
    --input_file=./datasets/annotated_patty_ingredient.hdf5 \
    --output_file=./datasets/generated_patty_ingredient.hdf5 \
    --ingredient_type=patty \
    --generation_num_trials=20 \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras

# Generate augmented cheese demonstrations
/home/vipin/IsaacSim/_build/linux-x86_64/release/python.sh scripts/mimic/generate_dataset.py \
    --task=LeIsaac-SO101-AssembleSandwich-Mimic-v0 \
    --input_file=./datasets/annotated_cheese_ingredient.hdf5 \
    --output_file=./datasets/generated_cheese_ingredient.hdf5 \
    --ingredient_type=cheese_slice \
    --generation_num_trials=20 \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras
```

**Parameters Explained:**
- `--ingredient_type`: **NEW!** Dynamically sets which ingredient object to track during generation
  - Options: `bread_slice_1`, `bread_slice_2`, `cheese_slice`, `patty`
  - Must match the ingredient used in your demonstrations
  - Automatically updates subtask object references and descriptions
- `--generation_num_trials=20`: Number of synthetic demonstrations to generate per source demo
- `--num_envs=1`: Number of parallel environments for generation

**Expected Output:** Each input demonstration generates ~20 synthetic variations, significantly expanding your dataset size.

**Important Note:** The `--ingredient_type` parameter is **required** for the sandwich assembly task to ensure MimicGen tracks the correct object during data generation. Make sure it matches the ingredient you demonstrated in your source dataset.

### Step 6: Convert IK Actions Back to Joint Space

Convert the generated IK actions back to joint-space actions for policy training.

```bash
# Convert bread demonstrations to joint space
/home/vipin/IsaacSim/_build/linux-x86_64/release/python.sh scripts/mimic/eef_action_process.py \
    --input_file=./datasets/generated_bread_ingredient.hdf5 \
    --output_file=./datasets/final_bread_ingredient.hdf5 \
    --to_joint \
    --device=cuda \
    --headless

# Convert patty demonstrations to joint space
/home/vipin/IsaacSim/_build/linux-x86_64/release/python.sh scripts/mimic/eef_action_process.py \
    --input_file=./datasets/generated_patty_ingredient.hdf5 \
    --output_file=./datasets/final_patty_ingredient.hdf5 \
    --to_joint \
    --device=cuda \
    --headless

# Convert cheese demonstrations to joint space
/home/vipin/IsaacSim/_build/linux-x86_64/release/python.sh scripts/mimic/eef_action_process.py \
    --input_file=./datasets/generated_cheese_ingredient.hdf5 \
    --output_file=./datasets/final_cheese_ingredient.hdf5 \
    --to_joint \
    --device=cuda \
    --headless
```

**Parameters Explained:**
- `--to_joint`: Convert IK actions back to joint-space actions
- Final datasets are ready for policy training

### Step 7: Convert to LeRobot Format for VLA Training

Convert each HDF5 dataset to LeRobot format with language prompts for GR00T N1.5 training. Process one ingredient type at a time for better organization and clarity.

```bash
# Convert bread ingredient dataset
python scripts/convert/isaaclab2lerobot.py \
    --repo_id your_username/bread_ingredient_dataset \
    --task "Grasp bread slice and place on plate"

# Convert patty ingredient dataset
python scripts/convert/isaaclab2lerobot.py \
    --repo_id your_username/patty_ingredient_dataset \
    --task "Grasp patty and place on plate"

# Convert cheese ingredient dataset
python scripts/convert/isaaclab2lerobot.py \
    --repo_id your_username/cheese_ingredient_dataset \
    --task "Grasp cheese slice and place on plate"
```

**Parameters Explained:**
- `--repo_id`: Hugging Face repository ID for dataset storage (optional, has default)
- `--task`: Language prompt describing the manipulation task (optional, has default)

**Default Values:**
- Default `--repo_id`: `'EverNorif/so101_test_orange_pick'`
- Default `--task`: `'Grab orange and place into plate'`

**Command-Line Options:**
```bash
# View all available options
python scripts/convert/isaaclab2lerobot.py --help

# Use default values (same as original behavior)
python scripts/convert/isaaclab2lerobot.py

# Specify custom repository ID only
python scripts/convert/isaaclab2lerobot.py --repo_id username/my_dataset

# Specify custom task description only
python scripts/convert/isaaclab2lerobot.py --task "Custom manipulation task"

# Specify both parameters
python scripts/convert/isaaclab2lerobot.py \
    --repo_id username/my_dataset \
    --task "Custom manipulation task"
```

**Language Prompt Integration:**
- Each ingredient type gets a specific language description
- Language prompts are integrated at the frame level
- Ready for VLA model training with language conditioning
- One dataset per ingredient type for better organization

### Step 8 (Optional): Visualize Dataset

Verify the converted dataset using LeRobot's visualization tools.

```bash
# Visualize the dataset
python -m lerobot.scripts.visualize_dataset \
    --repo-id your_username/sandwich_ingredients_dataset \
    --episode-index 1
```

## Language Prompt Integration for VLA Training

### Naming Conventions for Ingredient-Specific Datasets

When collecting demonstrations for different ingredient types, use descriptive filenames to enable automatic language prompt assignment:

```bash
# Recommended naming pattern
./datasets/bread_ingredient_demos.hdf5     → "Grasp bread slice and place on plate"
./datasets/patty_ingredient_demos.hdf5     → "Grasp patty and place on plate"
./datasets/cheese_ingredient_demos.hdf5    → "Grasp cheese slice and place on plate"
```

### Multi-Task Dataset Creation

The generalized sandwich ingredient environment enables creating a single multi-task dataset:

1. **Collect separate datasets** for each ingredient type using the same environment
2. **Process each dataset** through the complete MimicGen pipeline
3. **Combine datasets** during LeRobot conversion with appropriate language prompts
4. **Train VLA model** on the combined multi-task dataset

This approach allows the VLA model to learn the fundamental manipulation skills while using language prompts to specify the target ingredient type.

## Troubleshooting

### Common Issues in MimicGen Pipeline

#### 1. Teleoperation Collection Issues

**Problem:** SO101Leader device not detected
```bash
# Check device connection
ls /dev/leader
# If not found, check USB connections and device permissions
sudo chmod 666 /dev/ttyUSB*
```

**Problem:** Jerky or unstable teleoperation
- Ensure SO101Leader is properly calibrated
- Check for electromagnetic interference
- Reduce teleoperation speed if needed

#### 2. Replay Verification Failures

**Problem:** Demonstrations don't replay correctly
```bash
# Try different replay modes
--replay_mode=action    # Most reliable
--replay_mode=state     # Alternative if action replay fails
```

**Problem:** Robot goes out of bounds during replay
- Check initial robot position in environment configuration
- Verify workspace limits in demonstrations
- Re-collect demonstrations with more conservative motions

#### 3. Action Processing Issues

**Problem:** IK conversion fails
```bash
# Check for joint limit violations
# Verify end-effector poses are reachable
# Try with --debug flag for detailed error messages
```

**Problem:** Joint space conversion produces unrealistic motions
- Verify IK solutions are valid
- Check for singularities in robot configuration
- Consider adjusting IK solver parameters

#### 4. Annotation Failures

**Problem:** Automatic annotation doesn't detect subtask signals
```bash
# Switch to manual annotation
# Remove --auto flag and use interactive interface
/home/vipin/IsaacSim/_build/linux-x86_64/release/python.sh scripts/mimic/annotate_demos.py \
    --task=LeIsaac-SO101-AssembleSandwich-Mimic-v0 \
    --input_file=./datasets/processed_ingredient.hdf5 \
    --output_file=./datasets/annotated_ingredient.hdf5 \
    --device=cuda \
    --enable_cameras
```

**Problem:** Subtask signals trigger at wrong times
- Review environment configuration for signal thresholds
- Adjust grasp detection parameters
- Manually annotate critical demonstrations

#### 5. Generation Failures

**Problem:** MimicGen generation produces low success rates
```bash
# Increase number of trials
--generation_num_trials=50

# Adjust selection strategy parameters
# Check environment configuration for nn_k and other parameters
```

**Problem:** Generated demonstrations are unrealistic
- Review source demonstration quality
- Check annotation accuracy
- Adjust noise parameters in environment configuration

#### 6. Device Compatibility Issues

**Problem:** CUDA out of memory errors
```bash
# Reduce number of parallel environments
--num_envs=1

# Use CPU if necessary (slower)
--device=cpu
```

**Problem:** Isaac Sim crashes during processing
- Ensure sufficient GPU memory (8GB+ recommended)
- Close other GPU-intensive applications
- Use `--headless` flag to reduce memory usage

### Performance Optimization Tips

1. **Batch Processing:** Process multiple ingredient types in parallel if you have sufficient GPU memory
2. **Storage Management:** Use SSD storage for faster I/O during dataset processing
3. **Memory Management:** Monitor GPU memory usage and adjust batch sizes accordingly
4. **Quality Control:** Regularly verify intermediate outputs to catch issues early

### Getting Help

If you encounter issues not covered here:
1. Check Isaac Sim and LeIsaac documentation
2. Review environment configuration files for parameter details
3. Use `--debug` flags for detailed error messages
4. Consider collecting additional demonstrations if generation quality is poor

## Configuration Details

### Generalized Environment Parameters
- **Trials**: 20 (optimized for better generalization)
- **Neighbors**: 5 (increased for better selection)
- **Action Noise**: 0.002 (reduced for precise manipulation)
- **Interpolation Steps**: 8 (increased for smoother motion)

### Subtask Structure
1. **Grasp Phase**: `object_ref="ingredient"`, `subtask_term_signal="grasp_ingredient"`
2. **Place Phase**: `object_ref="plate"`, `subtask_term_signal=None` (final)

## Best Practices

### For VLA Training
1. Use `LeIsaac-SO101-AssembleSandwich-Mimic-v0` for language prompt training
2. Collect separate datasets for each ingredient type
3. Use descriptive filenames: `bread_demos.hdf5`, `patty_demos.hdf5`, `cheese_demos.hdf5`
4. Apply language prompts during HDF5 → LeRobot conversion

### For Basic Environments
1. Use `LeIsaac-SO101-AssembleSandwich-v0` for simple manipulation tasks
2. Use `LeIsaac-SO101-AssembleSandwich-BiArm-v0` for bi-manual coordination
3. These environments don't require MimicGen processing

## Summary

The sandwich assembly task provides a comprehensive framework for ingredient manipulation with language prompt support. The generalized `AssembleSandwich-Mimic-v0` environment is specifically designed for VLA training, allowing a single environment to handle multiple ingredient types through natural language descriptions.

**Key Benefits:**
- **Single Environment**: One environment for all ingredient types
- **Language Integration**: Built-in support for VLA training
- **MimicGen Ready**: Complete data augmentation pipeline
- **Optimized Parameters**: Enhanced settings for better generalization

**Recommended Workflow:**
1. Collect demonstrations for each ingredient type
2. Process through MimicGen pipeline (Steps 1-6)
3. Convert each ingredient dataset to LeRobot format with specific language prompts (Step 7)
4. Train VLA model on individual or combined datasets

This approach enables efficient training of vision-language-action models that can understand and execute ingredient manipulation tasks based on natural language instructions.
