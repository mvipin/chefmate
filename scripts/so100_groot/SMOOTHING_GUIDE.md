# GR00T Action Smoothing Guide

## Problem: Jittery/Shaky Robot Motion

When deploying GR00T models, you may notice the robot shakes or has jittery motion even though it understands the task. This is caused by:

1. **No action filtering** - Raw model outputs sent directly to motors
2. **High-frequency action updates** - Actions sent every 20ms without smoothing
3. **Action discontinuities** - Sudden jumps between action chunks
4. **Model noise** - VLA models can produce slightly noisy predictions

## Solution: Action Smoothing

We've created a smoothed deployment script that applies:

1. **Exponential Moving Average (EMA)** - Smooths actions over time
2. **Velocity Limiting** - Prevents sudden large movements
3. **Configurable sleep time** - Allows motors to settle between actions

## Usage

### Quick Start (Recommended)

Stop the current deployment (Ctrl+C) and run:

```bash
./scripts/so100_groot/05b_deploy_robot_smooth.sh medium
```

### Smoothing Levels

#### 1. Light Smoothing (Fastest, Minimal Filtering)
```bash
./scripts/so100_groot/05b_deploy_robot_smooth.sh light
```
- Smoothing alpha: 0.7
- Max velocity: 10.0
- Action sleep: 0.03s
- **Use when**: Task requires fast, responsive movements

#### 2. Medium Smoothing (Balanced) ⭐ **RECOMMENDED**
```bash
./scripts/so100_groot/05b_deploy_robot_smooth.sh medium
```
- Smoothing alpha: 0.4
- Max velocity: 5.0
- Action sleep: 0.05s
- **Use when**: General pick-and-place tasks (like cheese task)

#### 3. Heavy Smoothing (Slowest, Maximum Filtering)
```bash
./scripts/so100_groot/05b_deploy_robot_smooth.sh heavy
```
- Smoothing alpha: 0.2
- Max velocity: 3.0
- Action sleep: 0.08s
- **Use when**: Precision tasks, delicate objects, or very shaky motion

#### 4. Custom Smoothing
```bash
./scripts/so100_groot/05b_deploy_robot_smooth.sh custom
```
Then enter your own parameters when prompted.

## Parameter Explanation

### Smoothing Alpha (0.1 - 0.9)
- **Lower values (0.1-0.3)**: Very smooth, slower response
- **Medium values (0.4-0.6)**: Balanced smoothness and responsiveness
- **Higher values (0.7-0.9)**: Minimal smoothing, faster response

Formula: `smoothed_action = alpha * new_action + (1 - alpha) * previous_action`

### Max Velocity (1.0 - 10.0)
- **Lower values (1.0-3.0)**: Prevents large jumps, very smooth
- **Medium values (4.0-6.0)**: Balanced motion
- **Higher values (7.0-10.0)**: Allows faster movements

This limits how much each joint can change per action step.

### Action Sleep (0.02 - 0.1 seconds)
- **Lower values (0.02-0.04s)**: Faster execution, less settling time
- **Medium values (0.05-0.07s)**: Balanced execution speed
- **Higher values (0.08-0.1s)**: Slower execution, more settling time

Time to wait between sending actions to the robot.

## Comparison

| Aspect | Original | Light | Medium | Heavy |
|--------|----------|-------|--------|-------|
| Smoothness | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Responsiveness | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Jitter Reduction | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## Troubleshooting

### Still too shaky?
1. Try `heavy` smoothing level
2. Or use `custom` with:
   - Alpha: 0.15
   - Max velocity: 2.0
   - Action sleep: 0.1

### Too slow/sluggish?
1. Try `light` smoothing level
2. Or use `custom` with:
   - Alpha: 0.6
   - Max velocity: 8.0
   - Action sleep: 0.03

### Robot not reaching target?
- Smoothing may be too aggressive
- Try increasing alpha (e.g., 0.5 → 0.7)
- Or increase max velocity (e.g., 3.0 → 5.0)

### Robot overshooting?
- Not enough smoothing
- Try decreasing alpha (e.g., 0.5 → 0.3)
- Or decrease max velocity (e.g., 5.0 → 3.0)

## Advanced: Training Improvements

If smoothing helps but you want better base performance, consider:

1. **Collect more diverse data** - 50 episodes may not be enough
   - Target: 100-200 episodes for robust performance

2. **Improve demonstration quality**
   - Use slower, smoother teleoperation during recording
   - Avoid jerky movements in demonstrations

3. **Train longer** - You stopped at 6000 steps
   - Try resuming to 8000-10000 steps

4. **Adjust training hyperparameters**
   - Lower learning rate for smoother policies
   - Increase batch size for more stable gradients

5. **Use temporal augmentation**
   - Add noise to actions during training
   - This can make the policy more robust to jitter

## Files

- `scripts/so100_groot/eval_lerobot_smooth.py` - Smoothed evaluation script
- `scripts/so100_groot/05b_deploy_robot_smooth.sh` - Deployment wrapper with presets
- `scripts/so100_groot/05_deploy_robot.sh` - Original unsmoothed version

## Example Workflow

```bash
# Terminal 1: Start inference server (if not already running)
./scripts/so100_groot/04_start_inference_server.sh

# Terminal 2: Deploy with medium smoothing
./scripts/so100_groot/05b_deploy_robot_smooth.sh medium

# If still shaky, try heavy:
# Ctrl+C to stop, then:
./scripts/so100_groot/05b_deploy_robot_smooth.sh heavy

# If too slow, try light:
./scripts/so100_groot/05b_deploy_robot_smooth.sh light
```

## Technical Details

The smoothing implementation uses:

1. **Exponential Moving Average (EMA)**:
   ```python
   smoothed = alpha * new_action + (1 - alpha) * prev_action
   ```

2. **Velocity Limiting**:
   ```python
   delta = new_action - prev_action
   delta = clip(delta, -max_velocity, max_velocity)
   limited_action = prev_action + delta
   ```

3. **Per-joint filtering**: Each of the 6 joints (5 arm + 1 gripper) is filtered independently

This approach is commonly used in robotics to:
- Reduce sensor noise
- Smooth control signals
- Prevent mechanical stress from sudden movements
- Improve trajectory quality

## References

- Original eval script: `~/Isaac-GR00T/examples/SO-100/eval_lerobot.py`
- LeRobot documentation: https://github.com/huggingface/lerobot
- GR00T documentation: NVIDIA Isaac GR00T

