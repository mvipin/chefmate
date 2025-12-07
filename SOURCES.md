# Source Code Attribution

This repository contains comprehensive project documentation and reference implementations for the ChefMate project (multi-ingredient sandwich assembly using NVIDIA GR00T N1.5 VLA model).

## Reference Code Sources

### 1. Training Pipeline: `scripts/so100_groot/`

**Source Repository**: [lerobot](https://github.com/Seeed-Projects/lerobot)  
**Original Location**: `scripts/so100_groot/`  
**Copied Date**: 2025-12-06  
**Purpose**: GR00T N1.5 training scripts, deployment configuration, and troubleshooting documentation

**Key Files**:
| File | Description |
|------|-------------|
| `03_train_model.sh` | GR00T N1.5 fine-tuning configuration with corrected language conditioning |
| `LANGUAGE_CONDITIONING_FIX.md` | Critical fix for language conditioning failure |
| `MULTITASK_TRAINING_ANALYSIS.md` | Technical analysis of multitask training challenges |
| `README.md` | Complete workflow documentation |
| `04_start_inference_server.sh` | Inference server setup |
| `05_deploy_robot.sh` | Robot deployment script |

### 2. Task Implementation: `src/assemble_sandwich/`

**Source Repository**: [leisaac](https://github.com/mvipin/leisaac) (fork of LightwheelAI/leisaac)  
**Original Location**: `source/leisaac/leisaac/tasks/assemble_sandwich/`  
**Copied Date**: 2025-12-06  
**Purpose**: Isaac Sim environment configuration, MimicGen integration, and task definition

**Key Files**:
| File | Description |
|------|-------------|
| `assemble_sandwich_mimic_env_cfg.py` | MimicGen-enabled environment with language prompts |
| `assemble_sandwich_env_cfg.py` | Base environment configuration |
| `mdp/observations.py` | Custom observation terms for subtask detection |
| `mdp/terminations.py` | Task termination conditions |
| `__init__.py` | Gym environment registration |
| `README.md` | Original task implementation documentation |

---

## Active Development Repositories

For active development and contributions, refer to the original repositories:

### Training Pipeline
- **Repository**: https://github.com/Seeed-Projects/lerobot
- **Branch**: `main`
- **Path**: `scripts/so100_groot/`

### Simulation Environment
- **Repository**: https://github.com/mvipin/leisaac
- **Branch**: `main`
- **Path**: `source/leisaac/leisaac/tasks/assemble_sandwich/`

---

## Version Information

| Component | Version/Commit | Date |
|-----------|----------------|------|
| lerobot training scripts | 2025-10-20 | Language conditioning fix applied |
| leisaac task implementation | 2025-10-16 | MimicGen integration complete |
| GR00T N1.5 | v1.5 | NVIDIA Isaac GR00T |
| Isaac Lab | 1.2.0 | NVIDIA simulation framework |

---

## License

This documentation repository is licensed under the MIT License.

Original source code licenses:
- **lerobot**: Apache 2.0 License
- **leisaac**: MIT License (LightwheelAI)
