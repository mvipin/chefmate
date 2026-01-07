# Evaluation Results

> **Navigation**: [← Simulation](simulation.md) | [Main README](../README.md) | [Troubleshooting →](troubleshooting.md)

---

<a id="9-evaluation-results"></a>
## 9. Evaluation Results

This section presents quantitative evaluation of the ChefMate system across 103 structured trials, designed to systematically assess the GR00T N1.5 VLA model's capabilities in language-conditioned manipulation.

<a id="evaluation-protocol"></a>
### Evaluation Protocol

The evaluation framework consists of **103 trials across 35 test conditions**, organized into 9 research questions (RQ1-RQ10, excluding RQ7). Each test condition includes 3 trials to measure consistency.

**Evaluation Resources:**
- Detailed protocol: [`assets/benchmarks/evaluation_guide.txt`](../assets/benchmarks/evaluation_guide.txt)
- Raw data: [`assets/benchmarks/evaluation_data_collection.csv`](../assets/benchmarks/evaluation_data_collection.csv)

#### Result Codes

| Code | Description |
|------|-------------|
| `SUCCESS` | Task completed correctly (target object placed in plate) |
| `PARTIAL` | Arm moved toward correct object but grasp/release failed |
| `WRONG_OBJECT` | Grasped incorrect object or ignored instruction constraints |
| `MISS` | Failed to grasp any object |
| `NO_ACTION` | No movement (expected for RQ3 task-complete scenarios) |
| `TIMEOUT` | No completion within 30 seconds |

#### Workspace Grid

```
         Camera View
    ┌─────────────────────────┐
    │  (1)      (2)      (3)  │
    │  L-Far    C-Far    R-Far│
    ├─────────────────────────┤
    │  (4)      (5)      (6)  │
    │  L-Mid    C-Mid    R-Mid│
    ├─────────────────────────┤
    │  (7)      (8)      (9)  │
    │  L-Near   C-Near   R-Near│
    └─────────────────────────┘
          Robot Base
```

Zone 5 (center) is the training position; all other zones test spatial generalization.

<a id="performance-by-research-question"></a>
### Performance by Research Question

| RQ | Focus | Trials | SUCCESS | PARTIAL | WRONG_OBJ | MISS | NO_ACTION | Success Rate |
|----|-------|--------|---------|---------|-----------|------|-----------|--------------|
| **RQ1** | Object Discrimination | 18 | 1 | 7 | 3 | 7 | 0 | 5.6% |
| **RQ2** | Manipulation Skills | 18 | 3 | 9 | 0 | 0 | 6 | 16.7% |
| **RQ3** | Task Completion Recognition | 9 | 6 | 0 | 0 | 0 | 3 | 66.7%* |
| **RQ4** | Source Position Perturbation | 12 | 3 | 6 | 0 | 3 | 0 | 25.0% |
| **RQ5** | Destination Perturbation | 12 | 9 | 3 | 0 | 0 | 0 | 75.0% |
| **RQ6** | Distractor Handling | 9 | 0 | 9 | 0 | 0 | 0 | 0.0% |
| **RQ8** | Prompt Variations | 9 | 7 | 2 | 0 | 0 | 0 | 77.8% |
| **RQ9** | Misleading Instructions | 12 | 3 | 0 | 9 | 0 | 0 | 25.0%** |
| **RQ10** | Novel Object Generalization | 3 | 0 | 0 | 3 | 0 | 0 | 0.0% |
| | **Overall** | **103** | **32** | **36** | **15** | **10** | **9** | **31.1%** |

*RQ3 expects `NO_ACTION` when task is already complete; SUCCESS indicates correct bread recognition, NO_ACTION for cheese (failure to recognize cheese)

**RQ9 measures robustness to invalid prompts; some WRONG_OBJECT outcomes are expected behavior

<a id="object-level-performance"></a>
### Object-Level Performance

Performance varies significantly by object type, revealing training data bias:

| Object | Total Trials | SUCCESS | PARTIAL | WRONG_OBJ | MISS | NO_ACTION | Success Rate |
|--------|--------------|---------|---------|-----------|------|-----------|--------------|
| **bread_1** | 69 | 26 | 24 | 6 | 6 | 7 | 37.7% |
| **bread_2** | 18 | 6 | 6 | 3 | 3 | 0 | 33.3% |
| **cheese** | 12 | 0 | 2 | 0 | 4 | 6 | 0.0% |
| **N/A** (RQ9-MI4) | 3 | 0 | 0 | 3 | 0 | 0 | 0.0% |

**Key Observation**: The model exhibits strong bias toward bread_1 manipulation, with near-zero performance on cheese. This reflects training data composition—demonstrations primarily focused on bread manipulation.

<a id="spatial-robustness-analysis"></a>
### Spatial Robustness Analysis

#### Source Position (RQ4) — Object Location Perturbation

| Zone | Position | Trials | SUCCESS | PARTIAL | MISS | Success Rate |
|------|----------|--------|---------|---------|------|--------------|
| 1 | Far-Left | 3 | 0 | 0 | 3 | 0.0% |
| 3 | Far-Right | 3 | 3 | 0 | 0 | 100.0% |
| 7 | Near-Left | 3 | 0 | 3 | 0 | 0.0% |
| 9 | Near-Right | 3 | 0 | 3 | 0 | 0.0% |

**Analysis**: The model shows asymmetric spatial generalization—right-side positions (zone 3) succeed while left-side (zone 1) fails completely. Near positions (7, 9) achieve PARTIAL success but fail gripper release.

#### Destination Position (RQ5) — Plate Location Perturbation

| Zone | Position | Trials | SUCCESS | PARTIAL | Success Rate |
|------|----------|--------|---------|---------|--------------|
| 1 | Far-Left | 3 | 3 | 0 | 100.0% |
| 3 | Far-Right | 3 | 0 | 3 | 0.0% |
| 7 | Near-Left | 3 | 3 | 0 | 100.0% |
| 9 | Near-Right | 3 | 3 | 0 | 100.0% |

**Analysis**: Destination perturbation shows opposite pattern—left and near positions succeed while far-right (zone 3) fails. This suggests the model has difficulty with extended reach trajectories.

<a id="language-conditioning-analysis"></a>
### Language Conditioning Analysis

#### Prompt Variation Robustness (RQ8)

| Prompt | Trials | SUCCESS | PARTIAL | Success Rate |
|--------|--------|---------|---------|--------------|
| "Grab the first bread and put it in the dish" | 3 | 3 | 0 | 100.0% |
| "Take bread one to the plate" | 3 | 2 | 1 | 66.7% |
| "Move bread_1 to the circular container" | 3 | 2 | 1 | 66.7% |

**Analysis**: The model demonstrates reasonable language invariance (77.8% overall), successfully interpreting paraphrased instructions. Performance is highest with informal language ("grab", "dish") matching training distribution.

#### Misleading Instruction Handling (RQ9)

| Test | Instruction | Scene Reality | Result | Interpretation |
|------|-------------|---------------|--------|----------------|
| MI1 | "Pick up bread from the **square** plate" | Plate is round | SUCCESS (3/3) | Correctly ignored shape mismatch |
| MI2 | "Pick up the **red** bread" | Bread is brown | WRONG_OBJECT (3/3) | Ignored color constraint |
| MI3 | "Put bread in the **blue** plate" | Plate is white | WRONG_OBJECT (3/3) | Ignored color constraint |
| MI4 | "Pick up the **banana**" | No banana present | WRONG_OBJECT (3/3) | Defaulted to bread |

**Analysis**: The model exhibits semantic flexibility for geometry (MI1) but ignores color constraints entirely (MI2-MI4). This indicates the VLM component may not be fully integrated into action decisions, with the diffusion head defaulting to learned manipulation primitives regardless of scene mismatch.

<a id="failure-mode-analysis"></a>
### Failure Mode Analysis

#### Failure Distribution

| Failure Mode | Count | Percentage | Primary Cause |
|--------------|-------|------------|---------------|
| PARTIAL (grasp/release failure) | 36 | 50.7% | Gripper timing/force control |
| WRONG_OBJECT | 15 | 21.1% | Object discrimination or instruction violation |
| MISS | 10 | 14.1% | Position generalization failure |
| NO_ACTION (unexpected) | 6* | 8.5% | Cheese recognition failure |
| TIMEOUT | 4 | 5.6% | Motion planning or release failures |

*NO_ACTION on cheese in RQ2 indicates model does not recognize cheese as manipulable object

#### Root Cause Analysis

1. **Gripper Control (50.7% of failures)**: Most PARTIAL outcomes involve successful grasp but failed release. The gripper tends to remain closed over the plate, suggesting the diffusion model's action head learned suboptimal gripper trajectories.

2. **Object Bias (cheese: 0% success)**: The model exhibits systematic failure on cheese manipulation across all RQs. Training data likely underrepresented cheese demonstrations, causing the VLM to not associate "cheese" with actionable regions.

3. **Spatial Asymmetry**: Left-side source positions (zones 1, 7) show degraded performance while right-side destinations (zone 3) fail. This may reflect camera placement bias or asymmetric training data distribution.

4. **Semantic Grounding Gap**: Misleading instruction tests (RQ9) reveal the action head operates semi-independently from VLM understanding—correct object recognition doesn't prevent incorrect actions.

<a id="key-findings"></a>
### Key Findings

#### Strengths

| Capability | Evidence | Success Rate |
|------------|----------|--------------|
| **Destination Robustness** | RQ5: Plate position variations handled well | 75.0% |
| **Language Invariance** | RQ8: Paraphrased prompts correctly interpreted | 77.8% |
| **Task Completion Detection** | RQ3: Recognizes when bread is already in plate | 66.7% (bread) |
| **bread_1 Manipulation** | Primary training object | 37.7% |

#### Limitations

| Limitation | Evidence | Impact |
|------------|----------|--------|
| **Cheese Recognition Failure** | 0% success across all cheese trials | Cannot generalize to undertrained objects |
| **Distractor Confusion** | RQ6: 0% success, 100% PARTIAL | Struggles with similar objects nearby |
| **Spatial Asymmetry** | Left zones fail, right zones succeed | Limited workspace coverage |
| **Gripper Control** | 50.7% of failures are PARTIAL | Release timing unreliable |
| **Semantic-Action Gap** | RQ9: Ignores color/existence constraints | VLM understanding doesn't always affect actions |

#### Comparison to Claimed Achievements

| Claim | Evaluation Result | Assessment |
|-------|-------------------|------------|
| **85%+ Task Success Rate** | 31.1% overall, 37.7% bread_1 | ❌ Not achieved in structured evaluation |
| **Compositional Generalization** | Cheese: 0%, bread_2: 33.3% | ⚠️ Limited to bread_1 |
| **Language Conditioning** | 77.8% prompt variation success | ✅ Demonstrated |
| **80% Demo Reduction via MimicGen** | N/A (training efficiency metric) | ⚠️ Not directly validated by eval |

#### Implications for GR00T N1.5 Architecture

1. **Eagle VLM Backbone**: The vision-language model correctly processes scene semantics (evidenced by MI1 geometry handling) but this understanding is not always propagated to action decisions.

2. **Diffusion Transformer Action Head**: Tends to default to learned primitives (bread manipulation) regardless of instruction specifics, suggesting the conditioning mechanism may require additional fine-tuning.

3. **Dual-Camera System**: Spatial asymmetry patterns suggest potential camera coverage or calibration issues affecting peripheral workspace regions.

4. **Training Data Balance**: Single-object bias strongly impacts generalization. MimicGen augmentation helped with position variation but did not resolve object category imbalance.

---

<a id="10-troubleshooting"></a>
