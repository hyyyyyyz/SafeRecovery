# SafeRecovery

**A Unified Safety-Centric Benchmark for Quadruped Robot Locomotion and Fall Recovery**

SafeRecovery is a four-axis evaluation protocol that jointly measures locomotion quality, safety compliance, recovery capability, and the coupling between safety and recovery for quadruped robots.

## Overview

Current evaluation for quadruped locomotion usually measures safety and fall recovery separately, so it cannot show whether a policy can recover safely after failure. SafeRecovery addresses this gap by providing a unified evaluation framework.

## Four Evaluation Axes

1. **Locomotion**: Velocity tracking under nominal conditions
2. **Safety**: Constraint violations across multiple categories
3. **Recovery**: Fall detection, self-righting success, and return-to-task ability
4. **Coupling**: Safety violations conditioned on recovery state (key contribution)

## Key Features

- **Common-rollout evaluation**: Method-specific termination disabled during testing for fair comparison
- **Cross-morphology support**: Evaluated on Unitree A1 and ANYmal-C
- **Domain randomization**: Rough terrain evaluation with friction and noise variation
- **Comprehensive baselines**: Vanilla PPO, CaT, Recovery PPO, and scripted controller

## Repository Structure

- `legged_gym/`: Simulation environment based on Isaac Gym
- `safe_recovery_eval_v2/`: Evaluation results (JSON)
- `run_*.py`: Experiment scripts for training and evaluation
- `eval_*.py`: Evaluation scripts for specific experiments

## Main Results

| Method | Recovery Success | Falls/min | Violations/sec |
|--------|-----------------|-----------|----------------|
| Vanilla PPO | 0.07% | 8.07 | 0.21 |
| CaT | 3.08% | 5.75 | 0.16 |
| **Recovery PPO** | **57.71%** | **0.14** | 0.44 |

## Requirements

- NVIDIA GPU (RTX 3090 Ti or better recommended)
- Isaac Gym
- Python 3.8+
- PyTorch

## Citation

```bibtex
@inproceedings{huang2026saferecovery,
  title={SafeRecovery: A Unified Safety-Centric Benchmark for Quadruped Robot Locomotion and Fall Recovery},
  author={Huang, Yaozeng and Zhang, Ziyi and Jiang, Shuhao and Li, Jingjie and Zhou, Yanyun and Shi, Fei},
  booktitle={ICIC},
  year={2026}
}
```

## License

This project is released for research purposes.
