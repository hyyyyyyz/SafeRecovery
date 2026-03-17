#!/bin/bash
set -e
source /home/hurricane/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym
cd /home/hurricane/RL/CSGLoco
export LD_LIBRARY_PATH=/home/hurricane/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH

echo "=========================================="
echo "Multi-seed training batch started: $(date)"
echo "=========================================="

# Vanilla PPO: 5 seeds, 1500 iterations each
for seed in 1 2 3 4 5; do
    echo ""
    echo ">>> Vanilla PPO seed=$seed ($(date))"
    python legged_gym/legged_gym/scripts/train.py --task=safe_recovery_a1 --max_iterations=1500 --headless --seed=$seed 2>&1 | tail -5
    echo ">>> Done Vanilla seed=$seed"
done

# CaT PPO: 5 seeds, 1500 iterations each
for seed in 1 2 3 4 5; do
    echo ""
    echo ">>> CaT PPO seed=$seed ($(date))"
    python legged_gym/legged_gym/scripts/train.py --task=safe_recovery_a1_cat --max_iterations=1500 --headless --seed=$seed 2>&1 | tail -5
    echo ">>> Done CaT seed=$seed"
done

# Fallen-start Recovery: 4 more seeds (seed 0 already running), 2000 iterations each
for seed in 1 2 3 4 5; do
    echo ""
    echo ">>> Recovery (fallen-start) seed=$seed ($(date))"
    python legged_gym/legged_gym/scripts/train.py --task=safe_recovery_a1_fallen --max_iterations=2000 --headless --seed=$seed 2>&1 | tail -5
    echo ">>> Done Recovery seed=$seed"
done

echo ""
echo "=========================================="
echo "All multi-seed training complete: $(date)"
echo "=========================================="
