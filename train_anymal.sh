#!/bin/bash
set -e
source /home/hurricane/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym
cd /home/hurricane/RL/CSGLoco
export LD_LIBRARY_PATH=/home/hurricane/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH

echo "=========================================="
echo "ANYmal-C SafeRecovery training: $(date)"
echo "=========================================="

echo ">>> ANYmal-C Vanilla PPO"
python legged_gym/legged_gym/scripts/train.py --task=safe_recovery_anymal --max_iterations=1500 --headless 2>&1 | tail -5
echo ">>> Done Vanilla ANYmal"

echo ""
echo ">>> ANYmal-C CaT PPO"
python legged_gym/legged_gym/scripts/train.py --task=safe_recovery_anymal_cat --max_iterations=1500 --headless 2>&1 | tail -5
echo ">>> Done CaT ANYmal"

echo ""
echo ">>> ANYmal-C Recovery (fallen-start) PPO"
python legged_gym/legged_gym/scripts/train.py --task=safe_recovery_anymal_fallen --max_iterations=2000 --headless 2>&1 | tail -5
echo ">>> Done Recovery ANYmal"

echo ""
echo "=========================================="
echo "ANYmal-C training complete: $(date)"
echo "=========================================="
