#!/bin/bash
set -e
source /home/hurricane/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym
cd /home/hurricane/RL/CSGLoco
export LD_LIBRARY_PATH=/home/hurricane/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH

echo "Multi-seed training (3 seeds) started: $(date)"

# We already have seed 0 (original) and seed 1 (from earlier batch)
# Need seeds 2, 3 for Vanilla; 1, 2, 3 for CaT; 1, 2, 3 for Recovery

for seed in 2 3; do
    echo ">>> Vanilla PPO seed=$seed ($(date))"
    python legged_gym/legged_gym/scripts/train.py --task=safe_recovery_a1 --max_iterations=1500 --headless --seed=$seed 2>&1 | tail -3
    echo ">>> Done Vanilla seed=$seed"
done

for seed in 1 2 3; do
    echo ">>> CaT PPO seed=$seed ($(date))"
    python legged_gym/legged_gym/scripts/train.py --task=safe_recovery_a1_cat --max_iterations=1500 --headless --seed=$seed 2>&1 | tail -3
    echo ">>> Done CaT seed=$seed"
done

for seed in 1 2 3; do
    echo ">>> Recovery PPO seed=$seed ($(date))"
    python legged_gym/legged_gym/scripts/train.py --task=safe_recovery_a1_fallen --max_iterations=2000 --headless --seed=$seed 2>&1 | tail -3
    echo ">>> Done Recovery seed=$seed"
done

echo ""
echo ">>> ANYmal-C training..."
bash /home/hurricane/RL/CSGLoco/train_anymal.sh 2>&1

echo ""
echo ">>> Multi-seed evaluation..."
python legged_gym/legged_gym/scripts/eval_multiseed.py 2>&1

echo "ALL DONE: $(date)"
