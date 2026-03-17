#!/bin/bash
source /home/hurricane/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym
cd /home/hurricane/RL/CSGLoco
export LD_LIBRARY_PATH=/home/hurricane/miniconda3/envs/isaacgym/lib:\$LD_LIBRARY_PATH

echo "Waiting for sr_train to finish..."
while tmux has-session -t sr_train 2>/dev/null; do
    sleep 30
done

echo "Training done. Running 3-way evaluation at \$(date)..."
python legged_gym/legged_gym/scripts/eval_saferecovery.py 2>&1
echo "Evaluation complete at \$(date)"
