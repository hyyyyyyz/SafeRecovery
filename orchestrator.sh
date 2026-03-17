#!/bin/bash
source /home/hurricane/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym
cd /home/hurricane/RL/CSGLoco
export LD_LIBRARY_PATH=/home/hurricane/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH

echo "=========================================="
echo "Orchestrator started: $(date)"
echo "=========================================="

# Step 1: Wait for sr_train (fallen-start seed 0) to finish
echo "Step 1: Waiting for sr_train to complete..."
while tmux has-session -t sr_train 2>/dev/null; do
    sleep 30
done
echo "sr_train done at $(date)"

# Step 2: Run 3-way eval (Vanilla vs CaT vs Recovery)
echo ""
echo "Step 2: Running 3-way evaluation..."
python legged_gym/legged_gym/scripts/eval_saferecovery.py 2>&1
echo "3-way eval done at $(date)"

# Step 3: Run multi-seed training batch
echo ""
echo "Step 3: Starting multi-seed training batch..."
bash /home/hurricane/RL/CSGLoco/batch_train.sh 2>&1

# Step 4: Run multi-seed evaluation
echo ""
echo "Step 4: Running multi-seed evaluation..."
python legged_gym/legged_gym/scripts/eval_multiseed.py 2>&1
echo "Multi-seed eval done at $(date)"

echo ""
echo "=========================================="
echo "ALL DONE: $(date)"
echo "=========================================="
