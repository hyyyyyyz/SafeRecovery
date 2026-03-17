#!/bin/bash
source /home/hurricane/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym
cd /home/hurricane/RL/CSGLoco
export LD_LIBRARY_PATH=/home/hurricane/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH

echo "=========================================="
echo "Phase 2 Orchestrator started: $(date)"
echo "=========================================="

# Wait for phase 1 orchestrator to finish
echo "Waiting for phase 1 orchestrator..."
while tmux has-session -t orchestrator 2>/dev/null; do
    sleep 60
done
echo "Phase 1 done at $(date)"

# Step 1: Fair 3-way eval (CaT termination disabled for all)
echo ""
echo "Step 1: Fair 3-way evaluation..."
python legged_gym/legged_gym/scripts/eval_saferecovery.py 2>&1
echo "Fair eval done at $(date)"

# Step 2: Recovery stress test + perturbation sweep
echo ""
echo "Step 2: Stress test + perturbation sweep..."
python legged_gym/legged_gym/scripts/eval_stress.py 2>&1
echo "Stress test done at $(date)"

# Step 3: Multi-seed evaluation
echo ""
echo "Step 3: Multi-seed evaluation..."
python legged_gym/legged_gym/scripts/eval_multiseed.py 2>&1
echo "Multi-seed eval done at $(date)"

# Step 4: ANYmal-C training
echo ""
echo "Step 4: ANYmal-C training..."
bash /home/hurricane/RL/CSGLoco/train_anymal.sh 2>&1
echo "ANYmal training done at $(date)"

echo ""
echo "=========================================="
echo "Phase 2 ALL DONE: $(date)"
echo "=========================================="
