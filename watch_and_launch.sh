#!/bin/bash
# Wait for sr_train tmux session to end, then launch batch training
source /home/hurricane/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym
cd /home/hurricane/RL/CSGLoco
export LD_LIBRARY_PATH=/home/hurricane/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH

echo "Waiting for sr_train session to complete..."
while tmux has-session -t sr_train 2>/dev/null; do
    sleep 30
done
echo "sr_train completed at $(date). Launching batch training..."
bash /home/hurricane/RL/CSGLoco/batch_train.sh
