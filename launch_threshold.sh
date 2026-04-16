#!/bin/bash
cd /home/hurricane/RL/CSGLoco
source /home/hurricane/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym
export LD_LIBRARY_PATH=/home/hurricane/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH
sleep 600
python run_threshold_sweep.py 2>&1 | tee threshold_sweep.log
