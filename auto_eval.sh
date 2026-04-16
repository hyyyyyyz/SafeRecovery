#!/bin/bash
# Auto-run evaluations after training completes

LOG_FILE="/home/hurricane/RL/CSGLoco/auto_eval.log"

echo "=== Auto Evaluation Triggered at $(date) ===" >> $LOG_FILE

source /home/hurricane/miniconda3/etc/profile.d/conda.sh
conda activate isaacgym
export LD_LIBRARY_PATH=/home/hurricane/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH
cd /home/hurricane/RL/CSGLoco

# Wait a bit for all training to fully complete
sleep 60

echo "Starting ANYmal multi-seed evaluation..." >> $LOG_FILE
python eval_anymal_multiseed.py >> $LOG_FILE 2>&1

echo "Starting Vanilla/CaT @2000 evaluation..." >> $LOG_FILE
python eval_vanilla_cat_2000.py >> $LOG_FILE 2>&1

echo "All evaluations completed at $(date)" >> $LOG_FILE

# Create completion flag
touch /home/hurricane/RL/CSGLoco/ALL_EVAL_COMPLETE
