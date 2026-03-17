"""Train Vanilla PPO and CaT PPO with 5 seeds each for statistical robustness."""
import subprocess
import sys
import os
import time

SEEDS = [1, 2, 3, 4, 5]
TASKS = ["safe_recovery_a1", "safe_recovery_a1_cat"]
MAX_ITER = 1500
BASE_CMD = (
    "source /home/hurricane/miniconda3/etc/profile.d/conda.sh && "
    "conda activate isaacgym && "
    "cd /home/hurricane/RL/CSGLoco && "
    "export LD_LIBRARY_PATH=/home/hurricane/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH && "
    "python legged_gym/legged_gym/scripts/train.py --task={task} --max_iterations={iters} --headless --seed={seed}"
)

for task in TASKS:
    for seed in SEEDS:
        cmd = BASE_CMD.format(task=task, iters=MAX_ITER, seed=seed)
        print(f"\n{'='*60}")
        print(f"Training: {task} seed={seed}")
        print(f"{'='*60}")
        ret = subprocess.run(["bash", "-lc", cmd], capture_output=False)
        if ret.returncode != 0:
            print(f"WARNING: {task} seed={seed} returned {ret.returncode}")
        print(f"Done: {task} seed={seed}")

print("\n\nAll multi-seed training complete!")
