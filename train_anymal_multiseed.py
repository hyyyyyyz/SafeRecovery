"""Train ANYmal-C with multiple seeds for statistical robustness."""
import subprocess
import sys
import os
import time

SEEDS = [2, 3, 4, 5, 6]  # Seed 1 already exists
TASKS = ["safe_recovery_anymal", "safe_recovery_anymal_cat", "safe_recovery_anymal_fallen"]
MAX_ITER = 1500  # Vanilla and CaT
MAX_ITER_FALLEN = 2000  # Recovery

BASE_CMD = (
    "source /home/hurricane/miniconda3/etc/profile.d/conda.sh && "
    "conda activate isaacgym && "
    "cd /home/hurricane/RL/CSGLoco && "
    "export LD_LIBRARY_PATH=/home/hurricane/miniconda3/envs/isaacgym/lib:$LD_LIBRARY_PATH && "
    "python legged_gym/legged_gym/scripts/train.py --task={task} --max_iterations={iters} --headless --seed={seed}"
)

for task in TASKS:
    for seed in SEEDS:
        iters = MAX_ITER_FALLEN if "fallen" in task else MAX_ITER
        cmd = BASE_CMD.format(task=task, iters=iters, seed=seed)
        print(f"\n{'='*60}")
        print(f"Training: {task} seed={seed}, iters={iters}")
        print(f"{'='*60}")
        ret = subprocess.run(["bash", "-lc", cmd], capture_output=False)
        if ret.returncode != 0:
            print(f"WARNING: {task} seed={seed} returned {ret.returncode}")
        print(f"Done: {task} seed={seed}")

print("\n\nAll ANYmal multi-seed training complete!")
