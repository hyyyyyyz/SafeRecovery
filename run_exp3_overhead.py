"""Exp 3: Protocol Overhead Profiling"""
import os, json, sys, time
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

OUTDIR = "/home/hurricane/RL/CSGLoco/safe_recovery_eval_v2"
os.makedirs(OUTDIR, exist_ok=True)

def profile_overhead(task_name, load_run, num_eval_steps=1000, num_envs=128):
    """Profile evaluation overhead: timing, memory, throughput."""
    argv_backup = sys.argv
    sys.argv = ["eval", "--task=%s" % task_name, "--num_envs=%d" % num_envs, "--headless"]
    args = get_args()
    sys.argv = argv_backup

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = num_envs
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.perturbation.enabled = True
    env_cfg.perturbation.force_range = [80, 150]
    env_cfg.perturbation.interval_range = [2.0, 5.0]
    env_cfg.safety.enable_constraint_termination = False

    # Timing: environment creation
    t0 = time.time()
    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
    env_create_time = time.time() - t0

    obs = env.get_observations()

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = load_run
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=task_name, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # Warmup
    for _ in range(10):
        actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())

    # Timing: rollout
    env.safety_logger.reset()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()

    for step in range(num_eval_steps):
        actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    rollout_time = time.time() - t0

    # Memory profiling
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        mem_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
    else:
        mem_allocated = 0
        mem_reserved = 0

    # Get safety summary
    summary = env.get_safety_summary()

    env.gym.destroy_sim(env.sim)

    # Calculate metrics
    dt = env.dt
    total_sim_time = num_eval_steps * dt
    steps_per_sec = num_eval_steps / rollout_time
    env_steps_per_sec = (num_eval_steps * num_envs) / rollout_time
    realtime_factor = total_sim_time / rollout_time

    profile = {
        "profile/env_create_time_sec": env_create_time,
        "profile/rollout_time_sec": rollout_time,
        "profile/num_eval_steps": num_eval_steps,
        "profile/num_envs": num_envs,
        "profile/steps_per_sec": steps_per_sec,
        "profile/env_steps_per_sec": env_steps_per_sec,
        "profile/realtime_factor": realtime_factor,
        "profile/mem_allocated_mb": mem_allocated,
        "profile/mem_reserved_mb": mem_reserved,
        "profile/sim_time_sec": total_sim_time,
        "safety/total_violations": summary.get("safety/total_violations", 0),
        "safety/violations_per_sec": summary.get("safety/total_violations", 0) / total_sim_time if total_sim_time > 0 else 0,
        "meta/task": task_name,
        "meta/load_run": str(load_run),
        "meta/timestamp": datetime.now().isoformat(),
    }

    return profile

def run_profiling():
    print("=" * 70)
    print("EXP 3: Protocol Overhead Profiling")
    print("=" * 70)

    configs = [
        ("safe_recovery_a1", "Mar18_16-16-27_", "Vanilla"),
        ("safe_recovery_a1_cat", "Mar18_18-05-33_", "CaT"),
        ("safe_recovery_a1_fallen", "Mar18_19-54-41_", "Recovery"),
    ]

    results = []
    for task, run, name in configs:
        print("\n--- Profiling %s ---" % name)
        try:
            r = profile_overhead(task, run, num_eval_steps=1000, num_envs=128)
            results.append({"name": name, "result": r})
            print("  Steps/sec: %.1f" % r["profile/steps_per_sec"])
            print("  Env steps/sec: %.1f" % r["profile/env_steps_per_sec"])
            print("  Realtime factor: %.1fx" % r["profile/realtime_factor"])
            print("  Memory: %.1f MB allocated" % r["profile/mem_allocated_mb"])
        except Exception as e:
            print("  ERROR: %s" % e)
            import traceback; traceback.print_exc()

    # Save results
    path = os.path.join(OUTDIR, "overhead_profile.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved: %s" % path)

    # Print comparison table
    print("\n" + "=" * 70)
    print("OVERHEAD COMPARISON")
    print("=" * 70)
    print("%-12s %12s %12s %12s %12s" % ("Method", "Steps/sec", "EnvSteps/sec", "Realtime", "Mem(MB)"))
    print("-" * 70)
    for r in results:
        print("%-12s %12.1f %12.1f %12.1fx %12.1f" % (
            r["name"],
            r["result"]["profile/steps_per_sec"],
            r["result"]["profile/env_steps_per_sec"],
            r["result"]["profile/realtime_factor"],
            r["result"]["profile/mem_allocated_mb"],
        ))

    return results

if __name__ == "__main__":
    print("Started: %s" % datetime.now().isoformat())
    results = run_profiling()
    print("\nALL DONE: %s" % datetime.now().isoformat())
