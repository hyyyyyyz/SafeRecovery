"""Evaluate Vanilla and CaT @2000 iterations."""
import os, json, sys, math
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

OUTDIR = "/home/hurricane/RL/CSGLoco/safe_recovery_eval_v2"
os.makedirs(OUTDIR, exist_ok=True)

def save_result(result, filename):
    path = os.path.join(OUTDIR, filename)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("    Saved: %s" % filename)

def evaluate(task_name, load_run, checkpoint_iter=2000, num_eval_steps=10000, num_envs=128,
             force_range=None, fallen_start=False, native_termination=False):
    """Standard evaluation."""
    argv_backup = sys.argv
    sys.argv = ["eval", "--task=%s" % task_name, "--num_envs=%d" % num_envs, "--headless"]
    args = get_args()
    sys.argv = argv_backup

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = num_envs
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False

    if force_range is not None:
        env_cfg.perturbation.enabled = force_range[1] > 0
        env_cfg.perturbation.force_range = force_range
    else:
        env_cfg.perturbation.enabled = True
        env_cfg.perturbation.force_range = [80, 150]
    env_cfg.perturbation.interval_range = [2.0, 5.0]

    if native_termination:
        env_cfg.safety.enable_constraint_termination = True
    else:
        env_cfg.safety.enable_constraint_termination = False

    if fallen_start:
        if not hasattr(env_cfg, "fallen_start"):
            class FallenStart:
                enabled = True
                fraction = 1.0
                roll_range = [-2.5, 2.5]
                pitch_range = [-1.5, 1.5]
                height_range = [0.08, 0.15]
            env_cfg.fallen_start = FallenStart()
        else:
            env_cfg.fallen_start.enabled = True
            env_cfg.fallen_start.fraction = 1.0

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = load_run
    train_cfg.runner.checkpoint = checkpoint_iter
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=task_name, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    env.safety_logger.reset()
    completed_episodes = 0
    episode_lengths = []
    cur_ep_len = torch.zeros(num_envs, dtype=torch.int64, device=env.device)

    for step in range(num_eval_steps):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        cur_ep_len += 1
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            for idx in done_ids:
                episode_lengths.append(cur_ep_len[idx].item())
            cur_ep_len[done_ids] = 0
            completed_episodes += len(done_ids)

    summary = env.get_safety_summary()
    dt = env.dt
    total_sim_time = num_eval_steps * dt

    summary["locomotion/completed_episodes"] = completed_episodes
    summary["locomotion/mean_episode_length_steps"] = (
        sum(episode_lengths) / len(episode_lengths) if episode_lengths else float("nan")
    )
    summary["safety/violations_per_sec"] = summary["safety/total_violations"] / (total_sim_time * num_envs) if total_sim_time > 0 else 0
    summary["recovery/falls_per_min"] = summary["recovery/fall_count"] / (total_sim_time * num_envs / 60.0) if total_sim_time > 0 else 0

    summary["meta/task"] = task_name
    summary["meta/load_run"] = str(load_run)
    summary["meta/checkpoint_iter"] = checkpoint_iter
    summary["meta/num_eval_steps"] = num_eval_steps
    summary["meta/num_envs"] = num_envs
    summary["meta/fallen_start"] = fallen_start
    summary["meta/native_termination"] = native_termination
    summary["meta/timestamp"] = datetime.now().isoformat()

    env.gym.destroy_sim(env.sim)
    return summary

def run_vanilla_cat_2000_eval():
    """Evaluate Vanilla and CaT @2000 iterations."""
    print("=" * 70)
    print("Vanilla/CaT @2000 Iterations Evaluation")
    print("=" * 70)

    tasks = {
        "safe_recovery_a1": "vanilla_2000",
        "safe_recovery_a1_cat": "cat_2000",
    }

    for task, method_name in tasks.items():
        print(f"\n--- {method_name} ---")
        log_dir = f"/home/hurricane/RL/CSGLoco/legged_gym/logs/{task}"

        if not os.path.exists(log_dir):
            print(f"  Log dir not found: {log_dir}")
            continue

        runs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
        runs.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)))

        # Take the most recent runs (up to 6 seeds)
        seed_id = 1
        for run in runs[-6:]:
            print(f"  Evaluating {run} (as seed {seed_id})...")
            try:
                r = evaluate(task, run, checkpoint_iter=2000, num_eval_steps=10000, num_envs=128,
                           force_range=[80, 150])
                save_result(r, f"{method_name}_seed{seed_id}_standard.json")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()
            seed_id += 1

    print("\n" + "=" * 70)
    print("Vanilla/CaT @2000 Evaluation Complete")
    print("=" * 70)

if __name__ == "__main__":
    print("Started: %s" % datetime.now().isoformat())
    run_vanilla_cat_2000_eval()
    print("\nALL DONE: %s" % datetime.now().isoformat())
