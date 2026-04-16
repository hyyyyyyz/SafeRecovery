"""Evaluate the scripted recovery controller on SafeRecovery A1 across multiple seeds."""
import os, json, sys, gc, math

import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import set_seed

sys.path.insert(0, os.path.dirname(__file__))
from scripted_controller import ScriptedRecoveryController

OUTDIR = "/home/hurricane/RL/CSGLoco/safe_recovery_eval_v2"
os.makedirs(OUTDIR, exist_ok=True)
SEEDS = [1, 2, 3]


def evaluate_scripted(seed, num_eval_steps=10000, num_envs=128, force_range=None, fallen_start=False):
    set_seed(seed)
    task_name = "safe_recovery_a1"
    argv_backup = sys.argv
    sys.argv = ["eval", f"--task={task_name}", f"--num_envs={num_envs}", f"--seed={seed}", "--headless"]
    args = get_args()
    sys.argv = argv_backup

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = num_envs
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.safety.enable_constraint_termination = False

    if force_range is not None:
        env_cfg.perturbation.enabled = force_range[1] > 0
        env_cfg.perturbation.force_range = force_range
    else:
        env_cfg.perturbation.enabled = True
        env_cfg.perturbation.force_range = [80, 150]
    env_cfg.perturbation.interval_range = [2.0, 5.0]

    if fallen_start:
        fs = env_cfg.fallen_start if hasattr(env_cfg, "fallen_start") else type("FS", (), {})()
        fs.enabled = True
        fs.fraction = 1.0
        fs.roll_range = [-2.5, 2.5]
        fs.pitch_range = [-1.5, 1.5]
        fs.height_range = [0.08, 0.15]
        env_cfg.fallen_start = fs

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    controller = ScriptedRecoveryController(num_envs, env.device)
    env.safety_logger.reset()

    for _ in range(num_eval_steps):
        actions = controller.get_actions(obs)
        obs, _, _, dones, _ = env.step(actions)
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            controller.reset(done_ids)

    summary = env.get_safety_summary()
    dt = env.dt
    total_sim_time = num_eval_steps * dt
    summary["safety/violations_per_sec"] = summary["safety/total_violations"] / (total_sim_time * num_envs) if total_sim_time > 0 else 0
    summary["recovery/falls_per_min"] = summary["recovery/fall_count"] / (total_sim_time * num_envs / 60.0) if total_sim_time > 0 else 0
    summary["meta/task"] = task_name
    summary["meta/controller"] = "scripted"
    summary["meta/seed"] = seed
    summary["meta/num_eval_steps"] = num_eval_steps
    summary["meta/num_envs"] = num_envs
    summary["meta/fallen_start"] = fallen_start

    env.gym.destroy_sim(env.sim)
    del env, obs
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def save_result(result, filename):
    path = os.path.join(OUTDIR, filename)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Saved: {path}")


def mean_std(xs):
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def aggregate(results):
    agg = {}
    keys = set()
    for r in results:
        keys.update(k for k, v in r.items() if isinstance(v, (int, float)))
    for k in sorted(keys):
        vals = [float(r[k]) for r in results if k in r]
        if vals:
            m, s = mean_std(vals)
            agg[k] = m
            agg[k + "_std"] = s
    agg["meta/controller"] = "scripted"
    agg["meta/seeds"] = SEEDS
    agg["meta/aggregate"] = True
    return agg


if __name__ == "__main__":
    print("=" * 60)
    print("Scripted Recovery Controller Evaluation (3 seeds)")
    print("=" * 60)

    standard = []
    print("\n[Exp 1] Standard eval: 10k steps, 80-150N perturbation")
    for seed in SEEDS:
        result = evaluate_scripted(seed=seed, num_eval_steps=10000, num_envs=128, force_range=[80, 150], fallen_start=False)
        save_result(result, f"scripted_seed{seed}_standard.json")
        standard.append(result)
    save_result(aggregate(standard), "scripted_standard.json")

    fallen = []
    print("\n[Exp 2] Fallen-start eval: 5k steps, 100% fallen, no perturbation")
    for seed in SEEDS:
        result = evaluate_scripted(seed=seed, num_eval_steps=5000, num_envs=128, force_range=[0, 0], fallen_start=True)
        save_result(result, f"scripted_seed{seed}_fallen_start.json")
        fallen.append(result)
    save_result(aggregate(fallen), "scripted_fallen_start.json")

    print("\nDone!")
