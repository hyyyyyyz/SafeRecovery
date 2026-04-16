"""Quantitative threshold sweep for SafeRecovery Axis 4 metrics.

Varies one parameter at a time from default:
  - torque_limit: 20, [25], 33.5 Nm
  - selfright_stable_duration: 0.1, [0.3], 0.5 s
  - recovery_vel_error_threshold: 0.3, [0.5], 0.8 m/s

Runs Recovery PPO (3 seeds) under each non-default configuration.
"""
import os, json, sys, gc
from datetime import datetime

import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

LOG_BASE = "/home/hurricane/RL/CSGLoco/legged_gym/logs"
OUTDIR = "/home/hurricane/RL/CSGLoco/safe_recovery_eval_v2"
os.makedirs(OUTDIR, exist_ok=True)

RECOVERY_RUNS = {
    1: "Mar18_19-54-41_",
    2: "Mar18_20-44-07_",
    3: "Mar18_21-33-50_",
}

# Sweep configurations: (param_name, config_path, values_to_test)
# Default values: torque_limit=25, selfright=0.3, vel_error=0.5
SWEEPS = [
    ("torque_limit", "safety", "torque_limit", [20.0, 33.5]),
    ("selfright_hold", "fall_detection", "selfright_stable_duration", [0.1, 0.5]),
    ("vel_threshold", "fall_detection", "recovery_vel_error_threshold", [0.3, 0.8]),
]


def evaluate(load_run, seed, torque_limit=25.0, selfright_dur=0.3, vel_thresh=0.5,
             num_eval_steps=10000, num_envs=128):
    task_name = "safe_recovery_a1_fallen"
    argv_backup = sys.argv
    sys.argv = ["eval", "--task=%s" % task_name, "--num_envs=%d" % num_envs, "--headless"]
    args = get_args()
    sys.argv = argv_backup

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = num_envs
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.safety.enable_constraint_termination = False

    # Apply threshold overrides
    env_cfg.safety.torque_limit = torque_limit
    env_cfg.fall_detection.selfright_stable_duration = selfright_dur
    env_cfg.fall_detection.recovery_vel_error_threshold = vel_thresh

    env_cfg.perturbation.enabled = True
    env_cfg.perturbation.force_range = [80, 150]
    env_cfg.perturbation.interval_range = [2.0, 5.0]

    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = load_run
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=task_name, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    env.safety_logger.reset()
    for step in range(num_eval_steps):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

    summary = env.get_safety_summary()
    dt = env.dt
    total_sim_time = num_eval_steps * dt
    summary["safety/violations_per_sec"] = summary["safety/total_violations"] / (total_sim_time * num_envs) if total_sim_time > 0 else 0
    summary["recovery/falls_per_min"] = summary["recovery/fall_count"] / (total_sim_time * num_envs / 60.0) if total_sim_time > 0 else 0
    summary["meta/torque_limit"] = torque_limit
    summary["meta/selfright_dur"] = selfright_dur
    summary["meta/vel_thresh"] = vel_thresh
    summary["meta/seed"] = seed

    env.gym.destroy_sim(env.sim)
    del env, ppo_runner, policy, obs
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def save_result(result, filename):
    path = os.path.join(OUTDIR, filename)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Threshold Sweep for Recovery PPO")
    print("=" * 60)

    for param_label, cfg_section, cfg_field, values in SWEEPS:
        for val in values:
            for seed, load_run in RECOVERY_RUNS.items():
                print(f"\n[{param_label}={val}, seed={seed}]")

                kwargs = {"torque_limit": 25.0, "selfright_dur": 0.3, "vel_thresh": 0.5}
                if param_label == "torque_limit":
                    kwargs["torque_limit"] = val
                elif param_label == "selfright_hold":
                    kwargs["selfright_dur"] = val
                elif param_label == "vel_threshold":
                    kwargs["vel_thresh"] = val

                try:
                    result = evaluate(
                        load_run=load_run, seed=seed,
                        num_eval_steps=10000, num_envs=128,
                        **kwargs
                    )
                    fname = f"threshold_{param_label}_{val}_seed{seed}.json"
                    save_result(result, fname)
                    print(f"  Viol/s: {result['safety/violations_per_sec']:.4f}")
                    print(f"  Falls/min: {result['recovery/falls_per_min']:.2f}")
                    rec_rate = result.get("recovery/success_rate", 0)
                    print(f"  Recovery: {rec_rate*100:.1f}%")
                    pct = result.get("coupling/pct_attempts_with_violation", 0)
                    print(f"  Rec w/ haz: {pct*100:.1f}%")
                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue

    print("\n" + "=" * 60)
    print("Threshold sweep complete!")
