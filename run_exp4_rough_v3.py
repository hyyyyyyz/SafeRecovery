"""Exp 4: A1 Rough Terrain Evaluation - Alternative Approach

Instead of using heightfield/trimesh which causes segfaults,
we use randomized obstacles on plane terrain with domain randomization.
This provides a form of "rough terrain" without the IsaacGym terrain issues.
"""
import os, json, sys, math
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

OUTDIR = "/home/hurricane/RL/CSGLoco/safe_recovery_eval_v2"
os.makedirs(OUTDIR, exist_ok=True)

# A1 seeds 1-3 on rough terrain
RUNS = {
    "Vanilla": {
        "task": "safe_recovery_a1",
        "seeds": {1: "Mar18_16-16-27_", 2: "Mar18_16-53-02_", 3: "Mar18_17-29-18_"},
    },
    "CaT": {
        "task": "safe_recovery_a1_cat",
        "seeds": {1: "Mar18_18-05-33_", 2: "Mar18_18-42-10_", 3: "Mar18_19-18-36_"},
    },
    "Recovery": {
        "task": "safe_recovery_a1_fallen",
        "seeds": {1: "Mar18_19-54-41_", 2: "Mar18_20-44-07_", 3: "Mar18_21-33-50_"},
    },
}

def save_result(result, filename):
    path = os.path.join(OUTDIR, filename)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("    Saved: %s" % filename)

def evaluate_rough_alternative(task_name, load_run, num_eval_steps=10000, num_envs=128,
                                force_range=None, fallen_start=False, native_termination=False):
    """Evaluate with rough terrain-like conditions using domain randomization."""
    argv_backup = sys.argv
    sys.argv = ["eval", "--task=%s" % task_name, "--num_envs=%d" % num_envs, "--headless"]
    args = get_args()
    sys.argv = argv_backup

    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = num_envs

    # Keep plane terrain but add significant domain randomization to simulate rough conditions
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.measure_heights = False  # Plane doesn't support height measurement

    # Aggressive domain randomization to simulate rough terrain effects
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.friction_range = [0.2, 1.5]  # Wider range for rough terrain
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.added_mass_range = [-1.0, 2.0]  # Simulate debris/uneven loading

    # Add noise to observations to simulate rough terrain sensor noise
    env_cfg.noise.add_noise = True
    env_cfg.noise.noise_scales.dof_pos = 0.05
    env_cfg.noise.noise_scales.dof_vel = 0.15
    env_cfg.noise.noise_scales.ang_vel = 0.15
    env_cfg.noise.noise_scales.lin_vel = 0.1
    env_cfg.noise.noise_scales.gravity = 0.05
    env_cfg.noise.noise_scales.imu = 0.15

    # Stronger perturbations
    if force_range is not None:
        env_cfg.perturbation.enabled = force_range[1] > 0
        env_cfg.perturbation.force_range = force_range
    else:
        env_cfg.perturbation.enabled = True
        env_cfg.perturbation.force_range = [100, 250]  # Stronger than standard 80-150
    env_cfg.perturbation.interval_range = [1.5, 4.0]  # More frequent

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

    print("  Creating rough-like env (plane + aggressive domain rand)...")
    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = load_run
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=task_name, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    env.safety_logger.reset()
    completed_episodes = 0
    episode_lengths = []
    cur_ep_len = torch.zeros(num_envs, dtype=torch.int64, device=env.device)

    print("  Running eval...")
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
    summary["meta/num_eval_steps"] = num_eval_steps
    summary["meta/num_envs"] = num_envs
    summary["meta/fallen_start"] = fallen_start
    summary["meta/native_termination"] = native_termination
    summary["meta/terrain_type"] = "rough_like_domain_rand"
    summary["meta/timestamp"] = datetime.now().isoformat()

    env.gym.destroy_sim(env.sim)
    return summary

def run_rough_eval():
    print("=" * 70)
    print("EXP 4: A1 Rough Terrain Evaluation (Seeds 1-3) - DOMAIN RAND VERSION")
    print("=" * 70)
    print("Note: Using plane terrain + aggressive domain randomization as")
    print("      rough terrain alternative due to IsaacGym heightfield issues.")

    all_results = {}
    for method, info in RUNS.items():
        print("\n--- %s ---" % method)
        seed_results = []
        for seed_id, run_name in sorted(info["seeds"].items()):
            print("  Seed %d (%s)..." % (seed_id, run_name))
            try:
                r = evaluate_rough_alternative(info["task"], run_name, num_eval_steps=10000, num_envs=128,
                                               force_range=[100, 250])
                save_result(r, "%s_seed%d_rough.json" % (method.lower(), seed_id))
                seed_results.append(r)
            except Exception as e:
                print("  ERROR: %s" % e)
                import traceback; traceback.print_exc()
        all_results[method] = seed_results

    KEY_METRICS = [
        "safety/violations_per_sec",
        "recovery/success_rate",
        "recovery/falls_per_min",
        "coupling/viol_per_active_rec_sec",
    ]
    print("\n--- Summary (Rough Terrain - Domain Rand) ---")
    for method, seeds in all_results.items():
        print("\n%s (n=%d):" % (method, len(seeds)))
        for k in KEY_METRICS:
            vals = [s.get(k, float("nan")) for s in seeds]
            vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
            if vals:
                m = sum(vals) / len(vals)
                s = math.sqrt(sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1))
                print("  %s: %.4f +/- %.4f" % (k.split("/")[-1], m, s))

    return all_results

if __name__ == "__main__":
    print("Started: %s" % datetime.now().isoformat())
    results = run_rough_eval()
    print("\nALL DONE: %s" % datetime.now().isoformat())
