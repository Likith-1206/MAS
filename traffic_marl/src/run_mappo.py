"""Train and evaluate MAPPO on the NYC traffic-signal map.

Example:
    python src/run_mappo.py --episodes 1 --seed 0 --skip-eval
    python src/run_mappo.py --episodes 100 --seed 0
"""

import argparse
import datetime
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from env_map import SumoMapEnv
import scenario_gen
from mappo import MAPPO


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_DIR = os.path.join(ROOT, "maps", "nyc16")

EVAL_SEEDS = [9001, 9002, 9003]


def train_seed(seed, episode):
    return 1000 * seed + episode


def check_seed_split(seed, episodes):
    used = {
        train_seed(seed, ep) for ep in range(episodes)
    }

    clash = used.intersection(EVAL_SEEDS)

    if clash:
        raise ValueError(
            f"Training seeds overlap evaluation seeds: {sorted(clash)}"
        )


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def evaluate(env, agent, method, training_seed):
    """Evaluate the trained policy on fixed test seeds."""

    rows = []
    trace_saved = False

    for scenario in scenario_gen.PATTERNS:
        for seed in EVAL_SEEDS:

            obs = env.reset(scenario=scenario, seed=seed)
            done = False
            trace = []

            while not done:
                actions = agent.act(
                    obs,
                    explore=False,
                    env=env,
                )

                obs, rewards, done, info = env.step(actions)

                if scenario == "rush_hour" and seed == EVAL_SEEDS[0]:
                    trace.append({
                        "time": int(info["time"]),
                        "queue": info["queue"],
                        "phase": info["phase"],
                    })

            metrics = env.metrics()

            rows.append({
                "method": method,
                "train_seed": training_seed,
                "scenario": scenario,
                "eval_seed": seed,
                **{
                    key: float(value)
                    for key, value in metrics.items()
                },
            })

            if trace and not trace_saved:
                trace_path = os.path.join(
                    ROOT,
                    "traces",
                    f"{method}_s{training_seed}.json",
                )

                with open(trace_path, "w") as file:
                    json.dump({
                        "method": method,
                        "scenario": scenario,
                        "eval_seed": seed,
                        "agents": env.agents,
                        "steps": trace,
                    }, file, indent=2)

                trace_saved = True

            print(
                f"[{method}] evaluation "
                f"{scenario:10s} seed={seed} "
                f"avg_wait={metrics['avg_wait_s']:.2f} "
                f"avg_queue={metrics['avg_queue']:.2f}",
                flush=True,
            )

    return rows

def train(episodes, seed, skip_eval=False, reward_mode="queue"):

    if not os.path.isdir(MAP_DIR):
        raise FileNotFoundError(
            f"Map directory not found: {MAP_DIR}"
        )

    if episodes <= 0:
        raise ValueError("episodes must be greater than zero.")

    check_seed_split(seed, episodes)

    for folder in ("results", "models", "traces"):
        os.makedirs(os.path.join(ROOT, folder), exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    method = "mappo" if reward_mode == "queue" else "mappo_pressure"
    tag = f"{method}_s{seed}"

    env = SumoMapEnv(MAP_DIR, reward_mode=reward_mode)

    agent = MAPPO(
        env,
        level=0,
        max_lanes=10,
    )

    curve = []
    start_time = time.time()

    try:
        # ---------------- Training ----------------

        for ep in range(episodes):

            scenario = scenario_gen.PATTERNS[
                ep % len(scenario_gen.PATTERNS)
            ]

            current_seed = train_seed(seed, ep)

            obs = env.reset(
                scenario=scenario,
                seed=current_seed,
            )

            done = False
            episode_return = 0.0
            steps = 0

            while not done:

                actions = agent.act(
                    obs,
                    explore=True,
                    env=env,
                )

                obs_next, rewards, done, info = env.step(actions)

                agent.store(rewards)

                episode_return += float(
                    np.mean(list(rewards.values()))
                )

                obs = obs_next
                steps += 1

            # Bootstrap the critic from the final observation.
            update_stats = agent.update(obs)

            metrics = env.metrics()

            row = {
                "episode": ep,
                "scenario": scenario,
                "seed": current_seed,
                "return": episode_return,
                "steps": steps,
                **{
                    key: float(value)
                    for key, value in metrics.items()
                },
                **update_stats,
            }

            curve.append(row)

            elapsed = time.time() - start_time

            print(
                f"[{method} seed={seed}] "
                f"episode {ep + 1}/{episodes} "
                f"{scenario:10s} "
                f"return={episode_return:8.2f} "
                f"avg_wait={metrics['avg_wait_s']:.2f} "
                f"avg_queue={metrics['avg_queue']:.2f} "
                f"entropy={update_stats['entropy']:.3f} "
                f"time={elapsed:.1f}s",
                flush=True,
            )

        # Save training curve and model.
        curve_path = os.path.join(
            ROOT, "results", f"curve_{tag}.csv"
        )

        pd.DataFrame(curve).to_csv(
            curve_path, index=False
        )

        model_path = os.path.join(
            ROOT, "models", f"{tag}.pt"
        )

        torch.save(agent.state_dict(), model_path)

        train_seconds = time.time() - start_time

        # ---------------- Evaluation ----------------

        if not skip_eval:

            rows = evaluate(
                env,
                agent,
                method,
                seed,
            )

            eval_path = os.path.join(
                ROOT, "results", f"eval_{tag}.csv"
            )

            pd.DataFrame(rows).to_csv(
                eval_path, index=False
            )

        # ---------------- Settings ----------------

        settings = {
            "method": method,
            "reward_mode": reward_mode,
            "seed": seed,
            "episodes": episodes,
            "communication_level": 0,
            "training_seed_formula": "1000*seed+episode",
            "evaluation_seeds": EVAL_SEEDS,
            "scenarios": scenario_gen.PATTERNS,
            "map": "maps/nyc16",
            "number_of_agents": agent.n_ag,
            **agent.hyperparams(),
            "git_commit": git_commit(),
            "train_seconds": round(train_seconds),
            "evaluation_skipped": skip_eval,
            "finished": datetime.datetime.now().isoformat(
                timespec="seconds"
            ),
        }

        settings_path = os.path.join(
            ROOT, "results", f"settings_{tag}.json"
        )

        with open(settings_path, "w") as file:
            json.dump(settings, file, indent=2)

        print("\nTraining completed.")
        print("Model:", model_path)
        print("Training results:", curve_path)

        if not skip_eval:
            print("Evaluation results:", eval_path)

    finally:
        env.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after training.",
    )

    # Select which reward function MAPPO should use.
    parser.add_argument(
        "--reward-mode",
        choices=["queue", "pressure_delta"],
        default="queue",
        help="Reward function: original queue penalty or pressure-aware queue reduction."
    )

    args = parser.parse_args()

    train(
        episodes=args.episodes,
        seed=args.seed,
        skip_eval=args.skip_eval,
        reward_mode=args.reward_mode,
    )
