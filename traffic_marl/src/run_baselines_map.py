"""Evaluate Fixed-Time and Max-Pressure on the NYC map."""

import os
import csv
import json
import argparse
import numpy as np
import torch

from env_map import SumoMapEnv
from controllers import FixedTime, MaxPressure
import scenario_gen


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_DIR = os.path.join(ROOT, "maps", "nyc16")

RESULTS_DIR = os.path.join(ROOT, "results")

EVAL_SEEDS = [9001, 9002, 9003]


def evaluate_controller(controller, env, scenarios):
    rows = []

    for scenario in scenarios:
        for seed in EVAL_SEEDS:
            print(
                f"[{controller.name}] "
                f"scenario={scenario}, seed={seed}"
            )

            obs = env.reset(scenario=scenario, seed=seed)
            controller.reset(env)

            done = False

            while not done:
                actions = controller.act(env, obs)
                obs, rewards, done, info = env.step(actions)

            metrics = env.metrics()

            row = {
                "controller": controller.name,
                "scenario": scenario,
                "seed": seed,
            }

            row.update(metrics)
            rows.append(row)

            print(
                f"  avg_wait={metrics['avg_wait_s']:.2f}s, "
                f"avg_queue={metrics['avg_queue']:.2f}, "
                f"throughput={metrics['throughput']}"
            )

    return rows


def save_results(rows, filename):
    path = os.path.join(RESULTS_DIR, filename)

    if not rows:
        raise RuntimeError("No evaluation results were generated.")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--controller",
        choices=["fixed_time", "max_pressure", "both"],
        default="both",
    )

    parser.add_argument("--green", type=int, default=30)

    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    env = SumoMapEnv(MAP_DIR)

    print("\nNYC MAP BASELINE EVALUATION")
    print("---------------------------")
    print("Map:", MAP_DIR)
    print("Number of agents:", len(env.agents))
    print("Maximum incoming lanes:", env.max_lanes)
    print("Maximum actions:", env.n_act)
    print("Scenarios:", scenario_gen.PATTERNS)
    print("Evaluation seeds:", EVAL_SEEDS)

    try:
        if args.controller in ("fixed_time", "both"):
            fixed = FixedTime(green_s=args.green)

            fixed_rows = evaluate_controller(
                fixed, env, scenario_gen.PATTERNS
            )

            save_results(
                fixed_rows,
                "fixed_time_nyc.csv"
            )

        if args.controller in ("max_pressure", "both"):
            max_pressure = MaxPressure()

            pressure_rows = evaluate_controller(
                max_pressure, env, scenario_gen.PATTERNS
            )

            save_results(
                pressure_rows,
                "max_pressure_nyc.csv"
            )

    finally:
        env.close()

    print("\nBaseline evaluation completed.")


if __name__ == "__main__":
    main()