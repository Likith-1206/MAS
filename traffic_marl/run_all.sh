#!/bin/bash
# Reproduces every result in results/. ~25 min on 1 CPU core. Edit EPISODES / seeds as needed.
cd "$(dirname "$0")"
EP=${EPISODES:-40}
python3 src/run_experiments.py baselines
for m in coop independent shared_c0 shared_c1 shared_c2; do python3 src/run_experiments.py train --method $m --episodes $EP --seed 0; done
for m in coop independent; do python3 src/run_experiments.py train --method $m --episodes $EP --seed 1; done  # 2nd seed (not run in the prototype delivery)
echo ALL_DONE
