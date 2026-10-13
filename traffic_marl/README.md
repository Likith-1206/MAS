# Multi-Agent Traffic Signal Coordination (MAS project) - working prototype

A complete, runnable first version of the project in the proposal: a 3x3 signalised grid in SUMO (libsumo), one agent per
intersection, three approaches compared (fixed-time, independent adaptive DQN, cooperative DQN with neighbour information),
plus a Max-Pressure reference baseline, an RQ2 communication ablation, statistics, plots and a Streamlit dashboard.

**Status:** every module runs end-to-end and produced the results in `results/`. Those results come from a *short* training
budget (about 40 episodes, 1-2 seeds, 1 CPU core) and are **preliminary**. The remaining project time is for longer training, more
seeds, tuning, extensions and the write-up (see `docs/02_timeline_and_work_split.md`).

## Quick start
```bash
pip install -r requirements.txt
bash run_all.sh                 # baselines + all training runs (EPISODES=40 by default, the value used for the delivered results; ~25 min on one core)
python src/make_plots.py        # tables, stat tests, plots -> results/
streamlit run src/dashboard.py  # network replay + results
```
Single runs: `python src/run_experiments.py train --method coop --episodes 100 --seed 0` (methods: `independent`, `coop`, `shared_c0`, `shared_c1`, `shared_c2`).

## Layout
| Path | Content |
|---|---|
| `src/scenarios.py` | 3x3 grid generation (`netgenerate`), 4 demand scenarios (uniform, asymmetric, rush_hour, corridor), Poisson-like arrivals |
| `src/env.py` | Multi-agent environment: obs / action / reward, yellow handling, metrics, optional neighbour information |
| `src/controllers.py` | Baseline 1 fixed-time (30 s per phase); reference Max-Pressure |
| `src/agents.py` | Double-DQN in PyTorch: `IndependentDQN` (one net per intersection) and `SharedDQN` (parameter sharing), reward mixing |
| `src/run_experiments.py` | Training + evaluation (3 eval seeds x 4 scenarios per method) |
| `src/make_plots.py` | Summary tables, paired statistical tests, bar plots, ablation, learning curves |
| `src/dashboard.py` | Streamlit dashboard (replay of queues/phases per intersection, results) |
| `results/`, `models/`, `traces/` | CSVs, plots, trained weights, replay traces |
| `docs/` | Learning guide, 30-day timeline and work split, results/report draft |

## Formulation
* **Observation** (per intersection): queue and vehicle count per incoming lane (8 lanes, normalised), current phase one-hot, time in phase.
  Cooperative agent adds, for each of 4 neighbour slots (N/E/S/W), the neighbour's mean queue (level 1) and current phase (level 2); missing neighbours are zero-padded.
* **Action:** which green phase to serve next (2 phases: north-south / east-west); switching inserts a 2 s yellow. Decision every 5 s.
* **Reward:** minus total queue on the incoming lanes / 10. Cooperative agent: `r_i + 0.5 * mean(r_neighbours)` (discourages pushing congestion onto neighbours).
* **Demand:** Bernoulli arrivals (approximately Poisson) from 12 boundary entries to all other exits, 1200 s episodes, `DEMAND_SCALE = 2.2` in `scenarios.py` (calibrated to be congested but not gridlocked).
* **Methods:** `fixed_time`; `max_pressure` (reference, not in the proposal); `independent` (separate network per intersection, no information exchange);
  `coop` (shared network + neighbour info level 2 + reward mixing); `shared_c0/c1/c2` (shared network, reward not mixed, neighbour info level 0/1/2 -> isolates the effect of communication, RQ2).

## Known limitations (be upfront about these in the report)
* Only 2 green phases (no protected left-turn phases) -> binary action. A 4-phase network is a natural extension.
* Short training, few seeds: differences between the learned methods can be within noise; use `results/stat_tests.csv`.
* `independent` differs from `coop` in more than communication (separate vs shared weights, no reward mixing); the `shared_c*` ablation separates those factors.
* Travel-time metrics count only vehicles that entered the network; vehicles waiting to be inserted are not counted.
* Learned agents are evaluated after training with epsilon = 0 on unseen seeds; scenarios are cycled during training (one policy for all scenarios).
