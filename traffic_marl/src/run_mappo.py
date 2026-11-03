"""Train + evaluate MAPPO on the same env, scenarios, seeds and metrics as the DQN methods.

  python src/run_mappo.py --episodes 100 --seed 0                 # shared actor, own obs only (level 0)
  python src/run_mappo.py --episodes 100 --seed 0 --level 2 --beta 0.5   # + neighbour info and reward mixing (like `coop`)

Outputs (same naming scheme as run_experiments.py): results/curve_<tag>.csv, results/eval_<tag>.csv,
results/settings_<tag>.json, models/<tag>.pt, traces/<tag>.json  with tag = mappo[_c<level>]_s<seed>.
The `method` column is "mappo" (level 0) or "mappo_c<level>", so it never collides with the DQN files.
"""
import argparse, datetime, json, os, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import SumoGridEnv
import scenarios as S
from run_experiments import ROOT, EVAL_SEEDS, train_seed, check_seed_split, eval_policy, git_commit
from agents import mix_rewards
from mappo import MAPPO


def train(episodes, seed, level, beta):
    method = "mappo" if level == 0 else f"mappo_c{level}"
    tag = f"{method}_s{seed}"
    check_seed_split(seed, episodes)
    for d in ("results", "models", "traces"):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)
    np.random.seed(seed); torch.manual_seed(seed)
    env = SumoGridEnv(comm_level=0)
    agent = MAPPO(env, level=level)
    curve, t0 = [], time.time()
    for ep in range(episodes):
        scn = S.SCENARIOS[ep % len(S.SCENARIOS)]
        obs = env.reset(scn, train_seed(seed, ep)); done, ret = False, 0.0
        while not done:
            act = agent.act(obs, explore=True)
            obs2, r, done, _ = env.step(act)
            agent.store(mix_rewards(env, r, beta))
            ret += np.mean(list(r.values())); obs = obs2          # return logged on the raw reward
        st = agent.update(obs)                                    # time-limit end -> bootstrap from last obs
        m = env.metrics()
        curve.append(dict(episode=ep, scenario=scn, ret=ret, avg_queue=float(m["avg_queue"]),
                          avg_wait_s=float(m["avg_wait_s"]), **st))
        print(f"[{method} s{seed}] ep {ep+1}/{episodes} {scn:10s} return={ret:8.1f} avg_wait={m['avg_wait_s']:.1f} "
              f"ent={st['entropy']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(curve).to_csv(os.path.join(ROOT, "results", f"curve_{tag}.csv"), index=False)
    torch.save(agent.state_dict(), os.path.join(ROOT, "models", f"{tag}.pt"))
    train_time = time.time() - t0
    rows = eval_policy(method, env, lambda o: agent.act(o, explore=False), train_seed=seed)
    env.close()
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", f"eval_{tag}.csv"), index=False)
    settings = dict(method=method, seed=seed, episodes=episodes, comm_level=level, reward_mix_beta=beta,
                    train_seed_formula="1000*seed+episode", eval_seeds=EVAL_SEEDS, scenarios=list(S.SCENARIOS),
                    **agent.hyperparams(), git_commit=git_commit(), train_seconds=round(train_time),
                    finished=datetime.datetime.now().isoformat(timespec="seconds"))
    json.dump(settings, open(os.path.join(ROOT, "results", f"settings_{tag}.json"), "w"), indent=2)
    print(f"[{method} s{seed}] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--beta", type=float, default=0.0)
    a = ap.parse_args()
    train(a.episodes, a.seed, a.level, a.beta)
