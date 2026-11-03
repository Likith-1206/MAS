"""Compare MAPPO with DQN variants and baselines (mean +- std over all eval episodes; lower wait/queue is better)."""
import glob, os, sys
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(ROOT, "results", "eval_*.csv"))])
cols = ["avg_wait_s", "avg_queue", "throughput", "avg_trip_s"]
g = d.groupby("method")[cols]
tab = g.mean().round(2).join(g.std().round(2), rsuffix="_std").join(d.groupby("method").train_seed.nunique().rename("n_seeds"))
tab = tab.sort_values("avg_wait_s")
print(tab.to_string())
tab.to_csv(os.path.join(ROOT, "results", "comparison_mappo.csv"))
order = list(tab.index)
colors = ["tab:red" if m.startswith("mappo") else "tab:gray" for m in order]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, c, t in zip(ax, ["avg_wait_s", "throughput"], ["Average wait (s) - lower is better", "Throughput (vehicles) - higher is better"]):
    a.barh(order, tab[c], xerr=tab[c + "_std"], color=colors); a.set_title(t); a.invert_yaxis()
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "results", "comparison_mappo.png"), dpi=150)
print("saved results/comparison_mappo.csv and results/comparison_mappo.png")
