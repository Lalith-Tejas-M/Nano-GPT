"""Aggregate sweep results into plots and a markdown table.
  python plot.py --results results --out figures
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# two-sided 95% Student-t critical values by degrees of freedom (df = n - 1)
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
       11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
       25: 2.060, 30: 2.042}


def t_crit(df):
    """Student-t multiplier for a 95% interval; nearest tabulated df at or below `df`, 1.96 beyond 30."""
    if df < 1:
        return 0.0
    if df > 30:
        return 1.96
    return T95[max(k for k in T95 if k <= df)]


def load(results_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "result.json"))):
        with open(path) as f:
            r = json.load(f)
        name = os.path.basename(os.path.dirname(path))
        size, ln, _ = name.split("_")
        r.update(size=size, ln=ln, name=name)
        runs.append(r)
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    p.add_argument("--metric", choices=["best_val", "final_val"], default="best_val",
                   help="best_val = lowest validation loss over eval checkpoints (early-stopping proxy); final_val = last checkpoint")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    runs = load(a.results)
    if not runs:
        raise SystemExit(f"No result.json files found in {a.results}")

    groups = defaultdict(list)
    for r in runs:
        groups[(r["ln"], r["size"])].append(r)
    stats = {}
    for key, rs in groups.items():
        vals = np.array([r[a.metric] for r in rs])
        stats[key] = dict(
            n=len(rs), params=rs[0]["n_params"], params_ne=rs[0]["n_params_non_embedding"],
            mean=vals.mean(), std=vals.std(ddof=1) if len(vals) > 1 else 0.0,
            ci95=t_crit(len(vals) - 1) * vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0,
        )

    # 1) scaling plot (pre-LN)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    slope_txt = ""
    for ln, marker in (("pre", "o"), ("post", "s")):
        ks = sorted([k for k in stats if k[0] == ln], key=lambda k: stats[k]["params"])
        if not ks:
            continue
        x = np.array([stats[k]["params"] for k in ks], dtype=float)
        y = np.array([stats[k]["mean"] for k in ks])
        e = np.array([stats[k]["std"] for k in ks])
        ax.errorbar(x, y, yerr=e, marker=marker, capsize=3, label=f"{ln}-LN (mean +/- std over seeds)")
        if ln == "pre" and len(x) >= 2:
            slope, intercept = np.polyfit(np.log10(x), np.log10(y), 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, 10 ** (intercept + slope * np.log10(xs)), "--", alpha=0.6, label=f"log-log fit, slope {slope:.3f}")
            slope_txt = f"Log-log slope of validation loss vs. parameters (pre-LN): {slope:.3f}\n"
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Parameters (total)"); ax.set_ylabel(f"Validation loss, {a.metric} (nats/char)")
    ax.set_title("Scaling: validation loss vs. model size"); ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "scaling.png"), dpi=150); plt.close(fig)

    # 2) training curves (seed 0, pre-LN)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r in runs:
        if r["ln"] == "pre" and r["name"].endswith("seed0"):
            ax.plot([h["iter"] for h in r["history"]], [h["val"] for h in r["history"]], label=f"{r['size']} ({r['n_params']/1e6:.2f}M)")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Validation loss"); ax.set_title("Validation loss during training (seed 0)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(os.path.join(a.out, "curves.png"), dpi=150); plt.close(fig)

    # 3) table
    lines = ["| LN | size | params | non-embedding params | n seeds | val loss (mean, " + a.metric + ") | std | 95% CI half-width (Student-t) |",
             "|---|---|---|---|---|---|---|---|"]
    for k in sorted(stats, key=lambda k: (k[0], stats[k]["params"])):
        s = stats[k]
        lines.append(f"| {k[0]} | {k[1]} | {s['params']:,} | {s['params_ne']:,} | {s['n']} | {s['mean']:.4f} | {s['std']:.4f} | {s['ci95']:.4f} |")
    md = "\n".join(lines) + "\n\n" + slope_txt
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
