"""Aggregate sweep results into statistical summaries, plots, and markdown reports.

Supports:
- CLI usage: python plot.py --results results --out figures --metric best_val
- Library usage: aggregate_and_plot(...)
"""
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Two-sided 95% Student-t critical values by degrees of freedom (df = n - 1)
T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    25: 2.060,
    30: 2.042,
}


def t_crit(df: int) -> float:
    """Student-t multiplier for 95% confidence interval (df = n - 1)."""
    if df < 1:
        return 0.0
    if df > 30:
        return 1.96
    return T95[max(k for k in T95 if k <= df)]


def compute_group_stats(vals: list[float], n_params: int, n_params_ne: int) -> dict:
    """Computes mean, sample std (ddof=1), and Student-t 95% CI half-width."""
    arr = np.array(vals, dtype=float)
    n = len(arr)
    if n == 0:
        return dict(
            n=0,
            raw_values=[],
            params=n_params,
            params_ne=n_params_ne,
            mean=0.0,
            std=0.0,
            ci95=0.0,
        )
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    crit = t_crit(n - 1)
    ci95 = float(crit * std / np.sqrt(n)) if n > 1 else 0.0
    return dict(
        n=n,
        raw_values=[round(v, 6) for v in vals],
        params=n_params,
        params_ne=n_params_ne,
        mean=round(mean, 6),
        std=round(std, 6),
        ci95=round(ci95, 6),
    )


def load_runs(results_dir: str) -> list[dict]:
    """Safely loads and validates all result.json files found in results_dir."""
    runs = []
    pattern = os.path.join(results_dir, "*", "result.json")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, encoding="utf-8") as f:
                r = json.load(f)
        except Exception as e:
            print(f"Warning: could not parse {path}: {e}")
            continue

        dir_name = os.path.basename(os.path.dirname(path))
        # Infer size and ln placement from config or dir name
        cfg = r.get("config", {})
        ln = cfg.get("ln_placement")
        size = None

        m = re.match(r"^([a-zA-Z0-9]+)_(pre|post)_seed(\d+)$", dir_name)
        if m:
            size_candidate, ln_candidate, seed_candidate = m.group(1), m.group(2), int(m.group(3))
            size = size or size_candidate
            ln = ln or ln_candidate
            seed = cfg.get("seed", seed_candidate)
        else:
            seed = cfg.get("seed", 0)

        # Infer size if not in name
        if not size:
            n_layer = cfg.get("n_layer")
            n_embd = cfg.get("n_embd")
            if (n_layer, n_embd) == (2, 64):
                size = "xs"
            elif (n_layer, n_embd) == (4, 128):
                size = "s"
            elif (n_layer, n_embd) == (6, 192):
                size = "m"
            elif (n_layer, n_embd) == (6, 384):
                size = "l"
            else:
                size = f"L{n_layer}E{n_embd}"

        ln = ln or "pre"
        r.update(size=size, ln=ln, name=dir_name, seed=seed)
        runs.append(r)
    return runs


def aggregate_and_plot(results_dir: str = "results", out_dir: str = "figures", metric: str = "best_val") -> dict:
    """Aggregates metrics, computes Student-t statistics, and generates figures and summaries."""
    os.makedirs(out_dir, exist_ok=True)
    runs = load_runs(results_dir)
    if not runs:
        raise ValueError(f"No valid result.json files found in '{results_dir}'")

    # Group by (ln_placement, size)
    groups = defaultdict(list)
    for r in runs:
        if metric in r and math.isfinite(r[metric]):
            groups[(r["ln"], r["size"])].append(r)

    stats = {}
    for key, rs in groups.items():
        vals = [r[metric] for r in rs]
        n_params = rs[0].get("n_params", 0)
        n_params_ne = rs[0].get("n_params_non_embedding", 0)
        stats[key] = compute_group_stats(vals, n_params, n_params_ne)

    # 1. Scaling Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    slope_text = ""
    for ln, marker in (("pre", "o"), ("post", "s")):
        ks = sorted([k for k in stats if k[0] == ln], key=lambda k: stats[k]["params"])
        if not ks:
            continue
        x = np.array([stats[k]["params"] for k in ks], dtype=float)
        y = np.array([stats[k]["mean"] for k in ks])
        e = np.array([stats[k]["std"] for k in ks])

        ax.errorbar(
            x,
            y,
            yerr=e,
            marker=marker,
            capsize=4,
            linewidth=1.5,
            label=f"{ln}-LN (mean \u00b1 sample std, ddof=1)",
        )

        if ln == "pre" and len(x) >= 2 and all(y > 0) and x.min() != x.max():
            slope, intercept = np.polyfit(np.log10(x), np.log10(y), 1)
            xs = np.geomspace(x.min(), x.max(), 50)
            ys = 10 ** (intercept + slope * np.log10(xs))
            ax.plot(xs, ys, "--", alpha=0.7, label=f"Pre-LN log-log fit (slope {slope:.3f})")
            slope_text = f"Pre-LN log-log validation loss scaling slope: {slope:.4f}\n"

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total Parameters (log scale)")
    ax.set_ylabel(f"Validation Loss [{metric}] (nats/char, log scale)")
    ax.set_title("Model Size Scaling: Validation Loss vs. Parameter Count")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3, linestyle=":")
    fig.tight_layout()
    scaling_plot_path = os.path.join(out_dir, "scaling.png")
    fig.savefig(scaling_plot_path, dpi=150)
    plt.close(fig)

    # 2. Validation Loss Curves Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    curves_plotted = 0
    for r in sorted(runs, key=lambda x: (x.get("n_params", 0), x.get("seed", 0))):
        if r["ln"] == "pre" and r.get("seed", 0) == 0 and "history" in r:
            iters = [h["iter"] for h in r["history"] if "iter" in h and "val" in h]
            val_losses = [h["val"] for h in r["history"] if "iter" in h and "val" in h]
            if iters and val_losses:
                label = f"{r['size']} ({r.get('n_params', 0)/1e6:.2f}M params)"
                ax.plot(iters, val_losses, label=label, linewidth=1.5)
                curves_plotted += 1

    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Validation Loss (nats/char)")
    ax.set_title("Validation Loss Trajectory During Training (Seed 0, Pre-LN)")
    if curves_plotted > 0:
        ax.legend()
    ax.grid(True, alpha=0.3, linestyle=":")
    fig.tight_layout()
    curves_plot_path = os.path.join(out_dir, "curves.png")
    fig.savefig(curves_plot_path, dpi=150)
    plt.close(fig)

    # 3. Markdown Summary Table
    md_lines = [
        f"# Scaling Study Summary ({metric})",
        "",
        "> **Note on Metrics & Uncertainty:**",
        f"> - Metric reported: `{metric}`.",
        "> - `best_val` is the minimum validation loss across periodic evaluation checkpoints (an early-stopping proxy, not an unbiased test-set metric).",
        "> - Standard deviation is computed with sample degrees of freedom (ddof=1).",
        "> - Student-t 95% Confidence Interval half-widths use two-sided critical values (t=4.303 for n=3, df=2).",
        "> - Note: 3 seeds provide preliminary sample dispersion; error intervals are rough approximations.",
        "",
        "| LN Placement | Size | Parameters | Non-Embedding Params | N Seeds | Raw Values | Mean Val Loss | Std Dev (ddof=1) | 95% CI Half-Width |",
        "|:---:|:---:|---:|---:|:---:|:---|---:|---:|---:|",
    ]

    for k in sorted(stats, key=lambda k: (k[0], stats[k]["params"])):
        s = stats[k]
        raw_str = ", ".join(f"{v:.4f}" for v in s["raw_values"])
        md_lines.append(
            f"| {k[0]} | {k[1]} | {s['params']:,} | {s['params_ne']:,} | {s['n']} | [{raw_str}] | {s['mean']:.4f} | {s['std']:.4f} | \u00b1{s['ci95']:.4f} |"
        )

    md_lines.append("")
    if slope_text:
        md_lines.append(f"**Empirical Scaling Slope:** {slope_text.strip()}")
        md_lines.append(
            "> *Caution: The log-log linear fit is descriptive empirical evidence across 4 small model configurations on Tiny Shakespeare; it does not constitute proof of an asymptotic or universal scaling law.*"
        )
    md_lines.append("")

    summary_md_path = os.path.join(out_dir, "summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # 4. Machine-Readable Summary JSON
    export_stats = {f"{k[0]}_{k[1]}": v for k, v in stats.items()}
    summary_json_path = os.path.join(out_dir, "summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump({"metric": metric, "stats": export_stats}, f, indent=2)

    return {"stats": export_stats, "summary_md": summary_md_path, "summary_json": summary_json_path}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="nanoGPT Results Aggregation and Plotting")
    p.add_argument("--results", default="results", help="Directory containing run results")
    p.add_argument("--out", default="figures", help="Directory to save figures and summaries")
    p.add_argument(
        "--metric",
        choices=["best_val", "final_val"],
        default="best_val",
        help="Metric to aggregate: 'best_val' (early stopping proxy) or 'final_val'",
    )
    args = p.parse_args()

    try:
        res = aggregate_and_plot(results_dir=args.results, out_dir=args.out, metric=args.metric)
        with open(res["summary_md"], encoding="utf-8") as f:
            print(f.read())
    except Exception as e:
        import sys
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)
