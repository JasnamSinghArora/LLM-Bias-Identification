"""Paper tables and figures from run folders."""
import os
import sys
import csv
import math

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import wilcoxon

from constants import constants
import k_analysis
import bell_curve

MODEL_NAMES = {
    "llama-3.1-8b-instruct": "Llama-3.1-8B-Instruct",
    "qwen3-8b": "Qwen3-8B",
    "ministral-3-8b-instruct-2512-bf16": "Ministral-3-8B-Instruct",
}
DELTA = 1e-8  # spectral-gap stabiliser


def list_runs(runs_root):
    runs = []
    for model in constants["TEST_LLMS"]:
        for cat in constants["BIAS_CATEGORIES"]:
            run_dir = os.path.join(runs_root, model["name"], cat["MF"])
            if os.path.isdir(run_dir):
                runs.append((model["name"], cat["MF"], run_dir))
    return runs


def label(model, mf):
    return f"{MODEL_NAMES.get(model, model)}: {mf.capitalize()}"


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def chosen_k(run_dir):
    path = os.path.join(run_dir, "bias_subspace_cache.pt")
    return int(torch.load(path)["bias_subspace"].shape[0]) if os.path.exists(path) else None


def singular_values(run_dir):
    path = os.path.join(run_dir, "singular_values.pt")
    return torch.load(path).float().cpu().numpy() if os.path.exists(path) else None


def benchmark_scores(run_dir):
    path = os.path.join(run_dir, "benchmark_scores.csv")
    return bell_curve.load_scores(path) if os.path.exists(path) else []


def optimized_scores(run_dir):
    recs = read_csv(os.path.join(run_dir, "optimized_scores.csv"))
    orig = np.array([float(r["original_score"]) for r in recs])
    opt = np.array([float(r["optimized_score"]) for r in recs])
    return orig, opt


def steps_per_prompt(run_dir):
    steps = {}
    for r in read_csv(os.path.join(run_dir, "optimization_trajectories.csv")):
        steps[r["prompt_index"]] = max(steps.get(r["prompt_index"], 0), int(r["step"]))
    return list(steps.values())


# ---------- tables ----------

def table_diversity(runs):
    """Table 2."""
    rows = []
    for model, mf, run_dir in runs:
        vals = {r["metric"]: float(r["value"]) for r in read_csv(os.path.join(run_dir, "cpd_quality.csv"))}
        if "mst_diversity" in vals:
            rows.append({"model": model, "MF": mf, "mst_diversity": vals["mst_diversity"]})
    return rows


def table_benchmark(runs):
    """Table 3."""
    rows = []
    for model, mf, run_dir in runs:
        scores = benchmark_scores(run_dir)
        if scores:
            rows.append({"model": model, "MF": mf, **bell_curve.describe(scores)})
    return rows


def table_spectral_gap(runs):
    """Table 4."""
    rows = []
    for model, mf, run_dir in runs:
        k = chosen_k(run_dir)
        S = singular_values(run_dir)
        if k is None or S is None:
            continue
        ks, means, _, gains = k_analysis.load_stats(os.path.join(run_dir, "out_k"))
        at = {kk: (m, g) for kk, m, g in zip(ks, means, gains)}
        rows.append({
            "model": model,
            "MF": mf,
            "k_star": k,
            "sigma_k": float(S[k - 1]),
            "sigma_k_plus_1": float(S[k]),
            "log_drop": math.log((S[k - 1] + DELTA) / (S[k] + DELTA)),
            "variance_share": float((S[:k] ** 2).sum() / (S ** 2).sum()),
            "mean_score_at_k": at.get(k, (math.nan, math.nan))[0],
            "gain_at_k": at.get(k, (math.nan, math.nan))[1],
            "gain_at_k_plus_1": at.get(k + 1, (math.nan, math.nan))[1],
        })
    return rows


def table_optimization(runs):
    """Tables 6 and 7."""
    rows = []
    for model, mf, run_dir in runs:
        orig, opt = optimized_scores(run_dir)
        if len(orig) == 0:
            continue
        delta = opt - orig
        steps = steps_per_prompt(run_dir)
        n = len(orig)
        rows.append({
            "model": model,
            "MF": mf,
            "n": n,
            "wilcoxon_p": float(wilcoxon(opt, orig).pvalue) if np.any(delta != 0) else 1.0,
            "mean_steps": float(np.mean(steps)) if steps else math.nan,
            "max_steps": int(max(steps)) if steps else 0,
            "benchmark_mean": float(orig.mean()),
            "benchmark_se": float(orig.std(ddof=1) / math.sqrt(n)),
            "optimized_mean": float(opt.mean()),
            "optimized_se": float(opt.std(ddof=1) / math.sqrt(n)),
            "benchmark_median": float(np.median(orig)),
            "optimized_median": float(np.median(opt)),
            "median_delta": float(np.median(delta)),
            "increased": int((delta > 0).sum()),
            "unchanged": int((delta == 0).sum()),
            "decreased": int((delta < 0).sum()),
        })
    return rows


# ---------- figures ----------

def grid(runs, draw, path, finish=None):
    """One panel per run, three per row."""
    ncols = 3
    nrows = math.ceil(len(runs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), squeeze=False)
    for ax, (model, mf, run_dir) in zip(axes.flat, runs):
        draw(ax, run_dir, label(model, mf))
    for ax in list(axes.flat)[len(runs):]:
        ax.set_visible(False)
    fig.tight_layout()
    if finish:
        finish(fig, axes)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def draw_benchmark(ax, run_dir, title):
    scores = benchmark_scores(run_dir)
    if scores:
        bell_curve.plot_distribution(ax, scores, title)


def draw_mean_score(ax, run_dir, title):
    k_analysis.plot_mean_bias_score(ax, os.path.join(run_dir, "out_k"), chosen_k(run_dir), title)


def draw_gain(ax, run_dir, title):
    k_analysis.plot_marginal_gain(ax, os.path.join(run_dir, "out_k"), chosen_k(run_dir), title)


def draw_singular(ax, run_dir, title):
    S = singular_values(run_dir)
    if S is not None:
        k_analysis.plot_singular_values(ax, S, chosen_k(run_dir), title)


def draw_gaussians(ax, run_dir, title):
    k_analysis.plot_bell_curves(ax, os.path.join(run_dir, "out_k"), title, colorbar=False)


def shared_colorbar(fig, axes):
    sm = cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(1, 20))
    sm.set_array([])
    fig.colorbar(sm, ax=list(axes.flat), label="k", shrink=0.6)


def draw_optimized(ax, run_dir, title):
    orig, opt = optimized_scores(run_dir)
    if len(orig) == 0:
        return
    lo = max(min(orig.min(), opt.min()), 1e-6)
    hi = max(orig.max(), opt.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), 40)
    ax.hist(np.maximum(orig, lo), bins=bins, density=True, alpha=0.6, color="lightsteelblue", label="Benchmark (original)")
    ax.hist(np.maximum(opt, lo), bins=bins, density=True, alpha=0.6, color="indianred", label="Optimized")
    ax.axvline(np.median(orig), color="black", linestyle="--", linewidth=1)
    ax.axvline(np.median(opt), color="crimson", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.text(0.97, 0.95, f"med0={np.median(orig):.3f}\nmed*={np.median(opt):.3f}\nn={len(orig)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_xlabel("Bias score (log scale)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    runs_root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PIPELINE_RUNS_DIR", os.path.join(here, "runs"))
    out_dir = os.environ.get("PIPELINE_RESULTS_DIR", os.path.join(here, "results"))
    runs = list_runs(runs_root)
    if not runs:
        sys.exit(f"no run folders under {runs_root}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"{len(runs)} run folders under {runs_root}")

    write_csv(os.path.join(out_dir, "table2_mst_diversity.csv"), table_diversity(runs))
    write_csv(os.path.join(out_dir, "table3_benchmark_stats.csv"), table_benchmark(runs))
    write_csv(os.path.join(out_dir, "table4_spectral_gap.csv"), table_spectral_gap(runs))
    write_csv(os.path.join(out_dir, "table6_7_optimization.csv"), table_optimization(runs))

    grid(runs, draw_benchmark, os.path.join(out_dir, "fig3_benchmark_distributions.png"))
    grid(runs, draw_mean_score, os.path.join(out_dir, "fig4_mean_bias_score_vs_k.png"))
    grid(runs, draw_gain, os.path.join(out_dir, "fig5_marginal_gain.png"))
    grid(runs, draw_singular, os.path.join(out_dir, "fig6_singular_values.png"))
    grid(runs, draw_gaussians, os.path.join(out_dir, "fig7_gaussians_per_k.png"), finish=shared_colorbar)
    grid(runs, draw_optimized, os.path.join(out_dir, "fig12_benchmark_vs_optimized.png"))


if __name__ == "__main__":
    main()
