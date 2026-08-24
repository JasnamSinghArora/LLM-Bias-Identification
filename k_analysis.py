import os
import csv
import math
import numpy as np
import matplotlib  # ADDED: needed to select a non-interactive backend before pyplot loads
matplotlib.use("Agg")  # ADDED: headless backend so the pipeline never blocks on GUI windows
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import make_interp_spline
from constants import constants  # ADDED: needed for the per-run directory

BUCKET_SIZE = 3

OUT_DIR = os.path.join(constants["RUN_DIR"], "out_k")  # EDITED: per-run directory (was the hardcoded "out_k")
KS = range(1, 21)


def load_scores(k):
    scores = []
    with open(f"{OUT_DIR}/{k}/benchmark_scores.csv", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if row and row[0] != "":
                scores.append(float(row[0]))
    return scores


def load_stats():
    ks = []
    means = []
    cis = []
    for k in KS:
        path = f"{OUT_DIR}/{k}/benchmark_scores.csv"
        if not os.path.exists(path):
            continue
        scores = load_scores(k)
        if not scores:
            continue
        n = len(scores)
        mean = sum(scores) / n
        var = sum((s - mean) ** 2 for s in scores) / (n - 1) if n > 1 else 0.0
        sem = math.sqrt(var / n)
        ks.append(k)
        means.append(mean)
        cis.append(1.96 * sem)  # 95% CI half-width
    return ks, means, cis


def plot_mean_bias_score():
    ks, means, cis = load_stats()

    plt.figure(figsize=(8, 5))
    plt.errorbar(ks, means, yerr=cis, marker="o", capsize=4)
    plt.xlabel("k (bias subspace dimension)")
    plt.ylabel("Mean bias score")
    plt.title("Mean bias score vs. k")
    plt.xticks(list(ks))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(constants["RUN_DIR"], "k_mean_bias_score.png"), dpi=150)  # EDITED: per-run path
    plt.close()  # EDITED: was plt.show() — the pipeline must not block on a window


def plot_marginal_gain():
    ks, means, _ = load_stats()

    # marginal gain of the k-th dimension = mean(k) - mean(k-1), with mean(0) = 0
    gains = []
    prev = 0.0
    for mean in means:
        gains.append(mean - prev)
        prev = mean

    plt.figure(figsize=(8, 5))
    plt.bar(ks, gains)
    plt.xlabel("k (bias subspace dimension)")
    plt.ylabel("Marginal gain in mean bias score")
    plt.title("Marginal gain contributed by each additional dimension")
    plt.xticks(list(ks))
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(constants["RUN_DIR"], "k_marginal_gain.png"), dpi=150)  # EDITED: per-run path
    plt.close()  # EDITED: was plt.show() — the pipeline must not block on a window


def plot_bell_curves():
    ks = []
    curves = []
    global_min = math.inf
    global_max = -math.inf
    for k in KS:
        path = f"{OUT_DIR}/{k}/benchmark_scores.csv"
        if not os.path.exists(path):
            continue
        scores = load_scores(k)
        if len(scores) < 2:
            continue
        n = len(scores)
        mean = sum(scores) / n
        std = math.sqrt(sum((s - mean) ** 2 for s in scores) / (n - 1))
        ks.append(k)
        curves.append((mean, std))
        global_min = min(global_min, min(scores))
        global_max = max(global_max, max(scores))

    xs = np.linspace(global_min, global_max, 500)
    colors = cm.viridis(np.linspace(0, 1, len(ks)))

    plt.figure(figsize=(9, 5))
    for (mean, std), k, color in zip(curves, ks, colors):
        pdf = (1.0 / (std * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((xs - mean) / std) ** 2)
        plt.plot(xs, pdf, color=color, label=f"k={k}")
    plt.xlabel("Bias score")
    plt.ylabel("Probability density")
    plt.title("Bias score distribution (fitted Gaussian) for each k")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(constants["RUN_DIR"], "k_bell_curves.png"), dpi=150)  # EDITED: per-run path
    plt.close()  # EDITED: was plt.show() — the pipeline must not block on a window


def bucket_scores(scores, size):
    lo = (int(min(scores)) // size) * size
    hi = (int(max(scores)) // size + 1) * size
    edges = list(range(lo, hi + 1, size))
    counts = [0] * (len(edges) - 1)
    for s in scores:
        idx = min(int((s - lo) // size), len(counts) - 1)
        counts[idx] += 1
    centers = [(edges[i] + edges[i + 1] - 1) / 2 for i in range(len(counts))]
    return centers, counts, edges


def plot_individual_bell_curves():
    ks = [k for k in KS if os.path.exists(f"{OUT_DIR}/{k}/benchmark_scores.csv")]

    ncols = 5
    nrows = math.ceil(len(ks) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = axes.flatten()

    for ax, k in zip(axes, ks):
        scores = load_scores(k)
        if len(scores) < 2:
            ax.set_visible(False)
            continue
        centers, counts, edges = bucket_scores(scores, BUCKET_SIZE)

        mu = float(np.mean(scores))
        sigma = float(np.std(scores))

        x_curve = np.linspace(edges[0], edges[-1], 400)
        pdf = (1 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x_curve - mu) / sigma) ** 2)
        y_curve = pdf * len(scores) * BUCKET_SIZE

        ax.scatter(centers, counts, color="steelblue", edgecolor="black",
                   s=30, zorder=3, label="Frequency")
        ax.plot(x_curve, y_curve, color="crimson", linewidth=2, label="Normal fit")

        if len(centers) > 3:
            spline = make_interp_spline(centers, counts, k=3)
            x_smooth = np.linspace(min(centers), max(centers), 400)
            ax.plot(x_smooth, spline(x_smooth), color="seagreen", linewidth=1.5,
                    linestyle="--", label="Connecting curve")

        ax.set_title(f"k={k}  (μ={mu:.1f}, σ={sigma:.1f})", fontsize=9)
        ax.set_xlabel("Score", fontsize=8)
        ax.set_ylabel("Frequency", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    for ax in axes[len(ks):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=9)
    fig.suptitle("Benchmark score bell curve per k", y=1.0, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(constants["RUN_DIR"], "k_bell_curves_individual.png"), dpi=150)  # EDITED: per-run path
    plt.close()  # EDITED: was plt.show() — the pipeline must not block on a window


def run_all():  # ADDED: single entry point for the pipeline's step-12 graphs
    plot_mean_bias_score()  # ADDED
    plot_marginal_gain()  # ADDED
    plot_bell_curves()  # ADDED
    plot_individual_bell_curves()  # EDITED: was a bare module-level call (and the only plot being produced)


if __name__ == "__main__":  # ADDED: guard so importing this module no longer runs it
    run_all()  # ADDED
