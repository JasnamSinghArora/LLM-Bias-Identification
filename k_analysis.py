import os
import csv
import math

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from constants import constants

OUT_DIR = os.path.join(constants["RUN_DIR"], "out_k")
SINGULAR_VALUES_PATH = os.path.join(constants["RUN_DIR"], "singular_values.pt")
KS = range(1, 21)
LINE = "#2f4b7c"


def load_scores(k, out_dir=OUT_DIR):
    path = f"{out_dir}/{k}/benchmark_scores.csv"
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        return [float(row[0]) for row in reader if row and row[0] != ""]


def load_stats(out_dir=OUT_DIR):
    """Mean, standard error and gain per k."""
    ks, means, sems = [], [], []
    for k in KS:
        scores = load_scores(k, out_dir)
        if not scores:
            continue
        n = len(scores)
        mean = sum(scores) / n
        var = sum((s - mean) ** 2 for s in scores) / (n - 1) if n > 1 else 0.0
        ks.append(k)
        means.append(mean)
        sems.append(math.sqrt(var / n))
    gains = [m - (means[i - 1] if i else 0.0) for i, m in enumerate(means)]
    return ks, means, sems, gains


def mark_k_star(ax, k_star, text="k*"):
    if k_star:
        ax.axvline(k_star, color="crimson", linestyle=":", linewidth=1.2)
        ax.text(k_star + 0.3, 0.05, f"{text}={k_star}", color="crimson",
                transform=ax.get_xaxis_transform(), fontsize=8)


def plot_mean_bias_score(ax, out_dir=OUT_DIR, k_star=None, title="Mean bias score vs. k"):
    ks, means, sems, _ = load_stats(out_dir)
    ax.errorbar(ks, means, yerr=sems, marker="s", markersize=3, capsize=2, color=LINE)
    mark_k_star(ax, k_star)
    ax.set_xlabel("k (bias subspace dimension)")
    ax.set_ylabel("Mean bias score")
    ax.set_title(title)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(True, alpha=0.3)


def plot_marginal_gain(ax, out_dir=OUT_DIR, k_star=None, title="Marginal gain per dimension"):
    ks, _, _, gains = load_stats(out_dir)
    ax.bar(ks, gains, color=["crimson" if k == k_star else LINE for k in ks])
    ax.set_xlabel("k (bias subspace dimension)")
    ax.set_ylabel("Marginal gain")
    ax.set_title(title)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(True, axis="y", alpha=0.3)


def plot_bell_curves(ax, out_dir=OUT_DIR, title="Fitted Gaussian per k", colorbar=True):
    curves = []
    lo, hi = math.inf, -math.inf
    for k in KS:
        scores = load_scores(k, out_dir)
        if len(scores) < 2:
            continue
        curves.append((k, float(np.mean(scores)), float(np.std(scores, ddof=1))))
        lo, hi = min(lo, min(scores)), max(hi, max(scores))
    if not curves:
        return
    xs = np.linspace(lo, hi, 500)
    colors = cm.viridis(np.linspace(0, 1, len(curves)))
    for (k, mean, std), color in zip(curves, colors):
        pdf = np.exp(-0.5 * ((xs - mean) / std) ** 2) / (std * math.sqrt(2 * math.pi))
        ax.plot(xs, pdf, color=color, linewidth=1)
    if colorbar:
        sm = cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(curves[0][0], curves[-1][0]))
        sm.set_array([])
        ax.figure.colorbar(sm, ax=ax, label="k")
    ax.set_xlabel("Bias score")
    ax.set_ylabel("Probability density")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_singular_values(ax, S, k_star=None, title="Leading singular values"):
    s = np.asarray(S[:20], dtype=float)
    ax.semilogy(np.arange(1, len(s) + 1), s, marker="o", markersize=3, color=LINE)
    if k_star:
        ax.axvspan(k_star, k_star + 1, color="crimson", alpha=0.15)
        mark_k_star(ax, k_star, text="i*")
    ax.set_xlabel("Index i")
    ax.set_ylabel("Singular value")
    ax.set_title(title)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(True, which="both", alpha=0.3)


def save(draw, path, **kwargs):
    fig, ax = plt.subplots(figsize=(8, 5))
    draw(ax, **kwargs)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_all(k_star=None):
    run_dir = constants["RUN_DIR"]
    save(plot_mean_bias_score, os.path.join(run_dir, "k_mean_bias_score.png"), k_star=k_star)
    save(plot_marginal_gain, os.path.join(run_dir, "k_marginal_gain.png"), k_star=k_star)
    save(plot_bell_curves, os.path.join(run_dir, "k_bell_curves.png"))
    if os.path.exists(SINGULAR_VALUES_PATH):
        S = torch.load(SINGULAR_VALUES_PATH).float().cpu().tolist()
        save(plot_singular_values, os.path.join(run_dir, "singular_values.png"), S=S, k_star=k_star)


if __name__ == "__main__":
    run_all()
