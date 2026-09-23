import os
import csv
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew

from constants import constants

CSV_PATH = os.path.join(constants["RUN_DIR"], "benchmark_scores.csv")
STATS_CSV_PATH = os.path.join(constants["RUN_DIR"], "benchmark_stats.csv")
BINS = 40


def load_scores(path):
    with open(path, newline="") as f:
        return [float(row["score"]) for row in csv.DictReader(f) if row.get("score", "").strip()]


def describe(scores):
    """Table 3 statistics of a score list."""
    s = np.asarray(scores, dtype=float)
    q1, median, q3 = np.percentile(s, [25, 50, 75])
    sd = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    return {
        "n": len(s),
        "mean": float(s.mean()),
        "median": float(median),
        "sd": sd,
        "min": float(s.min()),
        "max": float(s.max()),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "skewness": float(skew(s)),
        "se": sd / math.sqrt(len(s)),
    }


def plot_distribution(ax, scores, title, bins=BINS):
    """Density histogram with a Gaussian fit."""
    mu, sigma = float(np.mean(scores)), float(np.std(scores, ddof=1))
    ax.hist(scores, bins=bins, density=True, color="lightsteelblue", edgecolor="white")
    xs = np.linspace(min(scores), max(scores), 400)
    ax.plot(xs, np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi)),
            color="crimson", linewidth=2)
    ax.axvline(mu, color="black", linestyle="--", linewidth=1)
    ax.text(0.97, 0.95, f"μ={mu:.3f}\nσ={sigma:.3f}\nn={len(scores)}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_title(title)
    ax.set_xlabel("Benchmark bias score")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)


def main():
    scores = load_scores(CSV_PATH)
    stats = describe(scores)
    tmp = STATS_CSV_PATH + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats))
        w.writeheader()
        w.writerow(stats)
    os.replace(tmp, STATS_CSV_PATH)

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_distribution(ax, scores, f"{constants['MF']}: benchmark bias score distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(constants["RUN_DIR"], "bell_curve.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
