import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from constants import constants

CSV_PATH = os.path.join(constants["RUN_DIR"], "benchmark_scores.csv")
STATS_CSV_PATH = os.path.join(constants["RUN_DIR"], "benchmark_stats.csv")
BUCKET_SIZE = 7


def save_stats(path, n, mu, sigma, se):
    exists = os.path.exists(path)
    serial = 1
    if exists:
        with open(path) as f:
            serial = sum(1 for _ in f)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["serial", "n", "mu", "sigma", "se"])
        writer.writerow([serial, n, f"{mu:.2f}", f"{sigma:.2f}", f"{se:.2f}"])


def load_scores(path):
    with open(path, newline="") as f:
        return [float(row["score"]) for row in csv.DictReader(f) if row.get("score", "").strip()]


def bucket_scores(scores, size):
    lo = (int(min(scores)) // size) * size
    hi = (int(max(scores)) // size + 1) * size
    edges = list(range(lo, hi + 1, size))
    counts = [0] * (len(edges) - 1)
    for s in scores:
        idx = min(int((s - lo) // size), len(counts) - 1)
        counts[idx] += 1
    labels = [f"{edges[i]}-{edges[i + 1] - 1}" for i in range(len(counts))]
    centers = [(edges[i] + edges[i + 1] - 1) / 2 for i in range(len(counts))]
    return labels, centers, counts, edges


def main():
    scores = load_scores(CSV_PATH)
    labels, centers, counts, edges = bucket_scores(scores, BUCKET_SIZE)

    mu = float(np.mean(scores))
    sigma = float(np.std(scores))
    se = sigma / math.sqrt(len(scores))
    save_stats(STATS_CSV_PATH, len(scores), mu, sigma, se)
    x_curve = np.linspace(edges[0], edges[-1], 400)
    pdf = (1 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x_curve - mu) / sigma) ** 2)
    y_curve = pdf * len(scores) * BUCKET_SIZE

    _, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(centers, counts, color="steelblue", edgecolor="black",
               s=60, zorder=3, label="Frequency")
    ax.plot(x_curve, y_curve, color="crimson", linewidth=2, label="Normal fit")

    if len(centers) > 3:
        spline = make_interp_spline(centers, counts, k=3)
        x_smooth = np.linspace(min(centers), max(centers), 400)
        y_smooth = spline(x_smooth)
        ax.plot(x_smooth, y_smooth, color="seagreen", linewidth=2,
                linestyle="--", label="Connecting curve")

    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Benchmark Score Bell Curve (μ={mu:.2f}, σ={sigma:.2f})")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(constants["RUN_DIR"], "bell_curve.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
