import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

CSV_PATH = "k=5.csv"
BUCKET_SIZE = 3


def load_scores(path):
    with open(path) as f:
        return [float(line.strip()) for line in f if line.strip()]


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
    x_curve = np.linspace(edges[0], edges[-1], 400)
    pdf = (1 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-0.5 * ((x_curve - mu) / sigma) ** 2)
    y_curve = pdf * len(scores) * BUCKET_SIZE

    _, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(centers, counts, color="steelblue", edgecolor="black",
               s=60, zorder=3, label="Frequency")
    ax.plot(x_curve, y_curve, color="crimson", linewidth=2, label="Normal fit")

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
    plt.savefig("bell_curve.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
