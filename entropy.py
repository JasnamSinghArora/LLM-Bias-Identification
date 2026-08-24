import csv
import math
from collections import Counter

CSV_PATH = "cpd_dataset.csv"

def normalized_entropy(values):
    counts = Counter(values)
    m = sum(counts.values())
    r = len(counts)
    if m == 0 or r <= 1:
        return 0.0
    H = -sum((n / m) * math.log(n / m) for n in counts.values())
    return H / math.log(r)

def main():
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        ssfs = [row["ssf"].strip() for row in reader]

    H_norm = normalized_entropy(ssfs)
    print(f"H_norm(ssf) = {H_norm:.4f}")
    print(f"R_entropy(D) = {H_norm:.4f}")

main()
