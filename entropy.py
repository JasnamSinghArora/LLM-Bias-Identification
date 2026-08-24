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

def main(csv_path=CSV_PATH):  # EDITED: takes the csv path and returns the value so final_pipeline.py can call it per run
    with open(csv_path, newline="") as f:  # EDITED: parameterized path (was the hardcoded CSV_PATH)
        reader = csv.DictReader(f)
        ssfs = [row["ssf"].strip() for row in reader]

    H_norm = normalized_entropy(ssfs)
    print(f"H_norm(ssf) = {H_norm:.4f}")
    print(f"R_entropy(D) = {H_norm:.4f}")
    return H_norm  # ADDED: returned to the pipeline for the step-7 quality gate

if __name__ == "__main__":  # ADDED: guard so importing this module no longer runs it
    main()  # EDITED: was a bare module-level call
