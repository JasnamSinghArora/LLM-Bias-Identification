import csv
import math
from collections import Counter
from constants import constants

CSV_PATH = "cpd_dataset.csv"

def normalized_entropy(values):
    counts = Counter(values)
    m = sum(counts.values())
    r = len(counts)
    if m == 0 or r <= 1:
        return 0.0
    H = -sum((n / m) * math.log(n / m) for n in counts.values())
    return H / math.log(r)

# SSF terms label the data after generation; they are never used to generate it
def label(text):
    t = text.lower()
    for ssf in constants["SSF"]:
        if ssf.lower() in t:
            return ssf
    return "other"

def main(csv_path=CSV_PATH):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        labels = [label(row["sentence_A"]) for row in reader]

    H_norm = normalized_entropy(labels)
    print(f"H_norm(ssf) = {H_norm:.4f}")
    print(f"R_entropy(D) = {H_norm:.4f}")
    return H_norm

if __name__ == "__main__":
    main()
