import torch
import os
import csv
from prompts import prompt_for_pairs
from helper import hidden_states, create_inputs
from constants import constants

CACHE_PATH = os.path.join(constants["RUN_DIR"], "bias_subspace_cache.pt")
CPD_CSV_PATH = os.path.join(constants["RUN_DIR"], "cpd_dataset.csv")
DIFF_MATRIX_PATH = os.path.join(constants["RUN_DIR"], "cpd_diff_matrix.pt")
DIFF_PROGRESS_PATH = os.path.join(constants["RUN_DIR"], "cpd_diff_progress.pt")
SINGULAR_VALUES_PATH = os.path.join(constants["RUN_DIR"], "singular_values.pt")

def create_bias_sentence_pairs():

    if os.path.exists(CPD_CSV_PATH):
        pairs = []
        with open(CPD_CSV_PATH, newline="") as f:
            for row in csv.DictReader(f):
                pairs.append((row["sentence_A"], row["sentence_B"]))
        print(f"loaded {len(pairs)} existing pairs from {CPD_CSV_PATH}")
        return pairs

    pairs = []
    items = create_inputs(prompt_for_pairs, 0.5, 70, "CPD")
    for item in items:
        if not isinstance(item, dict) or "sentence_A" not in item or "sentence_B" not in item:
            print(f"skipping malformed item: {item}")
            continue
        pairs.append((item["sentence_A"], item["sentence_B"]))

    with open(CPD_CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sentence_A", "sentence_B"])
        for s1, s2 in pairs:
            w.writerow([s1, s2])

    return pairs
     
def get_CPD():
    pairs = create_bias_sentence_pairs()
    diff_vectors = []
    if os.path.exists(DIFF_PROGRESS_PATH):
        diff_vectors = torch.load(DIFF_PROGRESS_PATH)
        print(f"resuming CPD embedding from pair {len(diff_vectors)}")
    start = len(diff_vectors)
    for i, (sentence1, sentence2) in enumerate(pairs):
        if i < start:
            continue

        hidden_states1 = hidden_states(sentence1)
        hidden_states2 = hidden_states(sentence2)

        final_layer1 = hidden_states1[constants["LAYER"]]
        final_layer2 = hidden_states2[constants["LAYER"]]

        vector1 = final_layer1.mean(dim=1) # takes average of embedings of all tokens - maybe replace with only the embedding of key word
        vector1 = vector1.squeeze()

        vector2 = final_layer2.mean(dim=1)
        vector2 = vector2.squeeze()

        d = vector1 - vector2
        diff_vectors.append(d.float().cpu())
        if (i + 1) % 1000 == 0:
            torch.save(diff_vectors, DIFF_PROGRESS_PATH)
            print(f"CPD embeddings: {i + 1}/{len(pairs)}")
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    diff_matrix = torch.stack(diff_vectors)
    torch.save(diff_matrix, DIFF_MATRIX_PATH)
    if os.path.exists(DIFF_PROGRESS_PATH):
        os.remove(DIFF_PROGRESS_PATH)
    return diff_matrix

def load_CPD():
    return torch.load(DIFF_MATRIX_PATH)

def create_bias_subspace_helper(k):

    diff_matrix = load_CPD().float().cpu()
    bias_mean = diff_matrix.mean(dim=0, keepdim=True)
    diff_matrix_centered = diff_matrix - bias_mean
    U, S, Vh = torch.linalg.svd(diff_matrix_centered, full_matrices=False)
    torch.save(S, SINGULAR_VALUES_PATH)

    if k is None:
        log_S = torch.log(S + 1e-8)
        log_drops = log_S[:-1] - log_S[1:]
        k = torch.argmax(log_drops[2:20]).item() + 3
        
    print(k)
    
    bias_subspace = Vh[:k]
    return bias_subspace, bias_mean

def load_bias_subspace():
    if os.path.exists(CACHE_PATH):
        data = torch.load(CACHE_PATH)
        return data["bias_subspace"], data["bias_mean"]
    
def create_bias_subspace(k=None, CACHE_PATH=CACHE_PATH):
    bias_subspace, bias_mean = create_bias_subspace_helper(k)
    torch.save({"bias_subspace": bias_subspace, "bias_mean": bias_mean}, CACHE_PATH)
    return bias_subspace, bias_mean

if __name__ == "__main__":
    _, _ = create_bias_subspace()