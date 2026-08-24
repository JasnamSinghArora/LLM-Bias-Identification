import torch
import os
import csv
from prompts import prompt_for_pairs
from helper import hidden_states, create_inputs
from constants import constants

CACHE_PATH = os.path.join(constants["RUN_DIR"], "bias_subspace_cache.pt")  # EDITED: artifacts now live in the per-run directory
CPD_CSV_PATH = os.path.join(constants["RUN_DIR"], "cpd_dataset.csv")  # EDITED: per-run directory
DIFF_MATRIX_PATH = os.path.join(constants["RUN_DIR"], "cpd_diff_matrix.pt")  # ADDED: per-run path for the CPD difference matrix
DIFF_PROGRESS_PATH = os.path.join(constants["RUN_DIR"], "cpd_diff_progress.pt")  # ADDED: incremental checkpoint so a crash does not lose the embedding forward passes
SINGULAR_VALUES_PATH = os.path.join(constants["RUN_DIR"], "singular_values.pt")  # ADDED: spectrum saved for the paper's spectral-gap figure

def create_bias_sentence_pairs():

    if os.path.exists(CPD_CSV_PATH):  # ADDED: resume support — reuse the already-generated pairs instead of paying to regenerate them
        pairs = []  # ADDED
        with open(CPD_CSV_PATH, newline="") as f:  # ADDED
            for row in csv.DictReader(f):  # ADDED
                pairs.append((row["sentence_A"], row["sentence_B"]))  # ADDED
        print(f"loaded {len(pairs)} existing pairs from {CPD_CSV_PATH}")  # ADDED
        return pairs  # ADDED

    pairs = []
    ssfs = []
    items = create_inputs(prompt_for_pairs, 0.5, 70, "CPD")
    for item in items:
        if not isinstance(item, dict) or "sentence_A" not in item or "sentence_B" not in item or "ssf" not in item:
            print(f"skipping malformed item: {item}")
            continue
        if item["ssf"] not in constants["SSF"]:
            print(f"skipping item with unknown ssf")
            continue
        pairs.append((item["sentence_A"], item["sentence_B"]))
        ssfs.append(item["ssf"])

    with open(CPD_CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ssf", "sentence_A", "sentence_B"])
        for (s1, s2), ssf in zip(pairs, ssfs):
            w.writerow([ssf, s1, s2])

    return pairs
     
def get_CPD():
    pairs = create_bias_sentence_pairs()
    diff_vectors = []
    if os.path.exists(DIFF_PROGRESS_PATH):  # ADDED: resume a partially-computed diff matrix
        diff_vectors = torch.load(DIFF_PROGRESS_PATH)  # ADDED
        print(f"resuming CPD embedding from pair {len(diff_vectors)}")  # ADDED
    start = len(diff_vectors)  # ADDED: index of the first pair still to be embedded
    for i, (sentence1, sentence2) in enumerate(pairs):  # EDITED: enumerate so progress can be checkpointed and resumed
        if i < start:  # ADDED: already computed in a previous run
            continue  # ADDED

        hidden_states1 = hidden_states(sentence1)
        hidden_states2 = hidden_states(sentence2)

        final_layer1 = hidden_states1[constants["LAYER"]]
        final_layer2 = hidden_states2[constants["LAYER"]]

        vector1 = final_layer1.mean(dim=1) # takes average of embedings of all tokens - maybe replace with only the embedding of key word
        vector1 = vector1.squeeze()

        vector2 = final_layer2.mean(dim=1)
        vector2 = vector2.squeeze()

        d = vector1 - vector2
        diff_vectors.append(d.float().cpu())  # EDITED: moved to CPU float32 — keeping 100,000 vectors on MPS would exhaust GPU memory
        if (i + 1) % 1000 == 0:  # ADDED: periodic checkpoint + progress log
            torch.save(diff_vectors, DIFF_PROGRESS_PATH)  # ADDED
            print(f"CPD embeddings: {i + 1}/{len(pairs)}")  # ADDED
            if torch.backends.mps.is_available():  # ADDED
                torch.mps.empty_cache()  # ADDED

    diff_matrix = torch.stack(diff_vectors)
    torch.save(diff_matrix, DIFF_MATRIX_PATH)  # EDITED: per-run path (was hardcoded "cpd_diff_matrix.pt")
    if os.path.exists(DIFF_PROGRESS_PATH):  # ADDED: checkpoint no longer needed once the full matrix is saved
        os.remove(DIFF_PROGRESS_PATH)  # ADDED
    return diff_matrix

def load_CPD():  # ADDED: downstream steps load the saved matrix so the quality-checked dataset and the SVD input are identical (steps 6-8)
    return torch.load(DIFF_MATRIX_PATH)  # ADDED

def create_bias_subspace_helper(k):

    diff_matrix = load_CPD().float().cpu()  # EDITED: was get_CPD() — the SVD must run on the already-generated, quality-checked CPD matrix instead of regenerating a new dataset
    bias_mean = diff_matrix.mean(dim=0, keepdim=True)
    diff_matrix_centered = diff_matrix - bias_mean
    U, S, Vh = torch.linalg.svd(diff_matrix_centered, full_matrices=False)
    torch.save(S, SINGULAR_VALUES_PATH)  # ADDED: keep the singular value spectrum for the paper's spectral-gap figure

    if k is None:
        log_S = torch.log(S + 1e-8)
        log_drops = log_S[:-1] - log_S[1:]
        k = torch.argmax(log_drops[2:20]).item() + 3  # EDITED: unified the spectral-gap window with k_experiment.py (was log_drops[3:20] + 4)
        
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