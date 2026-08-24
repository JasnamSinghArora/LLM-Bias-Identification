import os
import csv
import torch
from bias_subspace import create_bias_sentence_pairs
from helper import hidden_states, model

EPS = 1e-8
NUM_LAYERS = model.config.num_hidden_layers
LAYERS = range(1, NUM_LAYERS + 1, 1)

def main():
    pairs = create_bias_sentence_pairs()
    bias_captures = []
    h1 = {}
    h2 = {}
    for s1, s2 in pairs:
        h1[s1] = hidden_states(s1)
        h2[s2] = hidden_states(s2)

    n_train = int(0.75 * len(pairs))
    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train:]

    def create_diff_matrix(pair_list, layer):
        return torch.stack([
            h1[s1][layer].mean(dim=1).squeeze() - h2[s2][layer].mean(dim=1).squeeze()
            for s1, s2 in pair_list
        ]).float().cpu()

    for layer in LAYERS:
        # fit the subspace on the train split
        train_matrix = create_diff_matrix(train_pairs, layer)
        bias_mean = train_matrix.mean(dim=0, keepdim=True)
        train_centered = train_matrix - bias_mean
        U, S, Vh = torch.linalg.svd(train_centered, full_matrices=False)

        log_S = torch.log(S + 1e-8)
        log_drops = log_S[:-1] - log_S[1:]
        k = torch.argmax(log_drops[2:20]).item() + 3
        bias_subspace = Vh[:k]

        # BiasCapture Calculations (pooled, on the held-out test split)
        test_centered = create_diff_matrix(test_pairs, layer) - bias_mean
        coords = test_centered @ bias_subspace.T
        captured = coords.pow(2).sum()
        total = test_centered.pow(2).sum()
        bias_capture = (captured / (total + EPS)).item()
        bias_captures.append((layer, bias_capture))

        os.makedirs(f"out_layer/{layer}", exist_ok=True)
        torch.save({"bias_subspace": bias_subspace, "bias_mean": bias_mean}, f"out_layer/{layer}/bias_subspace_cache.pt")

    os.makedirs("out_layer", exist_ok=True)
    with open("out_layer/bias_capture.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "bias_capture"])
        writer.writerows(bias_captures)

main()
