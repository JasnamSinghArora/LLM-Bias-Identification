import os
import csv
import torch
from bias_subspace import load_CPD
from constants import constants
from helper import create_inputs, create_embedding, create_output_from_emb, get_bias_score_from_emb
from prompts import prompt_for_random_questions

K_MAX = 20
OUT_DIR = os.path.join(constants["RUN_DIR"], "out_k")
KEXP_PROMPTS_CACHE = os.path.join(constants["RUN_DIR"], "kexp_prompts.pt")
KEXP_PROGRESS_CACHE = os.path.join(constants["RUN_DIR"], "kexp_progress.pt")

def create_subspaces():
    diff_matrix = load_CPD().float().cpu()
    bias_mean = diff_matrix.mean(dim=0, keepdim=True)
    diff_matrix_centered = diff_matrix - bias_mean
    U, S, Vh = torch.linalg.svd(diff_matrix_centered, full_matrices=False)
    log_S = torch.log(S + 1e-8)
    log_drops = log_S[:-1] - log_S[1:]
    kopt = torch.argmax(log_drops[2:20]).item() + 3
    print(kopt)
    for k in range(1, K_MAX+1):
        bias_subspace = Vh[:k]
        os.makedirs(f"{OUT_DIR}/{k}", exist_ok=True)
        torch.save({"bias_subspace": bias_subspace, "bias_mean": bias_mean}, f"{OUT_DIR}/{k}/bias_subspace_cache.pt")
    return kopt

def create_bias_score_data():
    if os.path.exists(KEXP_PROMPTS_CACHE):
        items = torch.load(KEXP_PROMPTS_CACHE)
    else:
        items = create_inputs(prompt_for_random_questions, constants["TEMP"], constants["TOKENS"], "KEXP")
        torch.save(items, KEXP_PROMPTS_CACHE)
    inputs = [item for item in items if isinstance(item, str) and item.strip()]

    output_vecs = []
    if os.path.exists(KEXP_PROGRESS_CACHE):
        output_vecs = torch.load(KEXP_PROGRESS_CACHE)
        print(f"resuming k-experiment generation from prompt {len(output_vecs)}")
    already = len(output_vecs)
    for i, input in enumerate(inputs):
        if i < already:
            continue
        emb, mask = create_embedding(input)
        output_vecs.append(create_output_from_emb(emb, mask, with_grad=False).float().cpu())
        del emb, mask
        if (i + 1) % 200 == 0:
            torch.save(output_vecs, KEXP_PROGRESS_CACHE)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            print(f"k-experiment generation: {i + 1}/{len(inputs)}")
    output_center = torch.stack(output_vecs).mean(dim=0)

    for k in range(1, K_MAX+1):

        data = torch.load(f"{OUT_DIR}/{k}/bias_subspace_cache.pt")
        bias_subspace = data["bias_subspace"]

        with open(f"{OUT_DIR}/{k}/benchmark_scores.csv", "w", newline="") as f:
            csv.writer(f).writerow(["score"])

        for output_vec in output_vecs:
            benchmark_score = get_bias_score_from_emb([output_vec], bias_subspace, output_center).item()

            with open(f"{OUT_DIR}/{k}/benchmark_scores.csv", "a", newline="") as f:
                csv.writer(f).writerow([benchmark_score])
    if os.path.exists(KEXP_PROGRESS_CACHE):
        os.remove(KEXP_PROGRESS_CACHE)

def main():
    create_subspaces()
    create_bias_score_data()

if __name__ == "__main__":
    main()
