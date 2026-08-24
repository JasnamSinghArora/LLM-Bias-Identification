import os
import csv
import torch
from bias_subspace import load_CPD  # EDITED: was create_bias_sentence_pairs/get_CPD — the candidate subspaces must come from the saved, quality-checked CPD matrix instead of regenerating a dataset
from constants import constants
from helper import create_inputs, create_embedding, create_output_from_emb, get_bias_score_from_emb  # EDITED: switched to the embedding-based scoring path so k-scores are measured identically to the benchmark and the optimizer (was create_outputs_from_text/get_bias_score_from_text/hidden_states)
from prompts import prompt_for_random_questions

K_MAX = 20
OUT_DIR = os.path.join(constants["RUN_DIR"], "out_k")  # ADDED: per-run output directory (was the hardcoded "out_k" below)
KEXP_PROMPTS_CACHE = os.path.join(constants["RUN_DIR"], "kexp_prompts.pt")  # ADDED: the Y prompts are frozen here for resume support
KEXP_PROGRESS_CACHE = os.path.join(constants["RUN_DIR"], "kexp_progress.pt")  # ADDED: incremental checkpoint for the long generation loop

def create_subspaces():
    diff_matrix = load_CPD().float().cpu()  # EDITED: was get_CPD() — do not regenerate the dataset (SVD runs once on the final CPD matrix, step 8)
    bias_mean = diff_matrix.mean(dim=0, keepdim=True)
    diff_matrix_centered = diff_matrix - bias_mean
    U, S, Vh = torch.linalg.svd(diff_matrix_centered, full_matrices=False)
    log_S = torch.log(S + 1e-8)
    log_drops = log_S[:-1] - log_S[1:]
    kopt = torch.argmax(log_drops[2:20]).item() + 3
    print(kopt)
    for k in range(1, K_MAX+1):
        bias_subspace = Vh[:k]
        os.makedirs(f"{OUT_DIR}/{k}", exist_ok=True)  # EDITED: per-run path
        torch.save({"bias_subspace": bias_subspace, "bias_mean": bias_mean}, f"{OUT_DIR}/{k}/bias_subspace_cache.pt")  # EDITED: per-run path
    return kopt  # ADDED: returned so final_pipeline.py can record the spectral-gap choice

def create_bias_score_data():
    if os.path.exists(KEXP_PROMPTS_CACHE):  # ADDED: resume — reuse the frozen Y prompts instead of regenerating
        items = torch.load(KEXP_PROMPTS_CACHE)  # ADDED
    else:  # ADDED
        items = create_inputs(prompt_for_random_questions, constants["TEMP"], constants["TOKENS"], "KEXP")  # EDITED: kind "KEXP" (Y = 20,000 prompts; was kind "BIAS")
        torch.save(items, KEXP_PROMPTS_CACHE)  # ADDED
    inputs = [item["prompt"] for item in items if isinstance(item, dict) and "prompt" in item]  # ADDED: prompt_for_random_questions returns {prompt, ssf} objects; the old code assumed raw strings

    output_vecs = []  # EDITED: embedding-based generation replaces create_outputs_from_text (see the import note above)
    if os.path.exists(KEXP_PROGRESS_CACHE):  # ADDED: resume a partially-completed generation loop
        output_vecs = torch.load(KEXP_PROGRESS_CACHE)  # ADDED
        print(f"resuming k-experiment generation from prompt {len(output_vecs)}")  # ADDED
    already = len(output_vecs)  # ADDED: number of prompts completed in previous runs
    for i, input in enumerate(inputs):  # ADDED
        if i < already:  # ADDED: already computed in a previous run
            continue  # ADDED
        emb, mask = create_embedding(input)  # ADDED
        output_vecs.append(create_output_from_emb(emb, mask, with_grad=False).float().cpu())  # ADDED
        del emb, mask  # ADDED
        if (i + 1) % 200 == 0:  # ADDED: periodic checkpoint + progress log
            torch.save(output_vecs, KEXP_PROGRESS_CACHE)  # ADDED
            if torch.backends.mps.is_available():  # ADDED
                torch.mps.empty_cache()  # ADDED
            print(f"k-experiment generation: {i + 1}/{len(inputs)}")  # ADDED
    output_center = torch.stack(output_vecs).mean(dim=0)  # EDITED: center computed from the embedding-path output vectors (was text-output hidden states)

    for k in range(1, K_MAX+1):

        data = torch.load(f"{OUT_DIR}/{k}/bias_subspace_cache.pt")  # EDITED: per-run path
        bias_subspace = data["bias_subspace"]

        with open(f"{OUT_DIR}/{k}/benchmark_scores.csv", "w", newline="") as f:  # EDITED: per-run path
            csv.writer(f).writerow(["score"])

        for output_vec in output_vecs:  # EDITED: iterate the embedding-path output vectors (was text outputs with an empty-string skip)
            benchmark_score = get_bias_score_from_emb([output_vec], bias_subspace, output_center).item()  # EDITED: embedding-based scorer (was get_bias_score_from_text)

            with open(f"{OUT_DIR}/{k}/benchmark_scores.csv", "a", newline="") as f:  # EDITED: per-run path
                csv.writer(f).writerow([benchmark_score])
    if os.path.exists(KEXP_PROGRESS_CACHE):  # ADDED: checkpoint no longer needed once all scores are written
        os.remove(KEXP_PROGRESS_CACHE)  # ADDED

def main():
    create_subspaces()
    create_bias_score_data()

if __name__ == "__main__":  # ADDED: guard so final_pipeline.py can import this module without triggering a full run
    main()  # EDITED: was a bare module-level call
