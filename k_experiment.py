import os
import csv
import time
import torch
from bias_subspace import load_CPD
from constants import constants
from helper import create_inputs, create_embedding, create_output_from_emb, get_bias_score_from_emb
from prompts import prompt_for_random_questions
from runlog import log, atomic_save, Progress, fmt_duration

K_MAX = 20
SAVE_EVERY = 25  # prompts between checkpoints during generation
OUT_DIR = os.path.join(constants["RUN_DIR"], "out_k")
KEXP_PROMPTS_CACHE = os.path.join(constants["RUN_DIR"], "kexp_prompts.pt")
KEXP_PROGRESS_CACHE = os.path.join(constants["RUN_DIR"], "kexp_progress.pt")
KEXP_OUTPUT_VECS = os.path.join(constants["RUN_DIR"], "kexp_output_vecs.pt")

def create_subspaces():
    diff_matrix = load_CPD().float().cpu()
    bias_mean = diff_matrix.mean(dim=0, keepdim=True)
    diff_matrix_centered = diff_matrix - bias_mean
    log(f"SVD of the centered difference matrix {tuple(diff_matrix.shape)} (this can take a few minutes)")
    t0 = time.time()
    U, S, Vh = torch.linalg.svd(diff_matrix_centered, full_matrices=False)
    log_S = torch.log(S + 1e-8)
    log_drops = log_S[:-1] - log_S[1:]
    kopt = torch.argmax(log_drops[2:20]).item() + 3
    log(f"SVD finished in {fmt_duration(time.time() - t0)}; spectral-gap estimate k = {kopt}")
    for k in range(1, K_MAX+1):
        bias_subspace = Vh[:k].clone()  # clone: saving a view would write the whole 4096x4096 matrix
        os.makedirs(f"{OUT_DIR}/{k}", exist_ok=True)
        atomic_save({"bias_subspace": bias_subspace, "bias_mean": bias_mean}, f"{OUT_DIR}/{k}/bias_subspace_cache.pt")
    log(f"saved candidate subspaces k=1..{K_MAX} -> {OUT_DIR}")
    return kopt

def create_bias_score_data():
    if os.path.exists(KEXP_PROMPTS_CACHE):
        items = torch.load(KEXP_PROMPTS_CACHE)
    else:
        items = create_inputs(prompt_for_random_questions, constants["TEMP"], constants["TOKENS"], "KEXP")
        atomic_save(items, KEXP_PROMPTS_CACHE)
    inputs = [item for item in items if isinstance(item, str) and item.strip()]
    if len(inputs) != len(items):
        log(f"skipping {len(items) - len(inputs)} malformed k-experiment prompts")

    # generate one output per prompt, checkpointing as we go
    output_vecs = []
    for path in (KEXP_OUTPUT_VECS, KEXP_PROGRESS_CACHE):
        if os.path.exists(path):
            output_vecs = torch.load(path)
            break
    if len(output_vecs) > len(inputs):
        log("saved progress holds more outputs than there are prompts; starting the generation over")
        output_vecs = []
    prog = Progress("k-experiment generation", len(inputs), done=len(output_vecs))
    for i in range(len(output_vecs), len(inputs)):
        emb, mask = create_embedding(inputs[i])
        output_vecs.append(create_output_from_emb(emb, mask, with_grad=False).float().cpu())
        del emb, mask
        prog.step()
        if len(output_vecs) % SAVE_EVERY == 0 or len(output_vecs) == len(inputs):
            atomic_save(output_vecs, KEXP_PROGRESS_CACHE)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    output_center = torch.stack(output_vecs).mean(dim=0)

    # score every output under each candidate subspace
    log(f"scoring {len(output_vecs)} outputs under k=1..{K_MAX}")
    for k in range(1, K_MAX+1):
        data = torch.load(f"{OUT_DIR}/{k}/bias_subspace_cache.pt")
        bias_subspace = data["bias_subspace"]
        scores = [get_bias_score_from_emb([v], bias_subspace, output_center).item() for v in output_vecs]
        tmp = f"{OUT_DIR}/{k}/benchmark_scores.csv.tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["score"])
            for s in scores:
                w.writerow([s])
        os.replace(tmp, f"{OUT_DIR}/{k}/benchmark_scores.csv")
        log(f"  k={k:>2}: mean bias score {sum(scores) / len(scores):.4f}")

    atomic_save(output_vecs, KEXP_OUTPUT_VECS)
    if os.path.exists(KEXP_PROGRESS_CACHE):
        os.remove(KEXP_PROGRESS_CACHE)
    log(f"k-experiment scores saved -> {OUT_DIR}/<k>/benchmark_scores.csv")

def main():
    create_subspaces()
    create_bias_score_data()

if __name__ == "__main__":
    main()
