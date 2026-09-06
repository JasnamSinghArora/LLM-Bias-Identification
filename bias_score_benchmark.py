import csv
import os
import torch
from bias_subspace import load_bias_subspace

from helper import create_inputs, create_embedding, create_output_from_emb, get_bias_score_from_emb
from prompts import prompt_for_benchmark_questions
from constants import constants
from runlog import log, atomic_save, Progress

SAVE_EVERY = 25  # prompts between checkpoints during generation
BENCH_CSV_PATH = os.path.join(constants["RUN_DIR"], "benchmark_scores.csv")
CENTER_CACHE_PATH = os.path.join(constants["RUN_DIR"], "output_center_cache.pt")
PROMPTS_CACHE_PATH = os.path.join(constants["RUN_DIR"], "benchmark_prompts.pt")
PROGRESS_CACHE_PATH = os.path.join(constants["RUN_DIR"], "benchmark_progress.pt")
OUTPUT_VECS_PATH = os.path.join(constants["RUN_DIR"], "benchmark_output_vecs.pt")

def set_bias_score_benchmark():
    bias_subspace, _ = load_bias_subspace()

    if os.path.exists(PROMPTS_CACHE_PATH):
        items = torch.load(PROMPTS_CACHE_PATH)
    else:
        items = create_inputs(prompt_for_benchmark_questions, constants["TEMP"], constants["TOKENS"], "BIAS")
        atomic_save(items, PROMPTS_CACHE_PATH)
    prompts = [item for item in items if isinstance(item, str) and item.strip()]
    if len(prompts) != len(items):
        log(f"skipping {len(items) - len(prompts)} malformed benchmark prompts")

    # generate one output per prompt, checkpointing as we go
    output_vecs, meta = [], []
    for path in (OUTPUT_VECS_PATH, PROGRESS_CACHE_PATH):
        if os.path.exists(path):
            _p = torch.load(path)
            output_vecs, meta = _p["output_vecs"], _p["meta"]
            break
    if meta and meta != prompts[:len(meta)]:
        log("saved progress does not match the prompt list; starting the benchmark generation over")
        output_vecs, meta = [], []
    prog = Progress("benchmark generation", len(prompts), done=len(meta))
    for i in range(len(meta), len(prompts)):
        emb, mask = create_embedding(prompts[i])
        output_vecs.append(create_output_from_emb(emb, mask, with_grad=False).float().cpu())
        meta.append(prompts[i])
        del emb, mask
        prog.step()
        if len(meta) % SAVE_EVERY == 0 or len(meta) == len(prompts):
            atomic_save({"output_vecs": output_vecs, "meta": meta}, PROGRESS_CACHE_PATH)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    # score every output against the frozen subspace and freeze the baseline
    output_center = torch.stack(output_vecs).mean(dim=0)
    log(f"scoring {len(output_vecs)} benchmark outputs")
    scores = [get_bias_score_from_emb([z], bias_subspace, output_center).item() for z in output_vecs]
    tmp = BENCH_CSV_PATH + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "score"])
        for prompt_text, s in zip(meta, scores):
            w.writerow([prompt_text, s])
    os.replace(tmp, BENCH_CSV_PATH)

    baseline = sum(scores) / len(scores) if scores else 0
    atomic_save({"output_vecs": output_vecs, "meta": meta}, OUTPUT_VECS_PATH)
    atomic_save({"output_center": output_center, "baseline": baseline}, CENTER_CACHE_PATH)
    if os.path.exists(PROGRESS_CACHE_PATH):
        os.remove(PROGRESS_CACHE_PATH)
    log(f"baseline bias score = {baseline:.4f} over {len(scores)} prompts; saved {BENCH_CSV_PATH}")
    return baseline

if __name__ == "__main__":
    print(set_bias_score_benchmark())
