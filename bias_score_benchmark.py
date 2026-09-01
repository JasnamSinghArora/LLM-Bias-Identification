import csv
import os
import torch
from bias_subspace import load_bias_subspace

from helper import create_inputs, create_embedding, create_output_from_emb, get_bias_score_from_emb
from prompts import prompt_for_benchmark_questions
from constants import constants

BENCH_CSV_PATH = os.path.join(constants["RUN_DIR"], "benchmark_scores.csv")
CENTER_CACHE_PATH = os.path.join(constants["RUN_DIR"], "output_center_cache.pt")
PROMPTS_CACHE_PATH = os.path.join(constants["RUN_DIR"], "benchmark_prompts.pt")
PROGRESS_CACHE_PATH = os.path.join(constants["RUN_DIR"], "benchmark_progress.pt")

def set_bias_score_benchmark():
    bias_subspace, _ = load_bias_subspace()

    if os.path.exists(PROMPTS_CACHE_PATH):
        items = torch.load(PROMPTS_CACHE_PATH)
    else:
        items = create_inputs(prompt_for_benchmark_questions, constants["TEMP"], constants["TOKENS"], "BIAS")
        torch.save(items, PROMPTS_CACHE_PATH)
    output_vecs = []
    meta = []
    if os.path.exists(PROGRESS_CACHE_PATH):
        _p = torch.load(PROGRESS_CACHE_PATH)
        output_vecs, meta = _p["output_vecs"], _p["meta"]
        print(f"resuming benchmark generation from item {len(meta)}")
    already = len(meta)
    processed = 0
    for item in items:
        if not isinstance(item, str) or not item.strip():
            print(f"skipping malformed item: {item}")
            continue
        prompt_text = item
        processed += 1
        if processed <= already:
            continue

        emb, mask = create_embedding(prompt_text)
        output_vec = create_output_from_emb(emb, mask, with_grad=False).float().cpu()
        output_vecs.append(output_vec)
        meta.append(prompt_text)
        del emb, mask
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if processed % 200 == 0:
            torch.save({"output_vecs": output_vecs, "meta": meta}, PROGRESS_CACHE_PATH)
            print(f"benchmark generation: {processed} prompts done")

    output_center = torch.stack(output_vecs).mean(dim=0)

    with open(BENCH_CSV_PATH, "w", newline="") as f:
        f.write("prompt,score\n")

    total_score = 0
    score_count = 0
    for z, input in zip(output_vecs, meta):
        benchmark_score = get_bias_score_from_emb([z], bias_subspace, output_center).item()

        with open(BENCH_CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow([input, benchmark_score])

        total_score += benchmark_score
        score_count += 1

    baseline = total_score / score_count if score_count else 0
    torch.save({"output_center": output_center, "baseline": baseline}, CENTER_CACHE_PATH)
    if os.path.exists(PROGRESS_CACHE_PATH):
        os.remove(PROGRESS_CACHE_PATH)
    return baseline

if __name__ == "__main__":
    print(set_bias_score_benchmark())
