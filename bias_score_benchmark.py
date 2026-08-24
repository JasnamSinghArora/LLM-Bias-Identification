import csv
import os  # ADDED: needed for the per-run paths and resume checks
import torch
from bias_subspace import load_bias_subspace

from helper import create_inputs, create_embedding, create_output_from_emb, get_bias_score_from_emb
from prompts import prompt_for_benchmark_questions  # EDITED: was prompt_for_random_questions — the benchmark now runs on the Z bias-eliciting prompts (pipeline step 16)
from constants import constants

BENCH_CSV_PATH = os.path.join(constants["RUN_DIR"], "benchmark_scores.csv")  # ADDED: per-run path (was hardcoded "benchmark_scores.csv" below)
CENTER_CACHE_PATH = os.path.join(constants["RUN_DIR"], "output_center_cache.pt")  # ADDED: per-run path (was hardcoded "output_center_cache.pt" below)
PROMPTS_CACHE_PATH = os.path.join(constants["RUN_DIR"], "benchmark_prompts.pt")  # ADDED: the Z prompts are frozen here so optimization reuses the exact same prompts (step 21)
PROGRESS_CACHE_PATH = os.path.join(constants["RUN_DIR"], "benchmark_progress.pt")  # ADDED: incremental checkpoint for the long generation loop

def set_bias_score_benchmark():
    bias_subspace, _ = load_bias_subspace()  # EDITED: moved inside the function (was module level, which crashed on import before the subspace cache existed)
    ssf_set = set(constants["SSF"])

    if os.path.exists(PROMPTS_CACHE_PATH):  # ADDED: resume — reuse the frozen Z prompt set instead of regenerating
        items = torch.load(PROMPTS_CACHE_PATH)  # ADDED
    else:  # ADDED
        items = create_inputs(prompt_for_benchmark_questions, constants["TEMP"], constants["TOKENS"], "BIAS")  # EDITED: prompt source is now the bias-eliciting benchmark prompt
        torch.save(items, PROMPTS_CACHE_PATH)  # ADDED: freeze the Z prompts before any scoring
    output_vecs = []
    meta = []
    if os.path.exists(PROGRESS_CACHE_PATH):  # ADDED: resume a partially-completed generation loop
        _p = torch.load(PROGRESS_CACHE_PATH)  # ADDED
        output_vecs, meta = _p["output_vecs"], _p["meta"]  # ADDED
        print(f"resuming benchmark generation from item {len(meta)}")  # ADDED
    already = len(meta)  # ADDED: number of valid items completed in previous runs
    processed = 0  # ADDED: counts valid items so resume knows which to skip
    for item in items:
        if not isinstance(item, dict) or "prompt" not in item or "ssf" not in item:
            print(f"skipping malformed item: {item}")
            continue
        prompt_text = item["prompt"]
        ssf = item["ssf"]
        if ssf not in ssf_set:
            print(f"skipping item with unknown ssf '{ssf}': {prompt_text}")
            continue
        processed += 1  # ADDED
        if processed <= already:  # ADDED: already computed in a previous run
            continue  # ADDED

        emb, mask = create_embedding(prompt_text)
        output_vec = create_output_from_emb(emb, mask, with_grad=False).float().cpu()
        output_vecs.append(output_vec)
        meta.append((ssf, prompt_text))
        del emb, mask
        torch.mps.empty_cache()
        if processed % 200 == 0:  # ADDED: periodic checkpoint + progress log
            torch.save({"output_vecs": output_vecs, "meta": meta}, PROGRESS_CACHE_PATH)  # ADDED
            print(f"benchmark generation: {processed} prompts done")  # ADDED

    output_center = torch.stack(output_vecs).mean(dim=0)

    with open(BENCH_CSV_PATH, "w", newline="") as f:  # EDITED: per-run path
        f.write("ssf,prompt,score\n")

    total_score = 0
    score_count = 0
    for z, (ssf, input) in zip(output_vecs, meta):
        benchmark_score = get_bias_score_from_emb([z], bias_subspace, output_center).item()

        with open(BENCH_CSV_PATH, "a", newline="") as f:  # EDITED: per-run path
            csv.writer(f).writerow([ssf, input, benchmark_score])

        total_score += benchmark_score
        score_count += 1

    baseline = total_score / score_count if score_count else 0
    torch.save({"output_center": output_center, "baseline": baseline}, CENTER_CACHE_PATH)  # EDITED: per-run path
    if os.path.exists(PROGRESS_CACHE_PATH):  # ADDED: checkpoint no longer needed once everything is saved
        os.remove(PROGRESS_CACHE_PATH)  # ADDED
    return baseline

if __name__ == "__main__":  # ADDED: guard so final_pipeline.py can import this module without triggering a full run
    print(set_bias_score_benchmark())  # EDITED: was a bare module-level call
