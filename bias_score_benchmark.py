import csv
import torch
from bias_subspace import load_bias_subspace

bias_subspace, _ = load_bias_subspace()
from helper import create_inputs, create_embedding, create_output_from_emb, get_bias_score_from_emb
from prompts import prompt_for_random_questions
from constants import constants

def set_bias_score_benchmark():
    ssf_set = set(constants["SSF"])

    items = create_inputs(prompt_for_random_questions, constants["TEMP"], constants["TOKENS"], "BIAS")
    output_vecs = []
    meta = []
    for item in items:
        if not isinstance(item, dict) or "prompt" not in item or "ssf" not in item:
            print(f"skipping malformed item: {item}")
            continue
        prompt_text = item["prompt"]
        ssf = item["ssf"]
        if ssf not in ssf_set:
            print(f"skipping item with unknown ssf '{ssf}': {prompt_text}")
            continue

        emb, mask = create_embedding(prompt_text)
        output_vec = create_output_from_emb(emb, mask, with_grad=False).float().cpu()
        output_vecs.append(output_vec)
        meta.append((ssf, prompt_text))
        del emb, mask
        torch.mps.empty_cache()

    output_center = torch.stack(output_vecs).mean(dim=0)

    with open("benchmark_scores.csv", "w", newline="") as f:
        f.write("ssf,prompt,score\n")

    total_score = 0
    score_count = 0
    for z, (ssf, input) in zip(output_vecs, meta):
        benchmark_score = get_bias_score_from_emb([z], bias_subspace, output_center).item()

        with open("benchmark_scores.csv", "a", newline="") as f:
            csv.writer(f).writerow([ssf, input, benchmark_score])

        total_score += benchmark_score
        score_count += 1

    baseline = total_score / score_count if score_count else 0
    torch.save({"output_center": output_center, "baseline": baseline}, "output_center_cache.pt")
    return baseline

print(set_bias_score_benchmark())