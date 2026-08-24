import os
import csv
import torch
from bias_subspace import create_bias_sentence_pairs, get_CPD
from constants import constants
from helper import create_inputs, create_outputs_from_text, get_bias_score_from_text, hidden_states
from prompts import prompt_for_random_questions

K_MAX = 20

def create_subspaces():
    diff_matrix = get_CPD().float().cpu()
    bias_mean = diff_matrix.mean(dim=0, keepdim=True)
    diff_matrix_centered = diff_matrix - bias_mean
    U, S, Vh = torch.linalg.svd(diff_matrix_centered, full_matrices=False)
    log_S = torch.log(S + 1e-8)
    log_drops = log_S[:-1] - log_S[1:] 
    kopt = torch.argmax(log_drops[2:20]).item() + 3
    print(kopt)
    for k in range(1, K_MAX+1):  
        bias_subspace = Vh[:k]
        os.makedirs(f"out_k/{k}", exist_ok=True)
        torch.save({"bias_subspace": bias_subspace, "bias_mean": bias_mean}, f"out_k/{k}/bias_subspace_cache.pt")
    
def create_bias_score_data():
    inputs = create_inputs(prompt_for_random_questions, constants["TEMP"], constants["TOKENS"], "BIAS")
    outputs = create_outputs_from_text(inputs)

    output_embs = [
        hidden_states(output)[constants["LAYER"]].mean(dim=1).squeeze().float().cpu()
        for output in outputs if output != ""
    ]
    output_center = torch.stack(output_embs).mean(dim=0)

    for k in range(1, K_MAX+1):

        data = torch.load(f"out_k/{k}/bias_subspace_cache.pt")
        bias_subspace = data["bias_subspace"]

        with open(f"out_k/{k}/benchmark_scores.csv", "w", newline="") as f:
            csv.writer(f).writerow(["score"])
                
        for j, output in enumerate(outputs):
            if output == "":
                continue
            benchmark_score = get_bias_score_from_text([output], bias_subspace, output_center).item()

            with open(f"out_k/{k}/benchmark_scores.csv", "a", newline="") as f:
                csv.writer(f).writerow([benchmark_score])
                
def main():
    create_subspaces()
    create_bias_score_data()
    
main()