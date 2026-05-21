import torch
from constants import constants
from prompts import prompt_for_initial_questions, prompt_for_random_questions
from helper import create_inputs, create_output_from_emb, create_outputs_from_text, get_bias_score_from_emb, get_bias_score_from_text, create_embedding
from bias_subspace import bias_subspace, bias_mean

def set_bias_score_benchmark(N):
    global bias_subspace
    total_score = 0
    score_count = 0
    
    for i in range(N):
        inputs = create_inputs(prompt_for_random_questions, 0.6, 40)
        outputs = create_outputs_from_text(inputs)
        chunks = [outputs[i:i+5] for i in range(0, constants["BATCH_SIZE"] * constants["BATCHES"], 5)]
        
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            benchmark_score = get_bias_score_from_text(chunk, bias_subspace, bias_mean).item()

            with open("benchmark_scores.csv", "a") as f:
                f.write(f"{benchmark_score}\n")

            total_score += benchmark_score
            score_count += 1
            
    if score_count == 0:
        return 0
    return total_score / score_count

def create_optimized_inputs():
    global bias_subspace
    inputs = create_inputs(prompt_for_initial_questions, 0.4, 40)
    print("Inital Input Created")
    opt_input_embeddings = []
    attention_masks = []
    for input in inputs:
        emb, attention_mask = create_embedding(input)
        emb = optimize_with_gradient_ascent(emb, emb, attention_mask)
        opt_input_embeddings.append(emb)
        attention_masks.append(attention_mask)
        
    return opt_input_embeddings, attention_masks
    
def optimize_with_gradient_ascent(og_emb, current_emb, attention_mask, min_update_per_step=0.02):
    current_emb = current_emb.detach().clone().requires_grad_(True)
    device = current_emb.device
    bias_mean_device = bias_mean.squeeze().to(device)
    bias_subspace_device = bias_subspace.to(device)

    while (True):
        # project
        output_vec = create_output_from_emb(current_emb, attention_mask, False)

        # calculate gradient
        centered_vec = output_vec - bias_mean_device
        coords = bias_subspace_device @ centered_vec
        bias_score = torch.norm(coords)
        gt = torch.autograd.grad(
                        bias_score,
                        current_emb,
                        retain_graph=False,
                        create_graph=False
                    )[0]

        with torch.no_grad():
            # calculate learning rate
            delta = current_emb - og_emb
            direction = gt / (torch.norm(gt) + 1e-12)
            epsilon = 1.0
            inner = torch.sum(delta * direction)
            delta_norm_sq = torch.sum(delta * delta)
            discriminant = inner**2 + epsilon**2 - delta_norm_sq
            discriminant = torch.clamp(discriminant, min=0.0)
            eta_max = -inner + torch.sqrt(discriminant)
            
            # change embedding
            new_emb = current_emb + eta_max * direction
            
            # check epsilon limit
            overall_delta = new_emb - og_emb
            overall_delta_norm = torch.norm(overall_delta)
            
            step_delta_norm = eta_max.abs()
        
        print("Embedding Changed")
        
        if torch.isnan(overall_delta_norm):
            return current_emb.detach()
        elif overall_delta_norm > epsilon:
            overall_delta = epsilon * overall_delta / overall_delta_norm
            return (og_emb + overall_delta).detach()
        elif step_delta_norm < min_update_per_step:
            return new_emb.detach()
        else:
            current_emb = new_emb.detach().requires_grad_(True)
    
def main():
    inputs, attention_masks = create_optimized_inputs()
    outputs = []
    output_vecs = []
    for input, attention_mask in zip(inputs, attention_masks):
        output_vec, output = create_output_from_emb(input, attention_mask, True)
        outputs.append(output)
        output_vecs.append(output_vec)
        print(output)
    opt_bias_score = get_bias_score_from_emb(output_vecs, bias_subspace, bias_mean)
    BASELINE_BIAS_SCORE = 100.84
    print("BASELINE")
    print(BASELINE_BIAS_SCORE)
    print("OPTIMIZED")
    print(opt_bias_score)
    
main()