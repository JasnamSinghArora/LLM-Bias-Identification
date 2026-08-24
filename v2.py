import torch
import time
from constants import constants
from prompts import prompt_for_initial_questions, prompt_for_random_questions
from helper import create_inputs, create_output_from_emb, create_outputs_from_text, get_bias_score_from_emb, get_bias_score_from_text, create_embedding, get_optimized_epsilon
from bias_subspace import load_bias_subspace

bias_subspace, _ = load_bias_subspace()
output_center = torch.load("output_center_cache.pt")["output_center"]

def set_bias_score_benchmark(N):
    global bias_subspace
    total_score = 0
    score_count = 0
    
    for i in range(N):
        inputs = create_inputs(prompt_for_random_questions, constants["TEMP"], constants["TOKENS"], "BIAS")
        outputs = create_outputs_from_text(inputs)
        chunks = [outputs[i:i+5] for i in range(0, constants["BATCH_SIZE_BIAS"] * constants["BATCHES_BIAS"], 5)]
        
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            benchmark_score = get_bias_score_from_text(chunk, bias_subspace, output_center).item()

            with open("benchmark_scores.csv", "a") as f:
                f.write(f"{benchmark_score}\n")

            total_score += benchmark_score
            score_count += 1
            
    if score_count == 0:
        return 0
    return total_score / score_count

def create_optimized_inputs():
    global bias_subspace
    inputs = create_inputs(prompt_for_initial_questions, constants["TEMP"], constants["TOKENS"], "BIAS")
    print("Inital Input Created")
    opt_input_embeddings = []
    attention_masks = []
    for input in inputs:
        emb, attention_mask = create_embedding(input)
        emb = optimize_with_gradient_ascent(emb, emb, attention_mask)
        opt_input_embeddings.append(emb)
        attention_masks.append(attention_mask)
        
    return opt_input_embeddings, attention_masks
    
def optimize_with_gradient_ascent(og_emb, current_emb, attention_mask, min_update_per_step=0.3):
    current_emb = current_emb.detach().clone().float().requires_grad_(True)
    device = current_emb.device
    center_device = output_center.squeeze().to(device)
    bias_subspace_device = bias_subspace.to(device)
    projection_matrix = bias_subspace_device.T @ bias_subspace_device
    I = torch.eye(projection_matrix.shape[0], device=device)
    def M(e):
        out = create_output_from_emb(e, attention_mask, False).float()
        return (I - projection_matrix) @ out

    while (True):
        # project 
        output_vec = create_output_from_emb(current_emb, attention_mask, False).float()
        
        # calculate gradient
        centered_vec = output_vec - center_device
        coords = bias_subspace_device @ centered_vec
        bias_score = torch.norm(coords)
        gt = torch.autograd.grad(
                        bias_score,
                        current_emb,
                        retain_graph=False,
                        create_graph=False
                    )[0]
        
        # Semantic Preservation Calculations
        t0 = time.time()
        Jt = torch.autograd.functional.jacobian(M, current_emb, vectorize=True)
        gt_flat = gt.reshape(-1)
        J = Jt.reshape(Jt.shape[0], -1)
        JJT = J @ J.T
        pt_flat = gt_flat - J.T @ (torch.linalg.pinv(JJT) @ (J @ gt_flat))
        pt = pt_flat.reshape_as(gt)
        t1 = time.time()
        print ("Semantic Preservation Calculations in", t1-t0)
        
        with torch.no_grad():
            # calculate epsilon 
            direction = pt / (torch.norm(pt) + 1e-12)
            epsilon = get_optimized_epsilon(current_emb, direction, attention_mask, center_device, bias_subspace_device)
            print("epsilon calculated in", time.time()-t1)
            
            # change embedding
            new_emb = current_emb + epsilon * direction
            
            step_delta_norm = epsilon
        
        print("Embedding Changed by", step_delta_norm)
        
        if torch.isnan(new_emb).any():
            return current_emb.detach()
        elif step_delta_norm < min_update_per_step:
            return new_emb.detach()
        else:
            current_emb = new_emb.detach().requires_grad_(True)
    
def main():
    inputs, attention_masks = create_optimized_inputs()
    outputs = []
    output_vecs = []
    for input, attention_mask in zip(inputs, attention_masks):
        #output_vec, output = create_output_from_emb(input, attention_mask, True)
        output_vec = create_output_from_emb(input, attention_mask, False)
        #outputs.append(output)
        output_vecs.append(output_vec)
        #print(output)
    opt_bias_score = get_bias_score_from_emb(output_vecs, bias_subspace, output_center)
    BASELINE_BIAS_SCORE = 100.84
    print("BASELINE")
    print(BASELINE_BIAS_SCORE)
    print("OPTIMIZED")
    print(opt_bias_score)
    
main()