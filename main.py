import torch
import time
import os
import csv
from constants import constants
from helper import create_output_from_emb, get_bias_score_from_emb, create_embedding, get_optimized_epsilon, create_inputs_with_grad
from bias_subspace import load_bias_subspace

bias_subspace = None
output_center = None
BASELINE_BIAS_SCORE = None

def init_optimizer_state():
    global bias_subspace, output_center, BASELINE_BIAS_SCORE
    bias_subspace, _ = load_bias_subspace()
    _c = torch.load(os.path.join(constants["RUN_DIR"], "output_center_cache.pt"))
    output_center = _c["output_center"]
    BASELINE_BIAS_SCORE = _c["baseline"]

def create_optimized_inputs(prompts):
    global bias_subspace
    inputs = prompts
    print("Inital Input Created")
    opt_input_embeddings = []
    attention_masks = []
    for input in inputs:
        emb, attention_mask = create_embedding(input)
        emb = optimize_with_gradient_ascent(emb, attention_mask)
        opt_input_embeddings.append(emb)
        attention_masks.append(attention_mask)

    return opt_input_embeddings, attention_masks

def optimize_with_gradient_ascent(current_emb, attention_mask, min_update_per_step=0.1, traj_path=None, prompt_idx=None, max_steps=100):
    current_emb = current_emb.detach().clone().float().requires_grad_(True)
    device = current_emb.device
    centre_device = output_center.squeeze().to(device)
    bias_subspace_device = bias_subspace.to(device)
    projection_matrix = bias_subspace_device.T @ bias_subspace_device
    I = torch.eye(projection_matrix.shape[0], device=device)
    def M(e):
        input = create_inputs_with_grad(e, attention_mask).float()
        return (I - projection_matrix) @ input

    B = None
    M_prev = None
    emb_prev = None
    step_count = 0
    restart_every = 10

    while (True):
        output_vec = create_output_from_emb(current_emb, attention_mask, with_grad=True).float()

        # calculate gradient
        centered_vec = output_vec - centre_device
        coords = bias_subspace_device @ centered_vec
        bias_score = torch.norm(coords)
        gt = torch.autograd.grad(
                        bias_score,
                        current_emb,
                        retain_graph=False,
                        create_graph=False
                    )[0]

        # Semantic Preservation Calculations (Broyden rank-1)
        t0 = time.time()
        with torch.no_grad():
            M_curr = M(current_emb)

        if step_count % restart_every == 0:
            Jt = torch.autograd.functional.jacobian(M, current_emb, vectorize=True)
            B = Jt.reshape(Jt.shape[0], -1).detach()
        else:
            delta_e = (current_emb.detach() - emb_prev).reshape(-1)
            delta_y = M_curr - M_prev
            denom = delta_e @ delta_e + 1e-12
            B = B + torch.outer(delta_y - B @ delta_e, delta_e) / denom

        gt_flat = gt.reshape(-1)
        BBT = B @ B.T
        reg = 1e-6 * torch.eye(BBT.shape[0], device=device)
        sol = torch.linalg.solve(BBT + reg, B @ gt_flat)
        pt_flat = gt_flat - B.T @ sol
        pt = pt_flat.reshape_as(gt)

        emb_prev = current_emb.detach().clone()
        M_prev = M_curr.clone()
        step_count += 1

        t1 = time.time()
        print("Semantic Preservation Calculations in", t1-t0)


        with torch.no_grad():
            # calculate epsilon
            direction = pt / (torch.norm(pt) + 1e-12)
            epsilon = get_optimized_epsilon(current_emb, direction, attention_mask, centre_device, bias_subspace_device)
            print("epsilon calculated in", time.time()-t1)

            # change embedding
            new_emb = current_emb + epsilon * direction

            step_delta_norm = epsilon

        print("Embedding Changed by", step_delta_norm)

        if traj_path is not None:
            with open(traj_path, "a", newline="") as f:
                csv.writer(f).writerow([prompt_idx, step_count, float(bias_score.item()), float(step_delta_norm)])

        if torch.isnan(new_emb).any():
            return current_emb.detach()
        elif step_delta_norm < min_update_per_step:
            return new_emb.detach()
        elif step_count >= max_steps:
            return new_emb.detach()
        else:
            current_emb = new_emb.detach().requires_grad_(True)

def main():
    init_optimizer_state()
    prompts = []
    with open(os.path.join(constants["RUN_DIR"], "benchmark_scores.csv"), newline="") as f:
        for row in csv.DictReader(f):
            prompts.append(row["prompt"])
    inputs, attention_masks = create_optimized_inputs(prompts)
    outputs = []
    output_vecs = []
    for input, attention_mask in zip(inputs, attention_masks):
        #output_vec, output = create_output_from_emb(input, attention_mask, True)
        output_vec = create_output_from_emb(input, attention_mask, with_grad=False)
        #outputs.append(output)
        output_vecs.append(output_vec)
        #print(output)
    opt_bias_score = get_bias_score_from_emb(output_vecs, bias_subspace, output_center)
    print("BASELINE")
    print(BASELINE_BIAS_SCORE)
    print("OPTIMIZED")
    print(opt_bias_score)

if __name__ == "__main__":
    main()
