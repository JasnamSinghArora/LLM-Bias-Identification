import os
import csv
import math
import time

import torch

from constants import constants
from helper import (
    create_output_from_emb,
    get_bias_score_from_emb,
    create_embedding,
    create_inputs_with_grad,
    line_search,
)
from bias_subspace import load_bias_subspace
from runlog import log

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
    opt_input_embeddings = []
    attention_masks = []
    for prompt in prompts:
        emb, attention_mask = create_embedding(prompt)
        emb = optimize_with_gradient_ascent(emb, attention_mask)
        opt_input_embeddings.append(emb)
        attention_masks.append(attention_mask)

    return opt_input_embeddings, attention_masks


def optimize_with_gradient_ascent(current_emb, attention_mask, traj_path=None, prompt_idx=None, log_prefix=""):
    max_steps = constants["OPT_MAX_STEPS"]
    step_min = constants["OPT_STEP_MIN"]
    gain_min = constants["OPT_GAIN_MIN"]
    restart_every = constants["OPT_RESTART_EVERY"]
    lam = constants["OPT_REG"]
    h = constants["OPT_PROBE_OFFSET"]

    current_emb = current_emb.detach().clone().float().requires_grad_(True)
    device = current_emb.device
    centre_device = output_center.squeeze().to(device)
    bias_subspace_device = bias_subspace.to(device)
    projection_matrix = bias_subspace_device.T @ bias_subspace_device
    I = torch.eye(projection_matrix.shape[0], device=device)

    def M(e):
        # non-bias part of the input representation
        return (I - projection_matrix) @ create_inputs_with_grad(e, attention_mask).float()

    with torch.no_grad():
        M_0 = M(current_emb)
    scale = M_0.norm().item()
    tol = constants["OPT_SEMANTIC_TOL"] * scale  # tau
    drift_tol = constants["OPT_DRIFT_TOL"] * scale  # tau bar

    B = None
    M_curr = M_0
    M_prev = None
    emb_prev = None
    score_prev = None
    eta_prev = None
    step_count = 0

    while True:
        output_vec = create_output_from_emb(current_emb, attention_mask, with_grad=True).float()
        bias_score = torch.norm(bias_subspace_device @ (output_vec - centre_device))
        score = bias_score.item()

        # termination: relative gain below zeta
        if score_prev is not None and score - score_prev <= gain_min * score_prev:
            gain = (score - score_prev) / (score_prev + 1e-12)
            log(f"{log_prefix}stopped after {step_count} steps: relative gain {gain:+.4f} <= {gain_min}")
            return current_emb.detach()

        # termination: cumulative drift above tau bar
        drift = (M_curr - M_0).norm().item()
        if drift > drift_tol:
            log(f"{log_prefix}stopped after {step_count} steps: semantic drift {drift:.4f} > {drift_tol:.4f}")
            return current_emb.detach()

        gt = torch.autograd.grad(bias_score, current_emb, retain_graph=False, create_graph=False)[0]
        t0 = time.time()

        # Jacobian: exact every K steps, Broyden otherwise
        if step_count % restart_every == 0:
            Jt = torch.autograd.functional.jacobian(M, current_emb, vectorize=True)
            B = Jt.reshape(Jt.shape[0], -1).detach()
        else:
            delta_e = (current_emb.detach() - emb_prev).reshape(-1)
            delta_y = M_curr - M_prev
            B = B + torch.outer(delta_y - B @ delta_e, delta_e) / (delta_e @ delta_e + 1e-12)

        # project the gradient onto the null space
        gt_flat = gt.reshape(-1)
        BBT = B @ B.T
        sol = torch.linalg.solve(BBT + lam * torch.eye(BBT.shape[0], device=device), B @ gt_flat)
        pt = (gt_flat - B.T @ sol).reshape_as(gt)
        t1 = time.time()

        with torch.no_grad():
            direction = pt / (torch.norm(pt) + 1e-12)

            # curvature cap from one probe pass
            kappa = 2.0 * (M(current_emb + h * direction) - M_curr).norm().item() / h ** 2
            eta_cap = math.sqrt(2.0 * tol / kappa) if kappa > 0 else math.inf

            # first local maximum along the ray
            eta_init = eta_prev if eta_prev else constants["OPT_LINE_SEARCH_INIT"]
            eta_peak = line_search(current_emb, direction, attention_mask, centre_device, bias_subspace_device,
                                   eta_init, eta_cap)
            eta = min(eta_peak, eta_cap)

            # verify the semantic error, halving if needed
            M_new = M(current_emb + eta * direction)
            sem_err = (M_new - M_curr).norm().item()
            while sem_err > tol and eta > step_min:
                eta /= 2.0
                M_new = M(current_emb + eta * direction)
                sem_err = (M_new - M_curr).norm().item()
            new_emb = current_emb + eta * direction
        t2 = time.time()

        step_count += 1
        log(f"{log_prefix}step {step_count}/{max_steps}: bias score {score:.4f} | step size {eta:.4f}"
            f" (peak {eta_peak:.4f}, cap {eta_cap:.4f}) | semantic error {sem_err:.4f} <= {tol:.4f}"
            f" | Jacobian {t1 - t0:.1f}s, line search {t2 - t1:.1f}s")

        if traj_path is not None:
            with open(traj_path, "a", newline="") as f:
                csv.writer(f).writerow([prompt_idx, step_count, score, eta])

        if torch.isnan(new_emb).any():
            log(f"{log_prefix}stopped after {step_count} steps: NaN in the new embedding, keeping the previous one")
            return current_emb.detach()
        if eta <= step_min:
            log(f"{log_prefix}converged after {step_count} steps: step size {eta:.4f} <= {step_min}")
            return new_emb.detach()
        if step_count >= max_steps:
            log(f"{log_prefix}stopped at the {max_steps}-step cap")
            return new_emb.detach()

        emb_prev = current_emb.detach().clone()
        M_prev = M_curr
        M_curr = M_new
        score_prev = score
        eta_prev = eta
        current_emb = new_emb.detach().requires_grad_(True)


def main():
    init_optimizer_state()
    prompts = []
    with open(os.path.join(constants["RUN_DIR"], "benchmark_scores.csv"), newline="") as f:
        for row in csv.DictReader(f):
            prompts.append(row["prompt"])
    inputs, attention_masks = create_optimized_inputs(prompts)
    output_vecs = []
    for emb, attention_mask in zip(inputs, attention_masks):
        output_vecs.append(create_output_from_emb(emb, attention_mask, with_grad=False))
    opt_bias_score = get_bias_score_from_emb(output_vecs, bias_subspace, output_center)
    print("BASELINE")
    print(BASELINE_BIAS_SCORE)
    print("OPTIMIZED")
    print(opt_bias_score)


if __name__ == "__main__":
    main()
