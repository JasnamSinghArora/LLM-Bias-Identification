"""One-off check that the stage 9 optimization step fits in this machine's GPU memory.

Run it on a rented server before committing to the full pipeline, with the same environment
variables the pipeline uses, e.g.:
    PIPELINE_MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct PIPELINE_MF=gender python gpu_memory_check.py
It loads the model, takes the longest benchmark prompt, and runs the exact tensor operations of
one gradient-ascent step (bias gradient, vectorized Jacobian, Broyden solve). It prints the peak
memory if the step fits, and crashes with an out-of-memory error if it does not.
"""
import os
import time
import torch
from constants import constants
from runlog import log
import helper
from helper import create_output_from_emb, create_inputs_with_grad, create_embedding

here = os.path.dirname(os.path.abspath(__file__))
dev = helper.model.device


def mem():
    if dev.type == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"cuda now {torch.cuda.memory_allocated() / 1e9:.1f} GB, peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB of {total:.0f} GB"
    if dev.type == "mps":
        return f"mps now {torch.mps.current_allocated_memory() / 1e9:.1f} GB (no peak counter on mps)"
    return "cpu"


log(f"model {constants['MODEL_PATH']} loaded on {dev}; {mem()}")
prompts = torch.load(os.path.join(here, "datasets", constants["MF"], "benchmark_prompts.pt"))
prompt = max(prompts, key=len)  # longest prompt = worst case for memory
emb, mask = create_embedding(prompt)
hidden = helper.model.get_input_embeddings().weight.shape[1]  # works for plain and multimodal-wrapper models alike
log(f"longest benchmark prompt has {emb.shape[1]} tokens; hidden size {hidden}")

# a random orthonormal k=3 subspace stands in for the real one; memory use is identical
bs = torch.linalg.qr(torch.randn(hidden, 3))[0].T.to(dev)  # QR on cpu: not implemented on mps
P = bs.T @ bs
I = torch.eye(hidden, device=dev)
centre = torch.zeros(hidden, device=dev)
cur = emb.detach().clone().float().requires_grad_(True)

t = time.time()
out = create_output_from_emb(cur, mask, with_grad=True).float()
gt = torch.autograd.grad(torch.norm(bs @ (out - centre)), cur)[0]
log(f"1/3 bias gradient OK in {time.time() - t:.1f}s; {mem()}")


def M(e):
    return (I - P) @ create_inputs_with_grad(e, mask).float()


t = time.time()
Jt = torch.autograd.functional.jacobian(M, cur, vectorize=True)
B = Jt.reshape(Jt.shape[0], -1).detach()
log(f"2/3 vectorized Jacobian OK in {time.time() - t:.1f}s, B shape {tuple(B.shape)}; {mem()}")

t = time.time()
BBT = B @ B.T
sol = torch.linalg.solve(BBT + 1e-6 * torch.eye(BBT.shape[0], device=dev), B @ gt.reshape(-1))
log(f"3/3 Broyden solve OK in {time.time() - t:.1f}s; {mem()}")
log("this machine can run stage 9 for this model")
