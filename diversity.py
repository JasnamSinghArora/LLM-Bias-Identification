import torch
from scipy.sparse.csgraph import minimum_spanning_tree

DIFF_PATH = "cpd_diff_matrix.pt"

def main(diff_path=DIFF_PATH, max_samples=None):  # EDITED: parameterized path + optional subsampling, and returns the value to the pipeline
    diff = torch.load(diff_path).float().cpu()  # EDITED: parameterized path (was the hardcoded DIFF_PATH)
    if max_samples is not None and diff.shape[0] > max_samples:  # ADDED: the full m x m distance matrix needs ~40 GB at m=100,000
        diff = diff[torch.randperm(diff.shape[0])[:max_samples]]  # ADDED: random subsample keeps the MST computable
    m = diff.shape[0]
    E = diff / (diff.norm(p=2, dim=1, keepdim=True) + 1e-12) 
    D = torch.cdist(E, E)
    PSD = D ** 2
    
    q = 7
    D_sorted, _ = torch.sort(D, dim=1)
    tau = D_sorted[:, q]
    
    denom = 2.0 * tau[:, None] * tau[None, :]
    S = torch.exp(-PSD / (denom + 1e-12))   
    W = (1.0 - S).clamp_min(1e-9)
    W.fill_diagonal_(0.0)
    W_np = W.detach().cpu().numpy()
    
    mst = minimum_spanning_tree(W_np)
    D_mst = mst.sum() / (m - 1)
    return D_mst

if __name__ == "__main__":  # ADDED: guard so importing this module no longer runs it
    print(main())  # EDITED: was a bare module-level call
