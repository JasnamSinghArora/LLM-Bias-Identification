import torch
from scipy.sparse.csgraph import minimum_spanning_tree

DIFF_PATH = "cpd_diff_matrix.pt"

def main():
    diff = torch.load(DIFF_PATH).float().cpu()
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
 
print(main())
