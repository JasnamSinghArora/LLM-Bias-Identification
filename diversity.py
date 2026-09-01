import torch
from scipy.sparse.csgraph import minimum_spanning_tree

DIFF_PATH = "cpd_diff_matrix.pt"

def mst_diversity(diff, q=7):
    m = diff.shape[0]
    if m < 2:
        return 0.0
    E = diff / (diff.norm(p=2, dim=1, keepdim=True) + 1e-12)
    D = torch.cdist(E, E)
    PSD = D ** 2

    q_eff = min(q, m - 1)
    D_sorted, _ = torch.sort(D, dim=1)
    tau = D_sorted[:, q_eff]

    denom = 2.0 * tau[:, None] * tau[None, :]
    S = torch.exp(-PSD / (denom + 1e-12))
    W = (1.0 - S).clamp_min(1e-9)
    W.fill_diagonal_(0.0)
    W_np = W.detach().cpu().numpy()

    mst = minimum_spanning_tree(W_np)
    D_mst = mst.sum() / (m - 1)
    return D_mst

def main(diff_path=DIFF_PATH, max_samples=None):
    diff = torch.load(diff_path).float().cpu()
    if max_samples is not None and diff.shape[0] > max_samples:
        diff = diff[torch.randperm(diff.shape[0])[:max_samples]]
    return mst_diversity(diff)

if __name__ == "__main__":
    print(main())
