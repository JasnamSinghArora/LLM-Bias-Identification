import torch
from scipy.sparse.csgraph import minimum_spanning_tree

DIFF_PATH = "cpd_diff_matrix.pt"
Q = 7  # neighbour that sets the local bandwidth


def mst_diversity(diff, q=Q):
    m = diff.shape[0]
    if m < 2:
        return 0.0
    E = diff / (diff.norm(p=2, dim=1, keepdim=True) + 1e-12)
    D = torch.cdist(E, E)
    tau = D.kthvalue(min(q, m - 1) + 1, dim=1).values  # self sits at distance zero

    # locally scaled Gaussian kernel, in place
    W = D.pow_(2).div_(2.0 * tau[:, None] * tau[None, :] + 1e-12).neg_().exp_()
    W = W.neg_().add_(1.0).clamp_min_(1e-9)
    W.fill_diagonal_(0.0)

    mst = minimum_spanning_tree(W.numpy())
    return float(mst.sum() / (m - 1))


def main(diff_path=DIFF_PATH):
    diff = torch.load(diff_path).float().cpu()
    return mst_diversity(diff)


if __name__ == "__main__":
    print(main())
