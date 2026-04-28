import torch

def pairwise_distances(x):
    # Helper to compute squared Euclidean distances
    instances_norm = torch.sum(x**2, dim=1).reshape(-1, 1)
    dist = instances_norm + instances_norm.t() - 2 * torch.mm(x, x.t())
    return torch.clamp(dist, min=0.0)

def gaussian_kernel(x, sigma=None):
    dist = pairwise_distances(x)
    if sigma is None:
        # Median heuristic for sigma
        sigma = torch.median(dist[dist > 0]).detach()
    return torch.exp(-dist / (2 * sigma + 1e-8))

def hsic(K, torch_H, L):
    # H is the centering matrix: I - (1/n)11^T
    # HSIC = Tr(KHLH) / (n-1)^2
    n = K.shape[0]
    return torch.trace(K @ torch_H @ L @ torch_H) / ((n - 1) ** 2)

def get_differentiable_lingam_scores(X):
    """
    Computes differentiable independence scores for each column in X.
    Low score = likely 'root' node (independent of residuals).
    """
    n, p = X.shape
    H = torch.eye(n, device=X.device) - (1.0 / n) * torch.ones((n, n), device=X.device)
    scores = []

    for j in range(p):
        xj = X[:, j:j+1]
        Kj = gaussian_kernel(xj)
        total_hsic = 0.0
        
        for i in range(p):
            if i == j: continue
            
            xi = X[:, i:i+1]
            # Differentiable Least Squares Residual
            # beta = (xj^T @ xj)^-1 @ xj^T @ xi
            beta = torch.inverse(xj.t() @ xj + 1e-6 * torch.eye(1, device=X.device)) @ xj.t() @ xi
            residual = xi - xj @ beta
            
            Li = gaussian_kernel(residual)
            total_hsic += hsic(Kj, H, Li)
            
        scores.append(total_hsic)
        
    return torch.stack(scores)