import math
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))
from scipy.optimize import linear_sum_assignment

def stick_breaking(w):
    one_minus_w = 1 - w # [1-w1, 1-w2, ..., 1-wN]
    
    cumprod_one_minus_w = torch.cumprod(one_minus_w, dim=1) # [1-w1, (1-w1)(1-w2), ..., (1-w1)(1-w2)...(1-wN)]

    shifted_cumprod = torch.cat([torch.ones(w.shape[0], 1,device=w.device), cumprod_one_minus_w], dim=1) # [1, 1-w1, (1-w1)(1-w2), ..., (1-w1)(1-w2)...(1-wN-1)]
    w_one = torch.cat([w, torch.ones(w.shape[0], 1,device=w.device)], dim=1) # [w1, w2, ..., wN, 1]

    result = w_one * shifted_cumprod # [w1, w2(1-w1), w3(1-w1)(1-w2), ..., wN(1-w1)(1-w2)...(1-wN-1), (1-w1)(1-w2)...(1-wN)]

    return result

def log(msg,verbose=True):
        if verbose:
            print(msg)

def calculate_anneal_rate(t_start, t_min, max_epochs):
    """
    Calculates the multiplicative decay rate needed to reach t_min
    from t_start in exactly max_epochs.
    """
    if max_epochs <= 0:
        return 0.0
    rate = (t_min / t_start)**(1 / max_epochs) if t_start > 0 else 0.0
    
    return rate

def recon_loss_tensor(A_low, T, A_high):
    # 1. Compute column sums: shape (k,)
    column_norms = torch.sum(T, dim=0)
    
    # 2. Identify clusters that have active node assignments
    # If a cluster size is near 0, leave its norm as 1.0 to prevent division amplification
    safe_mask = column_norms > 1e-4
    column_norms_safe = torch.where(safe_mask, column_norms, torch.ones_like(column_norms))
    
    # 3. Normalize safely
    T_norm = T / column_norms_safe[None, :]
    
    # Zero out columns of completely empty clusters so they do not contribute to reconstruction
    T_norm = T_norm * safe_mask[None, :].float()
    
    # 4. Compute reconstruction and loss
    A_recon = T @ A_high @ T_norm.t()
    adj_recon_loss = torch.mean((A_recon - A_low) ** 2)
    
    return adj_recon_loss

def inter_cluster_dist(A_low,T):
    T_float = T.float()
    counts = T_float.sum(dim=0) +1e-8  # Shape: (n_high,)

    out_means = (T_float.t() @ A_low) / counts.unsqueeze(1) # Shape: (n_high, n_low)
    in_means = (A_low @ T_float) / counts.unsqueeze(0) # Shape: (n_low, n_high)

    out_sq = (out_means**2).sum(dim=1) # Shape: (n_high,)
    dists_out = out_sq.unsqueeze(1) + out_sq.unsqueeze(0) - 2 * (out_means @ out_means.t()) # Shape: (n_high, n_high)
    
    in_sq = (in_means**2).sum(dim=0)
    dists_in = in_sq.unsqueeze(1) + in_sq.unsqueeze(0) - 2 * (in_means.t() @ in_means)

    return dists_out.sum() + dists_in.sum()

def intra_cluster_dist(A_low,T):
    T_float = T.float()
    counts = T_float.sum(dim=0) + 1e-8 # Shape: (n_high,)

    out_sq_mean = ((T_float.t() @ A_low) / counts.unsqueeze(1))**2 # Shape: (n_high, n_low)
    in_sq_means = ((A_low @ T_float) / counts.unsqueeze(0))**2 # Shape: (n_low, n_high)

    out_mean_sq = (T_float.t() @ (A_low**2)) / counts.unsqueeze(1) # Shape: (n_high, n_low)
    in_mean_sq = (A_low**2 @ T_float) / counts.unsqueeze(0) # Shape: (n_low, n_high)
    
    out_variance = F.relu(out_mean_sq - out_sq_mean)
    in_variance = F.relu(in_mean_sq - in_sq_means)

    return out_variance.sum() + in_variance.sum()

def intra_cluster_penalty(A_low, T):
    """
    Penalises edges between low-level variables in the same cluster.
    
    Args:
        A_low (Tensor): Low-level adjacency matrix of shape (n_low, n_low)
        T (Tensor): Assignment matrix of shape (n_low, n_high)
    Returns:
        Tensor: Scalar penalty representing internal edge weights.
    """
    T_float = T.float()
    co_assignment = T_float @ T_float.t()
    internal_edges = co_assignment * (A_low ** 2)
    
    diag_mask = torch.eye(A_low.shape[0], device=A_low.device)
    internal_edges = internal_edges * (1.0 - diag_mask)

    return internal_edges.mean()

def inter_block_recon_penalty(A_low, T, A_high):
    # A_hat approximates the inter-block part of A_low
    with torch.no_grad():
        T_norm = T / (T**2).sum(dim=0).clamp(min=1e-8)
    A_hat = T @ A_high @ T_norm.T
    
    # Only measure error on inter-block entries
    inter_mask = 1.0 - T @ T.T
    diag_mask = 1.0 - torch.eye(A_low.shape[0], device=A_low.device)
    inter_mask = inter_mask * diag_mask
    
    return torch.mean(((A_low - A_hat)*inter_mask) ** 2)

def balance_penalty(T_soft):
    """
    Penalises unequal cluster sizes.
    Uses entropy of cluster sizes — maximised when all clusters are equal.
    """
    cluster_sizes = T_soft.sum(dim=0)  # (n_high,)
    probs = cluster_sizes / cluster_sizes.sum()
    entropy = -(probs * torch.log(probs + 1e-8)).sum()
    max_entropy = torch.log(torch.tensor(T_soft.shape[1], dtype=torch.float))
    return max_entropy - entropy  # 0 when perfectly balanced, positive otherwise

def dag_penalty_poly(A_high, s=1.0):
    # From Yu et al. (DAG-GNN): h(A) = tr((I + s*A∘A/k)^k) - k
    k = A_high.shape[0]
    A_high = A_high / (A_high.abs().max() + 1e-8) # normalise
    A_sq = A_high * A_high
    M = torch.eye(k, device=A_high.device) + s * A_sq / k
    return torch.trace(torch.linalg.matrix_power(M, k)) - k

def regularised_surjective_init(n_low, n_high, noise_std=0.1):
    if n_low < n_high:
        raise ValueError(
            f"Surjectivity impossible: n_low ({n_low}) must be >= n_high ({n_high}) "
            f"to allow each column to have a softmax value > 0.5."
        )
        
    # 1. Start with a baseline of low-variance random noise
    logits = torch.randn(n_low, n_high) * noise_std

    targets = torch.zeros(n_low, dtype=torch.long)
    targets[:n_high] = torch.arange(n_high)
    
    # Fill any remaining rows with random column assignments
    if n_low > n_high:
        targets[n_high:] = torch.randint(0, n_high, (n_low - n_high,))
    
    # Shuffle row assignments so the guaranteed coverage isn't locked to the top rows
    targets = targets[torch.randperm(n_low)]

    min_margin = math.log(max(1, n_high - 1)) 

    margin = min_margin + (5 * noise_std) + 0.1
    
    # 4. Inject the margin into the targeted positions
    logits[torch.arange(n_low), targets] += margin
    
    return logits

def sinkhorn(logits, temp=0.1, n_iters=20):
    # Gumbel noise for exploration (optional but recommended during training)
    noise = -torch.empty_like(logits).exponential_().log()
    log_alpha = (logits ) / temp

    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
    return log_alpha.exp()

class StraightThroughSoft(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_low, T):
        H_soft = torch.max(A_low[:, None, :] * T.T[None, :, :], dim=-1)[0]
        return H_soft

    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output

class StraightThroughHard(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, eps):
        return (H > eps).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Returning grad_output directly means the gradient wrt H is exactly 1
        return grad_output, None

import torch

def max_product_dag_series(A: torch.Tensor, m: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    Computes the max-product series: A*T + A*D*A*T + A*(D*A)^2*T + ...
    for a DAG adjacency matrix A (p x p), vector m (p), and matrix T (p x k).
    
    Parameters:
    A (torch.Tensor): Square adjacency matrix of shape (p, p)
    m (torch.Tensor): Vector of shape (p,) representing diag(m)
    T (torch.Tensor): Abstraction matrix of shape (p, k)
    
    Returns:
    torch.Tensor: The resulting matrix of shape (p, k)
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")
    if A.shape[0] != m.shape[0]:
        raise ValueError("Dimensions of A and vector m must match.")
    if A.shape[1] != T.shape[0]:
        raise ValueError("Dimensions of A and matrix T must match for multiplication.")
        
    p, k = T.shape
    device = A.device
    m = m.squeeze()

    # 1. Extract Layers (Generations) from Sinks to Sources
    out_degrees = (A > 0).sum(dim=1).clone()
    remaining = torch.ones(p, dtype=torch.bool, device=device)
    layers = []
    
    while remaining.any():
        sinks = (out_degrees == 0) & remaining
        if not sinks.any():
            raise ValueError("Graph contains cycles; it is not a valid DAG.")
            
        sink_indices = torch.where(sinks)[0]
        layers.append(sink_indices)
        
        remaining[sinks] = False
        parents_contrib = (A[:, sinks] > 0).sum(dim=1)
        out_degrees -= parents_contrib

    # 2. Compute Base Matrix B = A * T using max-product broadcasting
    # Shape: (p, p, 1) * (1, p, k) -> (p, p, k) -> max along dim 1 -> (p, k)
    B = (A.unsqueeze(2) * T.unsqueeze(0)).max(dim=1).values
    
    # 3. Initialize Core Matrices
    W = A * m.unsqueeze(0)  # W[i, k] = A[i, k] * m[k]
    C = B.clone()
    
    # 4. Process Layer-by-Layer (Dynamic Programming)
    for layer_idx in layers:
        if len(layer_idx) == 0:
            continue
            
        # Extract the submatrix for the current layer's nodes
        W_sub = W[layer_idx]  # Shape: (b, p) where b = len(layer_idx)
        
        # Perform batched max-product: W_sub (b, p) and C (p, k)
        # Tensor broadcasting maps (b, p, 1) * (1, p, k) -> (b, p, k)
        max_prod = (W_sub.unsqueeze(2) * C.unsqueeze(0)).max(dim=1).values  # Shape: (b, k)
        
        # Update the rows corresponding to the current layer (out-of-place to preserve autograd graph)
        C = C.index_put((layer_idx,), torch.max(B[layer_idx], max_prod))
        
    return C
        

class HailMary(nn.Module):
    def __init__(self, n_low, n_high, A_low, logits_init=None, init_assignment=None):
        super().__init__()
        self.n_low = n_low
        self.n_high = n_high
        
        self.register_buffer('A_low', A_low)  # (n_low, n_low) low-level weight matrix
        self.register_buffer('E_low', torch.linalg.inv(torch.eye(n_low, device=A_low.device) - A_low))
        self.register_buffer('A_low_binary', (A_low != 0).float())
        row_sum = torch.sum(A_low,dim=1)
        row_sum[row_sum == 0] = 1
        self.register_buffer('A_low_normalised', A_low/row_sum[:,None])
        self.T = nn.Parameter(torch.ones(n_low,n_high))

        # One logit per (low-level var, high-level block) pair
        logits_init = regularised_surjective_init(self.n_low, self.n_high) #  torch.zeros(n_low,n_high) # 
        self.logits = nn.Parameter(logits_init)

        self.node_positions = nn.Parameter(torch.linspace(0, self.n_high, self.n_high))

        self.order_logits = nn.Parameter(torch.randn(self.n_high, self.n_high))
        self.register_buffer('U', torch.triu(torch.ones(self.n_high, self.n_high), diagonal=1))

    @property
    def device(self):
        # Read live off a buffer instead of caching at construction time, so that
        # model.to(device) (e.g. moving to 'mps') keeps self.device in sync.
        return self.A_low.device

    def dag_mask(self,temp=1.):
        pos = (self.n_high - 1) - F.relu(self.n_high - 1 - self.node_positions)
        diff =self.node_positions.unsqueeze(0) - self.node_positions.unsqueeze(1) #pos.unsqueeze(0) - pos.unsqueeze(1) #
        hard_mask = (diff > 0).float()
        #temp = self.n_high / 3
        soft_mask = torch.sigmoid(diff/temp) - 0.5*torch.eye(self.n_high, device=self.node_positions.device)
        dag_mask = soft_mask + (hard_mask - soft_mask).detach()
        #dag_mask = torch.triu(torch.ones(self.n_high,self.n_high),diagonal=1)
        return self.U, soft_mask
    
    def dag_mask_sinkhorn(self, temp=0.1):
        P_soft = sinkhorn(self.order_logits, temp=temp)
        
        # Compute strict bijective permutation via Hungarian matching
        cost_matrix = -P_soft.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        P_hard = torch.zeros_like(P_soft)
        P_hard[row_ind, col_ind] = 1.0
        
        # Straight-Through Estimator
        P = P_soft + (P_hard - P_soft).detach()
        
        dag_mask = P.T @ self.U @ P
        soft_mask = P_soft.T @ self.U @ P_soft
        return dag_mask, soft_mask

    def get_partition(self, temp=1.):
        T_soft = F.softmax(self.logits/temp, dim=1)

        indices = torch.argmax(self.logits, dim=1)
        T_hard = F.one_hot(indices, self.n_high).float()
        T_hard = T_soft + (T_hard - T_soft).detach()  # Straight-through estimator
        
        return T_hard,T_soft

            
    
    def get_S(self,T):
        A_intra = T @ T.T * self.A_low
        I_low = torch.eye(self.n_low, device=T.device)
        S = torch.linalg.solve(I_low - A_intra, T)
        return S
    
    def get_A_high(self, T, S):
        # T, S: (n_low, n_high)
        AS = self.A_low @ S                       # (n_low, n_high), col j = A_low @ (t_j * s_j)
        A_high = T.T @ AS                                # (n_high, n_high), [i,j] = t_i^T @ A_low @ (t_j*s_j)
        t_norms = (T**2).sum(dim=0).clamp(min=1e-8)            # (n_high,) = ||t_i||^2 for binary T
        A_high = A_high / t_norms.unsqueeze(1)
        A_high = A_high * (1 - torch.eye(self.n_high, device=T.device))           # normalise each row by block size
        
        return A_high
    
    def get_E_high(self, T):
        E = self.E_low
        I = torch.eye(self.n_low,device=self.device)
        F_pi = torch.diag_embed(T.T) #(n_high, n_low, n_low)
        term = I + F_pi @ (E - I)
        ET = E @ T
        ET_batch = ET.unsqueeze(0).expand(self.n_high,-1,-1)
        T_batch = T.unsqueeze(0).expand(self.n_high, -1, -1) 
        lhs = torch.bmm(F_pi,ET_batch)
        term_T = torch.bmm(term, T_batch)  # (n_high, n_low, n_high)
        # extract column i from each batch i: term[i] @ T[:,i]
        rhs_col = term_T[torch.arange(self.n_high), :, torch.arange(self.n_high)]  # (n_high, n_low)
        rhs = rhs_col.unsqueeze(-1).expand(-1, -1, self.n_high)  # (n_high, n_low, n_high)

        dot_prod = torch.sum(lhs * rhs, dim=1)
        a_norm = torch.sum(lhs * lhs, dim=1)
        b_norm = torch.sum(rhs * rhs, dim=1)
        E_high_ij = dot_prod/(b_norm + 1e-8)

        return E_high_ij

    def massidda_loss(self, T_soft):
        # S[:,j] = (I - A_intra)^{-1} T[:,j]: within-cluster propagation of cluster j's identity column.
        # torch.linalg.solve makes no assumption about node ordering in A_low.                      # (n_low, n_high)
        A_intra = T_soft @ T_soft.T * self.A_low
        I_low = torch.eye(self.n_low, device=self.device)
        # H[l, j] = total influence of cluster j (through its internal paths) on node l.
        # Corresponds to W_ij @ (I - W_jj)^{-1} @ 1_j in the Massidda equations.
        H = torch.linalg.solve(I_low - A_intra.T, self.A_low.T).T                         # (n_low, n_high)

        # d_{i,j}[l] = T_soft[l,i] * H[l,j]: influence of cluster j within cluster i, weighted by membership.
        # Massidda condition: for each parent cluster i, all d_{i,j} vectors must be proportional.
        # We measure violation as (1 - cos^2) averaged over valid (i, j1<j2, j1≠i, j2≠i) triples.
        T_sq   = T_soft ** 2                          # (n_low, n_high)

        # Gram matrix of influence vectors within each cluster i.
        G      = torch.einsum('li,lj,lk->ijk', T_sq, H, H)  # (n_high, n_high, n_high)
        G_diag = torch.einsum('li,lj->ij', T_sq, H ** 2)    # (n_high, n_high): squared norms ||d_{i,j}||^2

        denom  = (G_diag[:, :, None] * G_diag[:, None, :]).clamp(min=1e-8)
        cos_sq = G ** 2 / denom                       # (n_high, n_high, n_high)

        n   = self.n_high
        idx = torch.arange(n, device=T_soft.device)
        excl_j1 = idx[:, None, None] != idx[None, :, None]   # i != j1
        excl_j2 = idx[:, None, None] != idx[None, None, :]   # i != j2
        upper   = torch.triu(torch.ones(n, n, device=T_soft.device, dtype=torch.bool), diagonal=1)
        # Only penalise triples where cluster j has non-negligible influence on cluster i.
        active  = G_diag > 1e-6                       # (n_high, n_high)
        mask    = excl_j1 & excl_j2 & upper[None, :, :] & active[:, :, None] & active[:, None, :]

        return (mask.float() * (1.0 - cos_sq)).sum()

    def principal_angle_loss(self, T_soft, A_high):
        """
        For each parent cluster i with >= 2 children in A_high, measures
        n_c - lambda_max(sum_j P_{i,j}), which is zero iff all child influence
        subspaces share a *common* direction (the Massidda mutual condition).
        This is strictly tighter than pairwise intersection for >= 3 children.
        """
        A_intra = T_soft @ T_soft.T * self.A_low
        I_low   = torch.eye(self.n_low, device=self.device)
        K = torch.linalg.solve((I_low - A_intra).T, self.A_low.T).T   # (n_low, n_low)
        eps = 1e-4

        edges = (A_high > 0).nonzero(as_tuple=False)   # (E, 2): (parent, child)
        if edges.shape[0] == 0:
            return torch.tensor(0.0, device=self.device)

        # Batch-compute all influence matrices and regularised projections in one shot
        a_vecs = T_soft[:, edges[:, 0]]                                  # (n_low, E)
        b_vecs = T_soft[:, edges[:, 1]]                                  # (n_low, E)
        M_all  = a_vecs.T.unsqueeze(-1) * K.unsqueeze(0) * b_vecs.T.unsqueeze(-2)  # (E, n_low, n_low)
        G_all  = torch.bmm(M_all.mT, M_all) + eps * I_low.unsqueeze(0)  # (E, n_low, n_low)
        P_all  = torch.bmm(M_all, torch.linalg.solve(G_all, M_all.mT))  # (E, n_low, n_low)

        loss = torch.tensor(0.0, device=self.device)

        for i in range(self.n_high):
            child_mask = edges[:, 0] == i
            n_c = int(child_mask.sum().item())
            if n_c < 2:
                continue

            # Sum all child projections: lambda_max = n_c iff a direction lies in all col(M_{i,j})
            P_sum = P_all[child_mask].sum(dim=0)   # (n_low, n_low)

            with torch.no_grad():
                v = P_sum.sum(dim=-1)              # (n_low,) warm start
                for _ in range(10):
                    v = P_sum @ v
                    nrm = v.norm().clamp(min=1e-12)
                    v = v / nrm

            # Gradient flows through P_sum; v is a fixed direction
            lambda_max = (v @ P_sum @ v) / (v @ v).clamp(min=1e-8)
            loss = loss + (n_c - lambda_max)

        return loss

    def intervention_loss(self, T_hard):
        E   = self.E_low                                           # (n_low, n_low)
        I   = torch.eye(self.n_low, device=self.device)
        T = T_hard.detach() * self.T


        # F_pi[i] = diag(T[:,i]): soft selector for cluster i, batched over all i
        F_pi  = torch.diag_embed(T_hard.T)                             # (n_high, n_low, n_low)

        # term[i] = I + F_pi[i] @ (E - I)
        term  = I + F_pi @ (E - I)                                # (n_high, n_low, n_low)

        # Phi[i] = term[i]^{-1} F_pi[i] E
        Phi   = torch.linalg.solve(term, F_pi @ E)                # (n_high, n_low, n_low)

        # Phi_T[i] = Phi[i] @ T
        Phi_T = Phi @ T                                            # (n_high, n_low, n_high)

        # proj_coeff[i,j] = T[:,i]^T Phi_T[i,:,j] / ||T[:,i]||^2
        t_norms_sq = (T ** 2).sum(dim=0).clamp(min=1e-8)          # (n_high,)
        proj_coeff = torch.einsum('ij,ijk->ik', T.T, Phi_T)       # (n_high, n_high)
        proj_coeff = proj_coeff / t_norms_sq.unsqueeze(1)

        # rank1[i] = P_i Phi_T[i]  where P_i = T[:,i] T[:,i]^T / ||T[:,i]||^2
        rank1   = T.T.unsqueeze(2) * proj_coeff.unsqueeze(1)      # (n_high, n_low, n_high)

        # ||(I - P_i) Phi_T[i]||_F^2 summed over i
        residual = Phi_T - rank1                                   # (n_high, n_low, n_high)
        return (residual ** 2).mean()
    
    def reconstruction_loss(self, T):
        A_low = self.A_low
        W_out = torch.mean((A_low[None, :, :] - A_low[:,None,:])**2,dim=-1)
        W_in = torch.mean((A_low.T[None,:,:] - A_low.T[:, None, :])**2,dim=-1)
        co_assign = T @ T.T                                   # (n_low, n_low)
        diag_mask = 1 - torch.eye(self.n_low, device=T.device)
        mask = co_assign * diag_mask
        recon_loss = (mask * (W_out + W_in)).mean()

        return recon_loss
    
    def _graph_loss_common(self, T):
        """co_assign and edge_weight depend only on T and self.A_low (not on the
        A_low_binary argument), so callers that invoke graph_loss more than once
        for the same T (e.g. forward()'s recon_loss + graph_loss) can compute
        this once and share it instead of redoing the O(n_low^2 * n_high) work."""
        co_assign = T @ T.T

        # Edge-weight modulator: H_w[i, j] = max |A_low| weight from node i to cluster j.
        # Weighting each cluster's CE contribution by the actual edge weight means a violation
        # involving a heavy edge is penalised proportionally more than one with a near-zero edge.
        H_w = torch.max(self.A_low.abs()[:, None, :] * T.T[None, :, :], dim=-1)[0] * (1 - T)
        edge_weight = torch.max(H_w[:, None, :], H_w[None, :, :]) # (n_low, n_low, n_high): max(H_w[i,j], H_w[i',j]) of weights for each (i,i',j)

        return co_assign, edge_weight

    def graph_loss(self, A_low_binary, T, intra_cluster=True, common=None):
        # H_soft[i, j]: connectivity strength of node i to cluster j [0,1], excluding own cluster.
        H_soft = torch.max(A_low_binary[:, None, :] * T.T[None, :, :], dim=-1)[0]
        H_soft = H_soft * (1 - T) if not intra_cluster else H_soft
        H_stable = torch.clamp(H_soft, min=1e-6, max=1. - 1e-6)
        H_logits = torch.logit(H_stable)

        target_i = H_stable.detach()[:, None, :]   # (n_low, 1, n_high)
        logits_j = H_logits[None, :, :]             # (1, n_low, n_high)

        # pairwise_ce[i, i', j]: CE loss for pair (i, i') on cluster j.  (n_low, n_low, n_high)
        pairwise_ce = target_i * F.softplus(-logits_j) + (1.0 - target_i) * F.softplus(logits_j)

        co_assign, edge_weight = common if common is not None else self._graph_loss_common(T)

        H_pen = (pairwise_ce * edge_weight).sum(dim=-1)  # (n_low, n_low)

        diag_mask = 1 - torch.eye(self.n_low, device=T.device)
        return (co_assign * diag_mask * H_pen).mean()
    
    def tree_loss(self, A_low_binary, T):
        """Encourages tree-like structure in the high-level graph by penalising nodes with multiple children."""
        H_soft = torch.max(A_low_binary[:, None, :] * T.T[None, :, :], dim=-1)[0] * (1 - T)
        H_pen = torch.sum(H_soft, dim=1)**2 - torch.sum(H_soft**2, dim=1)  # Penalise nodes with multiple children
        return H_pen.mean()
    
    def topo_loss(self,T):
        dag_mask, soft_mask = self.dag_mask()
        M = T.T @ self.A_low_binary @ T # M[i,j] is the number of edges between cluster i and cluster j
        return torch.sum((1 - dag_mask - torch.eye(self.n_high, device=self.device)) * M ** 2)


    def forward(self,temp=1., compute_graph=True, compute_topo=False, compute_intv=False):
        T_hard, T_soft = self.get_partition(temp)

        co_assign = T_soft @ T_soft.T
        edge_weight = self.A_low.abs() @ T_soft

        # Reconstruction Proxy - each node in a given cluster should have the same set of parent clusters
        H_soft_recon = torch.max(self.A_low_binary.T[:, None, :] * T_soft.T[None, :, :], dim=-1)[0]
        H_stable_recon = torch.clamp(H_soft_recon, min=1e-6, max=1. - 1e-6)
        H_logits_recon = torch.logit(H_stable_recon)
        target_i = H_stable_recon.detach()[:, None, :]   # (n_low, 1, n_high)
        logits_j = H_logits_recon[None, :, :]             # (1, n_low, n_high)

        pairwise_ce = target_i * F.softplus(-logits_j) + (1.0 - target_i) * F.softplus(logits_j)
        H_pen = (pairwise_ce * edge_weight).sum(dim=-1)  # (n_low, n_low)

        diag_mask = 1 - torch.eye(self.n_low, device=self.device)
        recon_loss = (co_assign * diag_mask * H_pen).mean()

        # Graph Conditions
        H_soft_graph = torch.max(self.A_low_binary[:, None, :] * T_soft.T[None, :, :], dim=-1)[0] * (1 - T_soft)
        H_stable_graph = torch.clamp(H_soft_graph, min=1e-6, max=1. - 1e-6)
        H_logits_graph = torch.logit(H_stable_graph)
        target_i = H_stable_graph.detach()[:, None, :]   # (n_low, 1, n_high)
        logits_j = H_logits_graph[None, :, :]             # (1, n_low, n_high)

        pairwise_ce = target_i * F.softplus(-logits_j) + (1.0 - target_i) * F.softplus(logits_j)
        H_pen = (pairwise_ce * edge_weight).sum(dim=-1)  # (n_low, n_low)

        diag_mask = 1 - torch.eye(self.n_low, device=self.device)
        graph_loss = (co_assign * diag_mask * H_pen).mean()

        # Tree Conditions - each high-level cluster should have at most one child
        tree_loss = torch.mean(torch.sum(H_soft_graph, dim=1)**2 - torch.sum(H_soft_graph**2, dim=1))

        entropy_loss = -torch.sum(T_soft * torch.log(T_soft + 1e-8), dim=1).mean()

        top_k,_ = torch.topk(T_soft,k=2,dim=0)
        col_deficit = F.relu(0.5 - top_k.mean(dim=0))
        surjective_loss = (self.n_high - torch.sum(T_hard.sum(dim=0) > 0))/self.n_high #(col_deficit * (-torch.log(T_soft + 1e-8).mean(dim=0))).sum() 
        #surjective_loss = F.relu(.5 - T_soft.max(dim=0).values).mean() #torch.var(torch.sum(T_soft,dim=0)) #F.relu(torch.prod(1-T_soft,dim=0) - 0.1).sum() #torch.sum(F.relu(1.0 - T_soft.sum(dim=0)))#torch.sum(T_soft.mean(dim=0) * torch.log(T_soft.mean(dim=0) + 1e-8)) + math.log(self.n_high)

        intv_loss = torch.tensor(0.0, device=self.device)
  

        return recon_loss, graph_loss, tree_loss, intv_loss, surjective_loss, entropy_loss

    def train_step(self, optimiser, lambda_recon=10, lambda_graph=1., lambda_topo=1., lambda_intv=1, lambda_surj=1, temp=1., compute_graph=True, compute_topo=True, compute_intv=True):
        optimiser.zero_grad()
        recon_loss, graph_loss, topo_loss, intv_loss, surjective_loss, entropy_loss = self.forward(temp, compute_graph=compute_graph, compute_topo=compute_topo, compute_intv=compute_intv)
        total_loss = (lambda_recon * recon_loss + 
                      lambda_graph * graph_loss +
                      lambda_topo * topo_loss +
                      lambda_intv * intv_loss +
                      lambda_surj * surjective_loss)
        total_loss.backward()
        optimiser.step()
        return recon_loss.item(), graph_loss.item(), topo_loss.item(), intv_loss.item(), surjective_loss.item(), entropy_loss.item()


    def train_graph_intv(self, graph_epochs, intv_epochs, verbose=True):
        self.train(graph_epochs, lambda_graph=1, lambda_topo=1., lambda_intv=0, lambda_surj=0, verbose=verbose, keep_history=False, compute_graph=True, compute_intv=False)
        self.train(intv_epochs, lambda_graph=1, lambda_topo=1., lambda_intv=1, lambda_surj=0, verbose=verbose, keep_history=False,  compute_graph=False, compute_intv=True)
    
    def train_graph_tree(self, num_epochs, verbose=True, keep_history=True):
        graph_epochs = int(num_epochs * 0.2)
        tree_epochs = num_epochs - graph_epochs
        self.train(graph_epochs, lambda_graph=1, lambda_topo=0., lambda_intv=0, lambda_surj=0, verbose=verbose, keep_history=False, compute_graph=True, compute_intv=False)
        history = self.train(tree_epochs, lambda_graph=1, lambda_topo=10., lambda_intv=0, lambda_surj=0, verbose=verbose, keep_history=keep_history,  compute_graph=True, compute_intv=False)
        return history

    def train(self, num_epochs, lambda_recon=10, lambda_graph=10., lambda_topo=100., lambda_intv=10, lambda_surj=1, verbose=True,keep_history=False, compute_graph=True, compute_topo=False, compute_intv=False):
        optimiser = torch.optim.Adam([{'params':[self.logits],'lr':1e-2}, {'params':[self.node_positions], 'lr':1e-2}],betas=(0.8,0.99))
        history = {'recon_loss':[], 'graph_loss':[], 'topo_loss':[], 'intv_loss':[], 'surjective_loss': [], 'entropy_loss': [], 'T':[], 'soft_mask':[]} if keep_history else None

        for epoch in range(num_epochs):
            recon_loss, graph_loss, topo_loss, intv_loss, surjective_loss, entropy_loss = self.train_step(optimiser, lambda_recon=lambda_recon, lambda_graph=lambda_graph, lambda_topo=lambda_topo, lambda_intv=lambda_intv, lambda_surj=lambda_surj, compute_graph=compute_graph, compute_topo=compute_topo, compute_intv=compute_intv)
            if epoch % 100 == 0 or epoch == num_epochs - 1:
                log(f"""Epoch {epoch+1}/{num_epochs},
                        Reconstruction: {recon_loss},
                        Graph: {graph_loss},
                        Topo: {topo_loss:.6f},
                        Intv: {intv_loss:.6f},
                        Surjective: {surjective_loss:.6f},
                        Entropy: {entropy_loss:.6f} """, verbose=verbose)
            if keep_history:
                history['recon_loss'].append(recon_loss)
                history['graph_loss'].append(graph_loss * lambda_graph)
                history['topo_loss'].append(topo_loss * lambda_topo)
                history['intv_loss'].append(intv_loss * lambda_intv)
                history['surjective_loss'].append(surjective_loss)
                history['entropy_loss'].append(entropy_loss)
                history['T'].append(self.get_partition()[1].cpu().detach().numpy())
                history['soft_mask'].append(self.dag_mask()[1].cpu().detach().numpy())
        return history

if __name__ == "__main__":
    from src.utils.plots import plot_matrix, visualize_causal_graph, plot_probability_path, plot_assignment_raster, animate_matrices
    from src.utils.generate_data import sample_low_level_graph, generate_easy_benchmark, sample_by_noise_adjustment_any_order
    from src.utils.losses import intervention_loss, observation_loss, abstraction_residual, graph_loss, graph_loss_entropy, reconstruction_loss, reconstruction_loss_with_data, reconstruction_loss_binary
    from src.utils.helper import process_output, get_S, get_A_high, is_dag, reconstruct_adj
    from src.models.wrapper import hc_fast, random
    import matplotlib.pyplot as plt
    from pathlib import Path
    from clustering import get_S, get_A_high, hierarchical_clustering
    import time

    DATA_TYPE = ['anomalies', 'easy_benchmark', 'standard_benchmark'][2]
    if DATA_TYPE=='anomalies':
        try:
            dir = Path('results/anomalous_graphs')
            files = [f for f in dir.glob('hail_mary*.npz')]
            gt_data = []
            for f in files:
                with np.load(f) as data:
                    intv_loss = data['intv_loss'].item()
                    if intv_loss > 1:
                        gt_data.append((data['A_low_gt'],data['T_gt'],intv_loss))
        except:
            raise Exception(f"Error loading anomalous graphs.")

        gt_data.sort(key= lambda x: x[2], reverse=True)
        A_low,T_gt,max_intv_loss = gt_data[0]
        n_low,n_high = T_gt.shape
        k = n_high
        S_gt = get_S(A_low=A_low, T=T_gt, T_hard=T_gt)
        A_high_gt = get_A_high(A_low=A_low, T=T_gt, priority='abstraction')

        
    elif DATA_TYPE=='easy_benchmark':
        seed = None #44
        n_low = 10 #number of low-level variables
        n_high = 4  #number of high-level variables
        k = n_high
        cluster_balance = 0.1
        blockiness = 0.5
        A_low, T_gt, S_gt, A_high_gt = generate_easy_benchmark(n_low=n_low,cluster_balance=cluster_balance,blockiness=blockiness,shuffle=False) #generate_generalized_benchmark(n_low=n_low,n_high=n_high,blockiness=1.0,shuffle=True,seed=seed)
    
    elif DATA_TYPE == 'standard_benchmark':
        seed = None #44
        n_low = 100 #number of low-level variables
        n_high = 10 #number of high-level variables
        k = n_high
        density = 0.4
        internal_density = 0.
        blockiness = 0.5
        heteregoneity = 1.
        A_low, T_gt, S_gt, A_high_gt = sample_low_level_graph(n_high,n_low,density,seed=seed,shuffle=True,blockiness=blockiness, heterogeneity=heteregoneity, internal_density=internal_density, graph_type='random')
        T_gt_binary = (T_gt != 0) * 1
        S_gt, T_gt, A_high_gt = process_output(T_hard=T_gt_binary, A_low=A_low)

    X_test = sample_by_noise_adjustment_any_order(A_low,n_samples=10000,gennorm_param=1.8,seed=seed)

    # Ground Truth Visualisation & Diagnostics 
    print(f"n_low = {n_low}, n_high = {n_high}")
    E = torch.linalg.inv(torch.eye(n_low) - torch.from_numpy(A_low).float())
    E_low = np.linalg.inv(np.eye(n_low) - A_low)
    E_high_gt = np.linalg.inv(np.eye(n_high) - A_high_gt)

    recon_loss_gt = reconstruction_loss_binary(A_low=A_low, T=T_gt, A_high=A_high_gt)
    recon_loss_data_gt = reconstruction_loss_with_data(A_low=A_low, X=X_test, T=T_gt, A_high=A_high_gt)
    graph_loss_gt = graph_loss(A_low=A_low,T_hard=T_gt_binary)
    obs_loss_gt = observation_loss(E_low=E_low, E_high=E_high_gt, S=S_gt, T=T_gt)
    intv_loss_gt = intervention_loss(E_low=E.cpu().numpy(), E_high=E_high_gt, S=S_gt, T=T_gt)
    abs_residual_gt = abstraction_residual(A_low=A_low, T=T_gt, A_high=A_high_gt)
    print(f"(Ground Truth) Reconstruction Loss: {recon_loss_gt:.4f}, Reconstruction (Data): {recon_loss_data_gt:.4f}, Graph Loss: {graph_loss_gt:.4f},  Observation Loss: {obs_loss_gt:.4f}, Intervention Loss: {intv_loss_gt:.4f}, Abstraction Residual: {abs_residual_gt}, Is Surjective?: {T_gt_binary.sum(axis=0).min() > 0}, Is DAG?: {is_dag(A_high_gt)}")
    #print(f"T_gt: {T_gt}")

    plot_matrix(
        matrix_list=[T_gt],
        title_list=['Endogenous Abstraction T'],
        fig_title='Ground Truth Matrices'
    )
    visualize_causal_graph(A_low, A_high_gt, T_gt, S_gt, title='Ground Truth Causal Abstraction',threshold=1e-3)

    #plot_probability_path(T_gt=T_gt_binary, A_low=A_low, LOSSES={'recon_loss':reconstruction_loss, 'graph_loss':graph_loss_entropy}, initial_T='surjective')
    
    if n_high <= 2:
       #plot_loss_landscape(A_low=torch.from_numpy(A_low).float(), T_gt=T_gt, idx1=0, idx2=1, flip_prob=0.5)
       pass

    # Training
    model = HailMary(n_low, n_high=k, A_low=torch.from_numpy(A_low).float(),logits_init=None)
    num_epochs = 1000 * int(np.log((n_low - k)*k + 1)+1)
    start_time = time.time()
    history = model.train(num_epochs=num_epochs, verbose=True, keep_history=True)
    runtime_hm = time.time() - start_time


    T_hard, T_soft = model.get_partition()
    dag_mask = model.dag_mask()[0].cpu().detach().numpy()
    print(f"Row-wise MinMax of T_soft: {torch.min(torch.max(T_soft,dim=1).values).item()}")
    #print(f"T_hard: {T_hard}")

    S_pred, T_pred, A_high_pred = process_output(T_hard=T_hard.cpu().detach().numpy(), A_low=A_low)
    E_high_pred = np.linalg.inv(np.eye(k) - A_high_pred)
    A_recon = reconstruct_adj(A_high_pred, T_pred)
    #print(f"A_low: {A_low}")
    #print(f"T: {T_pred}")
    #print(f"A_high: {A_high_pred}")
    #A_high_pred = dag_mask.cpu().detach().numpy() * A_high_pred

    print(f"(Ground Truth) Reconstruction Loss: {recon_loss_gt:.4f}, Reconstruction (Data): {recon_loss_data_gt:.4f}, Graph Loss: {graph_loss_gt:.4f},  Observation Loss: {obs_loss_gt:.4f}, Intervention Loss: {intv_loss_gt:.4f}, Abstraction Residual: {abs_residual_gt}, Is Surjective?: {T_gt_binary.sum(axis=0).min() > 0}, Is DAG?: {is_dag(A_high_gt)}")
    # Evaluation
    recon_loss = reconstruction_loss_binary(A_low=A_low, T=T_pred, A_high=A_high_pred)
    recon_loss_data = reconstruction_loss_with_data(A_low=A_low, T=T_pred, A_high=A_high_pred, X=X_test)
    graph_loss_ = graph_loss(A_low=A_low, T_hard=T_hard.cpu().detach().numpy())
    obs_loss = observation_loss(E_low=E_low, E_high=E_high_pred, S=S_pred, T=T_pred)
    intv_loss = intervention_loss(E_low=E.cpu().numpy(), E_high=E_high_pred, S=S_pred, T=T_pred)
    abs_residual = abstraction_residual(A_low=A_low,T=T_pred, A_high=A_high_pred)
    print(f"(Hail Mary) Time: {runtime_hm:.4f}, Reconstruction Loss: {recon_loss:.4f}, Reconstruction (Data): {recon_loss_data:.4f}, Graph Loss: {graph_loss_:.4f}, Observation Loss: {obs_loss:.4f}, Intervention Loss: {intv_loss:.4f}, Abstraction Residual: {abs_residual}, Is Surjective?: {T_hard.sum(axis=0).min() > 0}, Is DAG?: {is_dag(A_high_pred)}")

    # Hierarchical Clustering Baseline
    start_time = time.time()
    S_dict,T_dict,A_high_dict = hierarchical_clustering(A_low=A_low, n_high=k)
    runtime_hc = time.time() - start_time
    S_baseline, T_baseline, A_high_baseline = process_output(T_hard=T_dict[k], A_low=A_low)
    E_high_baseline = np.linalg.inv(np.eye(k) - A_high_baseline)
    A_recon_baseline = reconstruct_adj(A_high_baseline, T_baseline)

    recon_loss = reconstruction_loss_binary(A_low=A_low, T=T_baseline, A_high=A_high_baseline)
    recon_loss_data = reconstruction_loss_with_data(A_low=A_low, T=T_baseline, A_high=A_high_baseline, X=X_test)
    graph_loss_baseline = graph_loss(A_low=A_low, T_hard=T_dict[k])
    obs_loss = observation_loss(E_low=E_low, E_high=E_high_baseline, S=S_baseline, T=T_baseline)
    intv_loss = intervention_loss(E_low=E_low, E_high=E_high_baseline, S=S_baseline, T=T_baseline)
    abs_residual = abstraction_residual(A_low=A_low,T=T_baseline, A_high=A_high_baseline)
    print(f"(HC) Time: {runtime_hc:.4f}, Reconstruction Loss: {recon_loss:.4f}, Reconstruction (Data): {recon_loss_data:.4f}, Graph Loss: {graph_loss_baseline:.4f}, Observation Loss: {obs_loss:.4f}, Intervention Loss: {intv_loss:.4f}, Abstraction Residual: {abs_residual}, Is Surjective?: {T_dict[k].sum(axis=0).min() > 0}, Is DAG?: {is_dag(A_high_baseline)}")

    #Random
    T_random_binary = random(A_low,k=k)
    S_random, T_random, A_high_random = process_output(T_hard=T_random_binary, A_low=A_low)
    E_high_random = np.linalg.inv(np.eye(k) - A_high_random)
    A_recon_random = reconstruct_adj(A_high_random, T_random)

    recon_loss = reconstruction_loss_binary(A_low=A_low, T=T_random, A_high=A_high_random)
    recon_loss_data = reconstruction_loss_with_data(A_low=A_low, T=T_random, A_high=A_high_random, X=X_test)
    graph_loss_baseline = graph_loss(A_low=A_low, T_hard=T_random_binary)
    obs_loss = observation_loss(E_low=E_low, E_high=E_high_random, S=S_random, T=T_random)
    intv_loss = intervention_loss(E_low=E_low, E_high=E_high_random, S=S_random, T=T_random)
    abs_residual = abstraction_residual(A_low=A_low,T=T_random, A_high=A_high_random)
    print(f"(Random) Time: {runtime_hc:.4f}, Reconstruction Loss: {recon_loss:.4f}, Reconstruction (Data): {recon_loss_data:.4f}, Graph Loss: {graph_loss_baseline:.4f}, Observation Loss: {obs_loss:.4f}, Intervention Loss: {intv_loss:.4f}, Abstraction Residual: {abs_residual}, Is Surjective?: {T_random_binary.sum(axis=0).min() > 0}, Is DAG?: {is_dag(A_high_random)}")
    #print(f"T_baseline: {T_baseline}")
    
    # Visualising Hail Mary Results
    #plot_assignment_raster(history,T_gt=T_gt)
    #animate_matrices(history['soft_mask'], title='dag_mask',cmap='Blues')
    #animate_matrices(history['T'],title='abstraction',cmap='Reds')
    plot_matrix(
        matrix_list=[T_soft.cpu().detach().numpy(), T_dict[k]],
        title_list=['HM', 'HC'],
        fig_title='Partition Matrix'
    )
    visualize_causal_graph(A_low, A_high_pred, T_pred, S_pred, title='HM Abstraction')
    visualize_causal_graph(A_low, A_high_baseline, T_baseline, S_baseline, title='HC Abstraction')
    plot_matrix(
        matrix_list=[A_low, A_recon, A_recon_baseline],
        title_list=['Ground Truth', 'HM Reconstruction', 'HC Reconstruction'],
        fig_title='Adjacency Matrix'
    )
    #visualize_causal_graph(A_low, A_high_pred, T_pred, S_pred, title='Learned Causal Abstraction')
    
    #Plot a graph of loss vs epochs
    fig, ax = plt.subplots(1,2, figsize=(10,8))

    ax[0].plot(range(0,len(history['recon_loss']),10), history['recon_loss'][::10])
    ax[0].plot(range(0,len(history['intv_loss']),10), history['intv_loss'][::10])
    ax[0].plot(range(0,len(history['topo_loss']),10), history['topo_loss'][::10])
    ax[0].plot(range(0,len(history['graph_loss']),10), history['graph_loss'][::10])
    ax[0].legend(['Reconstruction Loss', 'Intervention Loss', 'Topo Loss', 'Graph Loss'])
    ax[0].set_yscale('log')
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")

    ax[1].plot(range(0,len(history['surjective_loss']),10), history['surjective_loss'][::10])
    ax[1].plot(range(0,len(history['entropy_loss']),10), history['entropy_loss'][::10])
    ax[1].legend(['Surjective Loss', 'Entropy Loss'])
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    plt.show()

    