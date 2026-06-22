import math
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))
from utils.generate_data import sample_low_level_graph
from utils.helper import enforce_dag, learn_dag, process_output, get_S, get_A_high
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
        

class HailMary(nn.Module):
    def __init__(self, n_low, n_high, A_low, logits_init=None, init_assignment=None):
        super().__init__()
        self.device = A_low.device
        self.n_low = n_low
        self.n_high = n_high
        
        self.register_buffer('A_low', A_low)  # (n_low, n_low) low-level weight matrix
        self.register_buffer('E_low', torch.linalg.inv(torch.eye(n_low) - A_low))
        self.register_buffer('A_low_binary', (A_low != 0).float())
        row_sum = torch.sum(A_low,dim=1)
        row_sum[row_sum == 0] = 1
        self.register_buffer('A_low_normalised', A_low/row_sum[:,None])

        # One logit per (low-level var, high-level block) pair
        logits_init = regularised_surjective_init(self.n_low, self.n_high) # torch.cat([torch.zeros(n_low,n_high-1),torch.ones(n_low,1)],dim=1) # torch.zeros(n_low,n_high) # 
        self.logits = nn.Parameter(logits_init)
        self.node_positions = nn.Parameter(torch.linspace(0, self.n_high, self.n_high))

        self.order_logits = nn.Parameter(torch.randn(self.n_high, self.n_high))
        self.register_buffer('U', torch.triu(torch.ones(self.n_high, self.n_high), diagonal=1))
    
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
        # Soft assignment for gradients, hard for forward logic
        T_soft = F.softmax(self.logits / temp, dim=1)  # (n_low, n_high), rows sum to 1
        indices = torch.argmax(self.logits, dim=1)
        T_hard = F.one_hot(indices, self.n_high).float()
        T_hard = (T_hard - T_soft).detach() + T_soft  # straight-through
        
        return T_hard, T_soft

            
    
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

    
    def abstraction_residual(self, T_soft, A_high=None):
        n_high = T_soft.shape[1]
        n_low = T_soft.shape[0]
        I = torch.eye(n_low, device=T_soft.device)

        # 1. Compute s_j for all clusters (loop over n_high, which is small)
        S_list = []
        for j in range(n_high):
            t_j = T_soft[:, j]
            A_jj = t_j.unsqueeze(1) * self.A_low * t_j.unsqueeze(0)
            s_j = torch.linalg.solve(I - A_jj, t_j)
            S_list.append(s_j)
        S = torch.stack(S_list, dim=1)  # (n_low, n_high)

        # 2. Influence matrix: col j = A_low @ (t_j * s_j)
        influence_matrix = self.A_low @ S #(T_soft * S)  # (n_low, n_high)

        # 3. Weighted variance of influence_j over block i, for all (i,j) pairs
        # weights[r, i] = T_soft[r, i]^2
        T_sq = T_soft ** 2                                          # (n_low, n_high)
        t_i_norms = T_sq.sum(dim=0).clamp(min=1e-8)                # (n_high,) effective block sizes

        # Weighted mean: a_star[i,j] = sum_r w[r,i] * influence[r,j] / sum_r w[r,i]
        if A_high is None:
            A_high = (T_soft.T @ influence_matrix) / t_i_norms.unsqueeze(1)  # (n_high, n_high)
            A_high = A_high * (1 - torch.eye(self.n_high, device=T_soft.device)) 
        diff = influence_matrix.unsqueeze(0) - A_high.unsqueeze(1) * T_soft.T.unsqueeze(2)

        # Weighted variance: sum_r w[r,i] * diff[i,r,j]^2 / sum_r w[r,i]
        weights = T_sq.T.unsqueeze(2)                               # (n_high, n_low, 1)
        residuals = (weights * diff ** 2).sum(dim=1) / t_i_norms.unsqueeze(1)  # (n_high, n_high)

        # Mask diagonal and sum
        diag_mask = 1.0 - torch.eye(n_high, device=self.device)
        return (residuals * diag_mask).sum()
    
    def intervention_loss(self, S, T, corollary_version=True):
        E_high = self.get_E_high(T)
        E_low = self.E_low
        I_low = torch.eye(self.n_low,device=self.device)
        e_j_batch = torch.eye(self.n_high, device=self.device)
        F_jj_batch = torch.diag_embed(e_j_batch)
        F_pi_batch = torch.diag_embed(torch.einsum('ij,kj->ik',e_j_batch,T)) # ~ 1e-1
       
        term = I_low + F_pi_batch @ (E_low - I_low) # ~ 1e-1
        if not corollary_version:
            lhs = (F_pi_batch @ E_low @ T).to(self.device)
            rhs = (term @ T @ F_jj_batch @ E_high).to(self.device)
            residual = (lhs - rhs) ** 2
            diag_mask = 1.0 - torch.eye(self.n_high, device=self.device)
            masked_residual = diag_mask.unsqueeze(1) * residual
            loss = (masked_residual).sum()
        else:
            lhs = (F_pi_batch @ S).to(self.device) # ~ 1e-2
            rhs = (term @ F_pi_batch @ T) # ~ 1e-3
            residual = (lhs - rhs)**2 # ~ 1e-4
            loss = residual.sum() 
        
        return loss
    
    def graph_loss(self, A_low_binary, T, entropy=True):
        H = A_low_binary @ T # H[i,j] is the [EXPECTED] number of edges between node i and cluster j
        if not entropy:
            eps = 1e-6
            H = StraightThroughHard.apply(H,eps)
            HH = H @ H.T # number of shared cluster connections between node i and node j
            H_norm_sq = (H**2).sum(1)
            H_pen = (H_norm_sq[:,None]+ H_norm_sq[None,:]- 2*HH) # ||H[i] - H[i']||^2
        else:
            H_soft = torch.max(A_low_binary[:, None, :] * T.T[None, :, :], dim=-1)[0] #StraightThroughSoft.apply(A_low_binary, T)
            H_stable = torch.clamp(H_soft, min=1e-6, max=1. - 1e-6)
        
            # 2. Derive pseudo-logits: log(p / (1-p))
            H_logits = torch.logit(H_stable)
            
            # 3. Use the stable cross-entropy formulation with detached targets
            target_i = H_stable.detach()[:, None, :] 
            logits_j = H_logits[None, :, :]  
            
            # Analytically stable element-wise binary cross-entropy
            pairwise_ce = target_i * F.softplus(-logits_j) + (1.0 - target_i) * F.softplus(logits_j)
            H_pen = pairwise_ce.sum(dim=-1)

        co_assign = T @ T.T                                   # (n_low, n_low)
        diag_mask = 1 - torch.eye(self.n_low, device=T.device)
        mask = co_assign * diag_mask
        graph_loss = (mask * H_pen).sum()

        return graph_loss

    def forward(self,temp=1.,T=None,compute_abs=False,compute_graph=True, compute_topo=False, compute_intv=False):
        if T is None:
            T_hard, T_soft = self.get_partition(temp) # soft T
            T = T_soft
        else:
            T_hard = T
        
        A_low_binary = (self.A_low != 0).float()
        S = self.get_S(T)
        dag_mask, soft_mask = self.dag_mask(temp)
        A_high = self.get_A_high(T,S)

        if compute_topo:
            A_high = A_high * dag_mask
            M = T.T @ A_low_binary @ T # M[i,j] is the number of edges between cluster i and cluster j
            topo_loss = torch.mean((1 - dag_mask - torch.eye(self.n_high)) * M ** 2) 
        else:
            topo_loss = torch.tensor(0.0, device=self.device)

        recon_loss = recon_loss_tensor(self.A_low_normalised, T_hard, self.get_A_high(T_hard,self.get_S(T_hard))) # self.graph_loss(A_low_binary.T, T_soft) #self.graph_loss(A_low_binary.T, T_soft)  #inter_block_recon_penalty(A_low_binary,T,A_high) #+ intra_cluster_penalty(A_low_binary,T)  inter_block_recon_penalty(self.A_low,T,A_high) # 
        entropy_loss = -torch.sum(T * torch.log(T + 1e-8), dim=1).mean()
        
        top_k,_ = torch.topk(T,k=2,dim=0) 
        col_deficit = F.relu(0.5 - top_k.mean(dim=0))  
        surjective_loss = (col_deficit * (-torch.log(T + 1e-8).mean(dim=0))).sum() #torch.sum(T.mean(dim=0) * torch.log(T.mean(dim=0) + 1e-8)) + math.log(self.n_high)  
        #surjective_loss = F.relu(.5 - T.max(dim=0).values).mean() #torch.var(torch.sum(T_soft,dim=0)) #F.relu(torch.prod(1-T_soft,dim=0) - 0.1).sum() #torch.sum(F.relu(1.0 - T.sum(dim=0)))
    
        abs_loss = self.abstraction_residual(T,A_high) if compute_abs else torch.tensor(0.0, device=self.device)
        graph_loss = self.graph_loss(A_low_binary, T_soft) if compute_graph else torch.tensor(0.0, device=self.device)
        intv_loss = self.intervention_loss(T=T,S=S) if compute_intv else torch.tensor(0.0, device=self.device)

        return recon_loss, abs_loss, graph_loss, topo_loss, intv_loss, surjective_loss, entropy_loss

    def train_step(self, optimiser, lambda_abs=1., lambda_graph=1., lambda_topo=1., lambda_intv=1, lambda_surj=1, lambda_entropy=1., temp=1., compute_abs=True, compute_graph=True, compute_topo=True, compute_intv=True):
        optimiser.zero_grad()
        recon_loss, abs_loss, graph_loss, topo_loss, intv_loss, surjective_loss, entropy_loss = self.forward(temp, compute_abs=compute_abs, compute_graph=compute_graph, compute_topo=compute_topo, compute_intv=compute_intv)
        total_loss = (recon_loss + 
                      (lambda_abs) * abs_loss + 
                      (lambda_graph) * graph_loss +
                      (lambda_topo) * topo_loss +
                      (lambda_intv) * intv_loss +
                      (lambda_surj) * surjective_loss + 
                      (lambda_entropy) * entropy_loss 
                      )
        total_loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        return recon_loss.item(), abs_loss.item(), graph_loss.item(), topo_loss.item(), intv_loss.item(), surjective_loss.item(), entropy_loss.item()
    
    def loss_tracker(self, loss, alpha=0.1, patience=50, min_delta=1e-4):
        pass

    def train(self, num_epochs, lambda_abs=1., lambda_graph=1, lambda_topo=1., lambda_intv=1e4, lambda_surj=0, lambda_min_entropy=0, lambda_max_entropy=0, verbose=True,keep_history=False,compute_abs=False, compute_graph=True, compute_topo=False, compute_intv=False):
        optimiser = torch.optim.Adam([{'params':[self.logits],'lr':1e-2}, {'params':[self.node_positions], 'lr':1e-2}],betas=(0.8,0.99))
        history = {'recon_loss':[], 'abs_loss': [], 'graph_loss':[], 'topo_loss':[], 'intv_loss':[], 'surjective_loss': [], 'entropy_loss': [], 'T':[], 'soft_mask':[]} if keep_history else None
        lambda_entropy = lambda_max_entropy
        rate = calculate_anneal_rate(lambda_max_entropy, lambda_min_entropy, num_epochs)
        for epoch in range(num_epochs):
            lambda_entropy *= rate
            recon_loss, abs_loss, graph_loss, topo_loss, intv_loss, surjective_loss, entropy_loss = self.train_step(optimiser, lambda_abs, lambda_graph, lambda_topo, lambda_intv=lambda_intv, lambda_surj=lambda_surj, lambda_entropy=lambda_entropy, compute_abs=compute_abs, compute_graph=compute_graph, compute_topo=compute_topo, compute_intv=compute_intv)
            if epoch % 100 == 0 or epoch == num_epochs - 1:
                log(f"""Epoch {epoch+1}/{num_epochs},
                        Reconstruction: {recon_loss},
                        Abstraction: {abs_loss:.6f}, 
                        Graph: {graph_loss},
                        Topo: {topo_loss:.6f},
                        Intv: {intv_loss:.6f},
                        Surjective: {surjective_loss:.6f},
                        Entropy: {entropy_loss} """, verbose=verbose)
            if keep_history:
                history['recon_loss'].append(recon_loss)
                history['abs_loss'].append(abs_loss * lambda_abs)
                history['graph_loss'].append(graph_loss * lambda_graph)
                history['topo_loss'].append(topo_loss * lambda_topo)
                history['intv_loss'].append(intv_loss * lambda_intv)
                history['surjective_loss'].append(surjective_loss * lambda_surj)
                history['entropy_loss'].append(entropy_loss * lambda_entropy)
                history['T'].append(self.get_partition()[1].cpu().detach().numpy())
                history['soft_mask'].append(self.dag_mask()[1].cpu().detach().numpy())
        return history

if __name__ == "__main__":
    from utils.plots import plot_matrix, visualize_causal_graph, plot_probability_path, plot_assignment_raster, animate_matrices
    from utils.generate_data import generate_easy_benchmark, generate_generalized_benchmark
    from utils.losses import intervention_loss, observation_loss, abstraction_residual, graph_loss, topo_loss, intervention_loss_direct, reconstruction_loss
    from utils.helper import enforce_dag, is_dag, reconstruct_adj
    import matplotlib.pyplot as plt
    from pathlib import Path
    from algorithms.alignment import get_S, get_A_high, hierarchical_clustering
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
        S_gt = get_S(A_low=A_low, T=T_gt)
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
        n_low = 50 #number of low-level variables
        n_high = 5 #number of high-level variables
        k = n_high
        density = 0.4
        internal_density = 0.
        blockiness = 0.1
        heteregoneity = 1.
        A_low, T_gt_binary, S_gt, A_high_gt = sample_low_level_graph(n_high,n_low,density,seed=seed,shuffle=True,blockiness=blockiness, heterogeneity=heteregoneity, internal_density=internal_density)
        S_gt, T_gt, A_high_gt = process_output(T_hard=T_gt_binary, A_low=A_low)

    

    # Ground Truth Visualisation & Diagnostics 
    print(f"n_low = {n_low}, n_high = {n_high}")
    E = torch.linalg.inv(torch.eye(n_low) - torch.from_numpy(A_low).float())
    E_low = np.linalg.inv(np.eye(n_low) - A_low)
    E_high_gt = np.linalg.inv(np.eye(n_high) - A_high_gt)

    recon_loss_gt = reconstruction_loss(A_low=A_low,A_hat=reconstruct_adj(A_high_gt, T_gt))
    graph_loss_gt = graph_loss(A_low=A_low,T=T_gt_binary)
    obs_loss_gt = observation_loss(E_low=E_low, E_high=E_high_gt, S=S_gt, T=T_gt)
    intv_loss_gt = intervention_loss_direct(E_low=E.cpu().numpy(), E_high=E_high_gt, S=S_gt, T=T_gt)
    abs_residual_gt = abstraction_residual(A_low=A_low, T=T_gt, A_high=A_high_gt)
    print(f"(Ground Truth) Reconstruction Loss: {recon_loss_gt:.4f}, Graph Loss: {graph_loss_gt:.4f},  Observation Loss: {obs_loss_gt:.4f}, Intervention Loss: {intv_loss_gt:.4f}, Abstraction Residual: {abs_residual_gt}, Is Surjective?: {T_gt_binary.sum(axis=0).min() > 0}, Is DAG?: {is_dag(A_high_gt)}")
    #print(f"T_gt: {T_gt}")

    plot_matrix(
        matrix_list=[T_gt],
        title_list=['Endogenous Abstraction T'],
        fig_title='Ground Truth Matrices'
    )
    visualize_causal_graph(A_low, A_high_gt, T_gt, S_gt, title='Ground Truth Causal Abstraction',threshold=1e-3)

    #plot_probability_path(T_gt=T_gt, A_low=A_low, LOSSES={'graph_loss':graph_loss,'abstraction_residual':abstraction_residual, 'intervention_loss':intervention_loss_direct}, initial_T='surjective')
    
    if n_high <= 2:
       #plot_loss_landscape(A_low=torch.from_numpy(A_low).float(), T_gt=T_gt, idx1=0, idx2=1, flip_prob=0.5)
       pass

    # Training
    model = HailMary(n_low, n_high=k, A_low=torch.from_numpy(A_low).float(),logits_init=None)
    num_epochs = 1000 * int(np.log(n_low*k)+1)
    start_time = time.time()
    history = model.train(num_epochs=num_epochs, verbose=True, keep_history=True)
    runtime_hm = time.time() - start_time


    T_hard, T_soft = model.get_partition()
    dag_mask = model.dag_mask()[0].cpu().detach().numpy()
    print(f"Row-wise MinMax of T_soft: {torch.min(torch.max(T_soft,dim=1).values).item()}")
    #print(f"T_hard: {T_hard}")

    S_pred, T_pred, A_high_pred = process_output(T_hard=T_hard.cpu().detach().numpy(), A_low=A_low, E_low=E_low, dag_mask=None)
    E_high_pred = np.linalg.inv(np.eye(k) - A_high_pred)
    A_recon = reconstruct_adj(A_high_pred, T_pred)
    #print(f"A_low: {A_low}")
    #print(f"T: {T_pred}")
    #print(f"A_high: {A_high_pred}")
    #A_high_pred = dag_mask.cpu().detach().numpy() * A_high_pred

    print(f"(Ground Truth) Reconstruction Loss: {recon_loss_gt:.4f}, Graph Loss: {graph_loss_gt:.4f},  Observation Loss: {obs_loss_gt:.4f}, Intervention Loss: {intv_loss_gt:.4f}, Abstraction Residual: {abs_residual_gt}, Is Surjective?: {T_gt_binary.sum(axis=0).min() > 0}, Is DAG?: {is_dag(A_high_gt)}")
    # Evaluation
    recon_loss = reconstruction_loss(A_low=A_low, A_hat=A_recon)
    graph_loss_ = graph_loss(A_low=A_low, T=T_hard.cpu().detach().numpy())
    obs_loss = observation_loss(E_low=E_low, E_high=E_high_pred, S=S_pred, T=T_pred)
    intv_loss = intervention_loss_direct(E_low=E.cpu().numpy(), E_high=E_high_pred, S=S_pred, T=T_pred)
    abs_residual = abstraction_residual(A_low=A_low,T=T_pred, A_high=A_high_pred)
    print(f"(Hail Mary) Time: {runtime_hm:.4f}, Reconstruction Loss: {recon_loss:.4f}, Graph Loss: {graph_loss_:.4f}, Observation Loss: {obs_loss:.4f}, Intervention Loss: {intv_loss:.4f}, Abstraction Residual: {abs_residual}, Is Surjective?: {T_hard.sum(axis=0).min() > 0}, Is DAG?: {is_dag(A_high_pred)}")

    # Hierarchical Clustering Baseline
    start_time = time.time()
    S_dict,T_dict,A_high_dict = hierarchical_clustering(A_low=A_low, n_high=k)
    runtime_hc = time.time() - start_time
    S_baseline, T_baseline, A_high_baseline = process_output(T_hard=T_dict[k], A_low=A_low, E_low=E_low)
    E_high_baseline = np.linalg.inv(np.eye(k) - A_high_dict[k])
    A_recon_baseline = reconstruct_adj(A_high_baseline, T_baseline)

    recon_loss = reconstruction_loss(A_low=A_low, A_hat=A_recon_baseline)
    graph_loss_baseline = graph_loss(A_low=A_low, T=T_dict[k])
    obs_loss = observation_loss(E_low=E_low, E_high=E_high_baseline, S=S_baseline, T=T_baseline)
    intv_loss = intervention_loss_direct(E_low=E_low, E_high=E_high_baseline, S=S_baseline, T=T_baseline)
    abs_residual = abstraction_residual(A_low=A_low,T=T_baseline, A_high=A_high_baseline)
    print(f"(HC) Time: {runtime_hc:.4f}, Reconstruction Loss: {recon_loss:.4f}, Graph Loss: {graph_loss_baseline:.4f}, Observation Loss: {obs_loss:.4f}, Intervention Loss: {intv_loss:.4f}, Abstraction Residual: {abs_residual}, Is Surjective?: {T_dict[k].sum(axis=0).min() > 0}, Is DAG?: {is_dag(A_high_baseline)}")

    
    # Visualising Hail Mary Results
    #plot_assignment_raster(history,T_gt=T_gt)
    #animate_matrices(history['soft_mask'], title='dag_mask',cmap='Blues')
    animate_matrices(history['T'],title='abstraction',cmap='Reds')
    plot_matrix(
        matrix_list=[T_soft.cpu().detach().numpy(), T_pred],
        title_list=['Partition', 'Abstraction'],
        fig_title='Learned Matrices'
    )
    visualize_causal_graph(A_low, A_high_pred, T_pred, S_pred, title='Learned Causal Abstraction')
    plot_matrix(
        matrix_list=[A_low, A_recon],
        title_list=['Ground Truth', 'Reconstruction'],
        fig_title='Adjacency Matrix'
    )
    #visualize_causal_graph(A_low, A_high_pred, T_pred, S_pred, title='Learned Causal Abstraction')
    
    #Plot a graph of loss vs epochs
    fig, ax = plt.subplots(1,2, figsize=(10,8))

    ax[0].plot(range(0,len(history['recon_loss']),10), history['recon_loss'][::10])
    ax[0].plot(range(0,len(history['abs_loss']),10), history['abs_loss'][::10])
    ax[0].plot(range(0,len(history['intv_loss']),10), history['intv_loss'][::10])
    ax[0].plot(range(0,len(history['topo_loss']),10), history['topo_loss'][::10])
    ax[0].plot(range(0,len(history['graph_loss']),10), history['graph_loss'][::10])
    ax[0].legend(['Reconstruction Loss', 'Abstraction Loss','Intervention Loss', 'Topo Loss', 'Graph Loss'])
    ax[0].set_yscale('log')
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")

    ax[1].plot(range(0,len(history['surjective_loss']),10), history['surjective_loss'][::10])
    ax[1].plot(range(0,len(history['entropy_loss']),10), history['entropy_loss'][::10])
    ax[1].legend(['Surjective Loss', 'Entropy Loss'])
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    plt.show()

    