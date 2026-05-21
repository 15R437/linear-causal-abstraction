import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))
from utils.generate_data import sample_low_level_graph
from utils.losses import intervention_loss, observation_loss, reconstruction_loss
from utils.helper import enforce_dag

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
    rate = (t_min / t_start)**(1 / max_epochs) - 1 if t_start > 0 else 0.0
    
    return rate

def reconstruction_loss(A_low, T, A_high):
    column_norms = torch.sum(T,dim=0)
        
    T_norm = T / (column_norms + 1e-6)
    A_recon = T @ A_high @ T_norm.T

    adj_recon_loss = torch.mean((A_recon - A_low)**2)
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

class HailMary(nn.Module):
    def __init__(self, n_low, n_high, A_low, init_assignment=None):
        super().__init__()
        self.n_low = n_low
        self.n_high = n_high
        self.register_buffer('A_low', A_low)  # (n_low, n_low) low-level weight matrix
        # One logit per (low-level var, high-level block) pair
        self.logits = nn.Parameter(torch.randn(n_low, n_high))

    def get_partition(self, temp=1.):
        # Soft assignment for gradients, hard for forward logic
        T_soft = F.softmax(self.logits / temp, dim=1)  # (n_low, n_high), rows sum to 1
        indices = torch.argmax(self.logits, dim=1)
        T_hard = F.one_hot(indices, self.n_high).float()
        T = (T_hard - T_soft).detach() + T_soft  # straight-through
        return T, T_soft
    
    def get_S(self,T):
        A_intra = T @ T.T * self.A_low
        I_low = torch.eye(self.n_low, device=T.device)
        S = torch.linalg.solve(I_low - A_intra, T)
        return S

    def abstraction_residual(self, T):
        # For each block pair (i, j), compute || A_ij s_j - a*_ij t_i ||^2
        # T: (n_low, n_high), binary (or soft)
        residual = 0.0
        for j in range(self.n_high):
            t_j = T[:, j]  # (n_low,)
            # A_jj: rows and cols in block j
            A_jj = T[:, j:j+1] * self.A_low * T[:, j:j+1].T  # soft masking
            I = torch.eye(self.n_low, device=T.device)
            s_j = torch.linalg.solve(I - A_jj, t_j)  # (I - A_jj)^{-1} t_j
            for i in range(self.n_high):
                if i == j:
                    continue
                t_i = T[:, i]  # (n_low,)
                # A_ij: effect of block j on block i
                A_ij_s_j = (T[:, i:i+1] * self.A_low * T[:, j:j+1].T) @ s_j
                # optimal a*_ij = (t_i . A_ij s_j) / ||t_i||^2
                a_star = (t_i @ A_ij_s_j) / (t_i @ t_i + 1e-8)
                residual = residual + torch.mean((A_ij_s_j - a_star * t_i) ** 2)
        return residual
    
    def vectorized_abstraction_residual(self, T):
        """
        Fully vectorized abstraction residual calculation (No loops).
        Matches the loop version mathematically component-for-component.
        """
        n_low, n_high = T.shape
        I_low = torch.eye(n_low, device=T.device)
        
        # --- 1. Compute s_j for all source clusters j simultaneously ---
        # In your loop: A_jj = T[:, j:j+1] * self.A_low * T[:, j:j+1].T
        # T.T has shape (n_high, n_low). We create a mask for each j.
        # mask_jj shape: (n_high, n_low, n_low)
        mask_jj = T.T.unsqueeze(-1) @ T.T.unsqueeze(-2)
        A_intra_3d = mask_jj * self.A_low.unsqueeze(0)
        
        # Solve (I - A_jj)^-1 @ t_j for all j in parallel
        # LHS shape: (n_high, n_low, n_low), RHS shape: (n_high, n_low, 1)
        S_batched = torch.linalg.solve(
            I_low.unsqueeze(0) - A_intra_3d, 
            T.T.unsqueeze(-1)
        )  # Shape: (n_high, n_low, 1)

        # --- 2. Compute inter-cluster effects (A_ij @ s_j) for all pairs (i, j) ---
        # In your loop: A_ij_s_j = (T[:, i:i+1] * self.A_low * T[:, j:j+1].T) @ s_j
        # We construct a 4D tensor where dim 0 is target cluster i, and dim 1 is source cluster j.
        # mask_ij shape: (n_high, n_high, n_low, n_low)
        mask_ij = T.T.unsqueeze(1).unsqueeze(-1) @ T.T.unsqueeze(0).unsqueeze(-2)
        A_inter_4d = mask_ij * self.A_low.unsqueeze(0).unsqueeze(0)
        
        # Align S_batched to be pulled from source index j (dim 1)
        # Shape for multiplication: (1, n_high, n_low, 1)
        S_expanded = S_batched.unsqueeze(0) 
        
        # Batch matrix multiply: (n_high, n_high, n_low, n_low) @ (1, n_high, n_low, 1)
        # actual_effects shape: (n_high, n_high, n_low)
        actual_effects = (A_inter_4d @ S_expanded).squeeze(-1)

        # --- 3. Compute optimal scalars (a_star) for all pairs ---
        # t_i needs to be structured to map to target index i (dim 0)
        # t_i_expanded shape: (n_high, n_high, n_low)
        t_i_expanded = T.T.unsqueeze(1).expand(-1, n_high, -1)
        
        # Dot product sums over the low-level variable axis (dim -1)
        dot_prods = torch.sum(t_i_expanded * actual_effects, dim=-1, keepdim=True)
        t_i_norms = torch.sum(t_i_expanded * t_i_expanded, dim=-1, keepdim=True)
        
        a_star = dot_prods / (t_i_norms + 1e-8)  # Shape: (n_high, n_high, 1)

        # --- 4. Compute target uniform profile and residuals ---
        target_effects = a_star * t_i_expanded  # Shape: (n_high, n_high, n_low)
        
        # Calculate the mean per block-pair matching `torch.mean((... - ...)**2)`
        squared_residuals = torch.mean((actual_effects - target_effects) ** 2, dim=-1) # (n_high, n_high)

        # --- 5. Mask out diagonal entries (i == j) and sum up the residual ---
        diag_mask = 1.0 - torch.eye(n_high, device=T.device)
        masked_residuals = squared_residuals * diag_mask
    
        return masked_residuals.sum()
    
    def improved_abstraction_residual(self, T_soft):
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
        influence_matrix = self.A_low @ (T_soft * S)  # (n_low, n_high)

        # 3. Weighted variance of influence_j over block i, for all (i,j) pairs
        # weights[r, i] = T_soft[r, i]^2
        T_sq = T_soft ** 2                                          # (n_low, n_high)
        t_i_norms = T_sq.sum(dim=0).clamp(min=1e-8)                # (n_high,) effective block sizes

        # Weighted mean: a_star[i,j] = sum_r w[r,i] * influence[r,j] / sum_r w[r,i]
        a_star = (T_sq.T @ influence_matrix) / t_i_norms.unsqueeze(1)  # (n_high, n_high)

        # diff[i, r, j] = influence[r,j] - a_star[i,j]
        # (n_high, n_low, n_high) — manageable: 50*500*50 = 1.25M floats
        diff = influence_matrix.unsqueeze(0) - a_star.unsqueeze(1)

        # Weighted variance: sum_r w[r,i] * diff[i,r,j]^2 / sum_r w[r,i]
        weights = T_sq.T.unsqueeze(2)                               # (n_high, n_low, 1)
        residuals = (weights * diff ** 2).sum(dim=1) / t_i_norms.unsqueeze(1)  # (n_high, n_high)

        # Mask diagonal and sum
        diag_mask = 1.0 - torch.eye(n_high, device=T_soft.device)
        return (residuals * diag_mask).sum()

    def get_A_high(self, T, S):
        # T, S: (n_low, n_high)
        ATS = self.A_low @ (T * S)                        # (n_low, n_high), col j = A_low @ (t_j * s_j)
        A_high = T.T @ ATS                                # (n_high, n_high), [i,j] = t_i^T @ A_low @ (t_j*s_j)
        t_norms = T.sum(dim=0).clamp(min=1e-8)            # (n_high,) = ||t_i||^2 for binary T
        A_high = A_high / t_norms.unsqueeze(1)            # normalise each row by block size
        return A_high * (1 - torch.eye(self.n_high, device=T.device))

    def forward(self,temp=1.):
        T, T_soft = self.get_partition(temp)
        S = self.get_S(T)
        A_high = self.get_A_high(T,S)
        
        abs_loss = self.improved_abstraction_residual(T_soft)
        surjective_loss = F.relu(0.51 - T_soft.max(dim=0).values).mean()#F.relu(torch.prod(1-T_soft,dim=0) - 0.1).sum()#torch.sum(F.relu(1.0 - T.sum(dim=0)))
        balance_loss = intra_cluster_penalty(self.A_low,T_soft) #reconstruction_loss(self.A_low, T_soft, A_high) #balance_penalty(T_soft)  #torch.var(torch.sum(T_soft,dim=0))
        
        return abs_loss, surjective_loss, balance_loss

    def train_step(self, optimizer, reg_param=1., temp=1.):
        optimizer.zero_grad()
        abs_loss, surjective_loss, balance_loss = self.forward(temp)
        total_loss = balance_loss + reg_param * (surjective_loss + abs_loss)
        total_loss.backward()
        optimizer.step()
        return abs_loss.item(), surjective_loss.item(), balance_loss.item()

    def train(self, num_epochs, reg_param=1., verbose=True,keep_history=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        history = {'abs_loss': [], 'surjective_loss': [], 'balance_loss': []} if keep_history else None
        for epoch in range(num_epochs):
            temp = max(0.05, 1.0 * (0.05**(epoch/num_epochs))) 
            abs_loss, surjective_loss, balance_loss = self.train_step(optimizer, reg_param, temp)
            if epoch % 100 == 0 or epoch == num_epochs - 1:
                log(f'Epoch {epoch+1}/{num_epochs}, Abs Loss: {abs_loss:.4f}, Surjective Loss: {surjective_loss:.4f}, Balance Loss: {balance_loss:.4f} ', verbose=verbose)
            if keep_history:
                history['abs_loss'].append(abs_loss)
                history['surjective_loss'].append(surjective_loss)
                history['balance_loss'].append(balance_loss)
        return history

def process_output(model):
    model.eval()
    with torch.no_grad():
        # 1. Recover T
        indices = torch.argmax(model.logits, dim=1)
        T = F.one_hot(indices, model.n_high).float()
        
        # 2. Recover S
        A_intra = T @ T.T * model.A_low
        I_low = torch.eye(model.n_low, device=T.device)
        S = torch.linalg.solve(I_low - A_intra, T)
        
        # 3. Recover A_high (A_tilde)
        A_high = torch.zeros(model.n_high, model.n_high, device=T.device)
        for j in range(model.n_high):
            s_j = S[:, j]
            for i in range(model.n_high):
                if i != j:
                    t_i = T[:, i]
                    A_ij_s_j = (T[:, i:i+1] * model.A_low * T[:, j:j+1].T) @ s_j
                    A_high[i, j] = (t_i @ A_ij_s_j) / (t_i @ t_i + 1e-8)

    # Convert to numpy for your evaluation downstream
    T_pred = T.cpu().numpy()
    S_pred = S.cpu().numpy()
    A_high_pred = enforce_dag(A_high.cpu().numpy())
    return S_pred, T_pred, A_high_pred

if __name__ == "__main__":
    from utils.plots import plot_matrix, visualize_causal_graph, plot_loss_landscape
    import matplotlib.pyplot as plt
    import time
    seed = None #44
    n_high = 50  #number of high-level variables
    n_low = 500 #number of low-level variables
    density = 0.4

    A_low, T_gt, S_gt, A_high_gt = sample_low_level_graph(n_high,n_low,density,seed=seed,shuffle=True)
    plot_matrix(
        matrix_list=[A_high_gt,T_gt,S_gt],
        title_list=['High-Level Adjacency','Endogenous Abstraction T','Exogenous Abstraction S'],
        fig_title='Ground Truth Matrices'
    )
    visualize_causal_graph(A_low, A_high_gt, T_gt, S_gt, title='Ground Truth Causal Abstraction')

    E = torch.linalg.inv(torch.eye(n_low) - torch.from_numpy(A_low).float())
    #plot_loss_landscape(E, T_gt, idx1=0, idx2=2, flip_prob=0.5)
    model = HailMary(n_low, n_high, A_low=torch.from_numpy(A_low).float())
    num_epochs = 1000 * int(np.log10(n_low)+1)
    start_time = time.time()
    history = model.train(num_epochs=1000, reg_param=1e3,verbose=True, keep_history=True)
    runtime = time.time() - start_time
    T_probs, _ = model.get_partition()
    S_pred, T_pred, A_high_pred = process_output(model)
    
    
    obs_loss = observation_loss(A_low=A_low, A_high=A_high_pred, S=S_pred, T=T_pred)
    intv_loss = intervention_loss(E_low=E.cpu().numpy(), S=S_pred, T=T_pred)
    print(f"Time: {runtime:.4f}, Observation Loss: {obs_loss:.4f}, Intervention Loss: {intv_loss:.4f}, Is Surjective?: {T_pred.sum(axis=0).min() > 0}")

    visualize_causal_graph(A_low, A_high_pred, T_pred, S_pred, title='Learned Causal Abstraction')
    plot_matrix(
        matrix_list=[T_probs.cpu().detach().numpy()],
        title_list=['Learned T'],
        fig_title='Learned Matrices'
    )
    #visualize_causal_graph(A_low, A_high_pred, T_pred, S_pred, title='Learned Causal Abstraction')
    
    #Plot a graph of loss vs epochs
    fig, ax = plt.subplots(1,2, figsize=(10,8))

    ax[0].plot(range(0,len(history['abs_loss']),10), history['abs_loss'][::10])
    ax[0].legend(['Abstraction Loss'])
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")

    ax[1].plot(range(0,len(history['surjective_loss']),10), history['surjective_loss'][::10])
    ax[1].plot(range(0,len(history['balance_loss']),10), history['balance_loss'][::10])
    ax[1].legend(['Surjective Loss', 'Balance Loss'])
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    plt.show()

    