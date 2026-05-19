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

def get_reconstruction_loss(A_low, T, A_high=None, E_high=None, T_type='binary'):
    if A_high is None:
        n_high = E_high.shape[0]
        A_high = torch.eye(n_high, device=E_high.device) - torch.linalg.inv(E_high)
    if T_type == 'binary':
        column_norms = torch.sum(T,dim=0)
    elif T_type == 'simplex':
        column_norms = torch.sum(T, dim=0)
    elif T_type == 'sphere':
        column_norms = torch.sqrt(torch.sum(T**2, dim=0))
        
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

class HailMary(nn.Module):
    def __init__(self, n_low, n_high, A_low):
        super().__init__()
        self.n_low = n_low
        self.n_high = n_high
        self.register_buffer('A_low', A_low)  # (n_low, n_low) low-level weight matrix
        # One logit per (low-level var, high-level block) pair
        self.logits = nn.Parameter(torch.randn(n_low, n_high))

    def get_partition(self):
        # Soft assignment for gradients, hard for forward logic
        T_soft = F.softmax(self.logits, dim=1)  # (n_low, n_high), rows sum to 1
        indices = torch.argmax(self.logits, dim=1)
        T_hard = F.one_hot(indices, self.n_high).float()
        T = (T_hard - T_soft).detach() + T_soft  # straight-through
        return T, T_soft

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

    def forward(self):
        T, T_soft = self.get_partition()
        # Main loss: abstraction residual
        abs_loss = self.abstraction_residual(T)
        # Surjectivity: every high-level block must have at least one member
        surjective_loss = torch.sum(F.relu(1.0 - T_soft.sum(dim=0)))
        return abs_loss, surjective_loss
    
    def train_step(self, optimizer, reg_param=1.):
        optimizer.zero_grad()
        abs_loss, surjective_loss = self.forward()
        total_loss = abs_loss + reg_param * surjective_loss
        total_loss.backward()
        optimizer.step()
        return abs_loss.item(), surjective_loss.item()

    def train(self, num_epochs, reg_param=1., verbose=True,keep_history=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        history = {'abs_loss': [], 'surjective_loss': []}
        for epoch in range(num_epochs):
            abs_loss, surjective_loss = self.train_step(optimizer, reg_param)
            if epoch % 100 == 0 or epoch == num_epochs - 1:
                log(f'Epoch {epoch+1}/{num_epochs}, Abs Loss: {abs_loss:.4f}, Surjective Loss: {surjective_loss:.4f} ', verbose=verbose)
            if keep_history:
                history['abs_loss'].append(abs_loss)
                history['surjective_loss'].append(surjective_loss)
        return history

def process_output(S,T,E_high):
    n_low,n_high = T.shape
    T_soft = T.detach().cpu().numpy()
    S = S.detach().cpu().numpy()
    E_high = E_high.detach().cpu().numpy()
    A_high = enforce_dag(np.eye(n_high) - np.linalg.inv(E_high))

    indices = np.argmax(T_soft, axis=1)
    T_pred = np.zeros_like(T_soft)
    T_pred[np.arange(n_low), indices] = 1 
   
    S_pred = T_pred * S
    return S_pred, T_pred, A_high

if __name__ == "__main__":
    from utils.plots import plot_matrix, visualize_causal_graph, plot_loss_landscape
    import matplotlib.pyplot as plt
    seed = None #44
    n_high = 2  #number of high-level variables
    n_low = 10 #number of low-level variables
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
    num_epochs = 1000 * int(np.log10(n_low))
    history = model.train(num_epochs=num_epochs, reg_param=1,verbose=True, keep_history=True)
    # --- Post-Training Extraction ---
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
    A_high_pred = A_high.cpu().numpy()
    
    obs_loss = observation_loss(A_low=A_low, A_high=A_high_pred, S=S_pred, T=T_pred)
    intv_loss = intervention_loss(E_low=E.cpu().numpy(), S=S_pred, T=T_pred)
    print(f"Observation Loss: {obs_loss:.4f}, Intervention Loss: {intv_loss:.4f}")

    visualize_causal_graph(A_low, A_high_pred, T_pred, S_pred, title='Learned Causal Abstraction')
    plot_matrix(
        matrix_list=[T_pred],
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
    ax[1].legend(['Surjective Loss'])
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    plt.show()

    