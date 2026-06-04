import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))
from utils.generate_data import sample_low_level_graph
from utils.losses import intervention_loss, observation_loss, abstraction_residual
from utils.helper import enforce_dag, is_dag

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

def inter_block_recon_penalty(A_low, T, A_high):
    # A_hat approximates the inter-block part of A_low
    with torch.no_grad():
        T_norm = T / (T**2).sum(dim=0).clamp(min=1e-8)
    A_hat = T @ A_high @ T_norm.T
    
    # Only measure error on inter-block entries
    inter_mask = 1.0 - T @ T.T
    diag_mask = 1.0 - torch.eye(A_low.shape[0], device=A_low.device)
    inter_mask = inter_mask * diag_mask
    
    return torch.mean(((A_low - A_hat) * inter_mask) ** 2)

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
    
    # 3. Calculate the exact structural margin needed
    # Protects against ln(0) if n_high happens to be 1
    min_margin = math.log(max(1, n_high - 1)) 
    
    # Add a 5-sigma buffer to ensure noise doesn't pull the softmax below 0.5
    margin = min_margin + (5 * noise_std) + 0.1
    
    # 4. Inject the margin into the targeted positions
    logits[torch.arange(n_low), targets] += margin
    
    return logits
        

class HailMary(nn.Module):
    def __init__(self, n_low, n_high, A_low, logits_init=None, init_assignment=None):
        super().__init__()
        self.n_low = n_low
        self.n_high = n_high
        self.register_buffer('A_low', A_low)  # (n_low, n_low) low-level weight matrix
        out_degrees = (self.A_low != 0).float().sum(dim=1)
        self.register_buffer('sink_weights', 1.0 / (1.0 + out_degrees)) # 1.0 for pure sinks, approaches 0 for heavy sources

        # One logit per (low-level var, high-level block) pair
        self.logits = nn.Parameter(regularised_surjective_init(self.n_low, self.n_high)) if logits_init is None else nn.Parameter(logits_init)
        self.node_positions = nn.Parameter(torch.linspace(0, self.n_high, self.n_high))
    
    def dag_mask(self,temp=1.):
        diff = self.node_positions.unsqueeze(0) - self.node_positions.unsqueeze(1)
        hard_mask = (diff > 0).float()
        soft_mask = torch.sigmoid(diff/temp)
        dag_mask = soft_mask + (hard_mask - soft_mask).detach()
        #dag_mask = torch.triu(torch.ones(self.n_high,self.n_high),diagonal=1)
        return dag_mask, soft_mask

    def get_partition(self, temp=1.):
        # Soft assignment for gradients, hard for forward logic
        T_soft = F.softmax(self.logits / temp, dim=1)  # (n_low, n_high), rows sum to 1
        indices = torch.argmax(self.logits, dim=1)
        T_hard = F.one_hot(indices, self.n_high).float()
        T = (T_hard - T_soft).detach() + T_soft  # straight-through
        T_gumbel = F.gumbel_softmax(self.logits, tau=temp, hard=True, dim=1)
        return T, T_soft, T_gumbel
    
    def get_S(self,T):
        A_intra = T @ T.T * self.A_low
        I_low = torch.eye(self.n_low, device=T.device)
        S = torch.linalg.solve(I_low - A_intra, T)
        return S
    
    def get_A_high(self, T, S):
        # T, S: (n_low, n_high)
        ATS = self.A_low @ (T * S)                        # (n_low, n_high), col j = A_low @ (t_j * s_j)
        A_high = T.T @ ATS                                # (n_high, n_high), [i,j] = t_i^T @ A_low @ (t_j*s_j)
        t_norms = T.sum(dim=0).clamp(min=1e-8)            # (n_high,) = ||t_i||^2 for binary T
        A_high = A_high / t_norms.unsqueeze(1)
        A_high = A_high * (1 - torch.eye(self.n_high, device=T.device))           # normalise each row by block size
        #A_high_dag = enforce_dag(A_high)
        return A_high * self.dag_mask()[0]

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
    
    def improved_abstraction_residual(self, T_soft, mask):
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
        a_star = mask * (T_sq.T @ influence_matrix) / t_i_norms.unsqueeze(1)  # (n_high, n_high)

        # diff[i, r, j] = influence[r,j] - a_star[i,j]
        # (n_high, n_low, n_high) — manageable: 50*500*50 = 1.25M floats
        diff = influence_matrix.unsqueeze(0) - a_star.unsqueeze(1)

        # Weighted variance: sum_r w[r,i] * diff[i,r,j]^2 / sum_r w[r,i]
        weights = T_sq.T.unsqueeze(2)                               # (n_high, n_low, 1)
        residuals = (weights * diff ** 2).sum(dim=1) / t_i_norms.unsqueeze(1)  # (n_high, n_high)

        # Mask diagonal and sum
        diag_mask = 1.0 - torch.eye(n_high, device=T_soft.device)
        return (residuals * diag_mask).sum()


    def forward(self,temp=1.,T=None):
        if T is None:
            T_hard, T_soft, _ = self.get_partition(temp) # soft T
            T = T_soft
        else:
            T_hard = T
            
        S = self.get_S(T)
        dag_mask, soft_mask = self.dag_mask(temp)
        A_high = self.get_A_high(T,S)
        
        column_maxes = T.max(dim=0).values  
        col_deficit = F.relu(0.5 - column_maxes)  
        surjective_loss = (col_deficit * (-torch.log(T + 1e-8).mean(dim=0))).mean()
        #surjective_loss = F.relu(.5 - T.max(dim=0).values).mean() #torch.var(torch.sum(T_soft,dim=0)) #F.relu(torch.prod(1-T_soft,dim=0) - 0.1).sum()#torch.sum(F.relu(1.0 - T.sum(dim=0)))
        
        abs_loss = self.improved_abstraction_residual(T,dag_mask)
        balance_loss = inter_block_recon_penalty(self.A_low,T,A_high) #reconstruction_loss(self.A_low, T_soft, A_high) #balance_penalty(T_soft)  
        #dag_loss = dag_penalty_poly(A_high)
        #entropy_loss = F.relu(1. - T_soft.max(dim=1).values).mean()
        base_entropy = -torch.sum(T * torch.log(T + 1e-8), dim=1)
        entropy_loss = (base_entropy * (1.0 + 2.0 * self.sink_weights)).mean()
        #entropy_loss = -torch.sum(T * torch.log(T + 1e-8), dim=1).mean()
        
        A_low_binary = (self.A_low != 0).float()
        M = T.T @ A_low_binary @ T # M[i,j] is the number of edges between cluster i and cluster j
        topo_loss = torch.mean((1 - dag_mask - torch.eye(self.n_high)) * M ** 2) 

        intra_loss = intra_cluster_penalty(A_low_binary, T)

       
        H = A_low_binary @ T # H[i,j] is the number of edges between node i and cluster j
        one = H/(1+H) + (1 - H/(1+H)).detach()
        C = one - F.relu(one - H) # C[i,j] = 0 iff node i is not connected to cluster j otherwise C[i,j] = 1
        CC = C @ C.T # CC[i,j] = number of shared cluster connections between node i and node j
        CC_norm = (C ** 2).sum(dim=1)                         # (n_low,)
        C_diff_sq = CC_norm.unsqueeze(1) + CC_norm.unsqueeze(0) - 2 * CC  # ||C[i] - C[i']||^2

        co_assign = T @ T.T                                   # (n_low, n_low)
        diag_mask = 1 - torch.eye(self.n_low, device=T.device)
        graph_loss = (co_assign * diag_mask * C_diff_sq).mean()
        
        """
        numerator = T.T @ C # numerator[i,j] = 0 iff cluster i is not connected to cluster j
        denominator = T.sum(dim=0, keepdim=True).T.clamp(min=1e-8)
        P = numerator / denominator  # Matrix of proportions between [0, 1]

        mask = 1.0 - torch.eye(self.n_high, device=T.device)
        graph_loss = (P * (1.0 - P) * mask).sum()
        """
        
        return balance_loss, abs_loss, graph_loss, topo_loss, surjective_loss, entropy_loss

    def train_step(self, optimiser, lambda_abs=1., lambda_graph=1., lambda_topo=1., lambda_surj=1., lambda_entropy=1., temp=1.):
        optimiser.zero_grad()
        balance_loss, abs_loss, graph_loss, topo_loss, surjective_loss, entropy_loss = self.forward(temp)
        total_loss = (balance_loss + 
                      (lambda_abs) * abs_loss + 
                      (lambda_graph) * graph_loss +
                      (lambda_topo) * topo_loss +
                      (lambda_surj) * surjective_loss + 
                      (lambda_entropy) * entropy_loss 
                      )
        total_loss.backward()
        optimiser.step()
        return balance_loss.item(), abs_loss.item(), graph_loss.item(), topo_loss.item(), surjective_loss.item(), entropy_loss.item()
    
    def loss_tracker(self, loss, alpha=0.1, patience=50, min_delta=1e-4):
        pass

    def train(self, num_epochs, lambda_abs=0., lambda_graph=1, lambda_topo=1., lambda_surj=1, lambda_min_entropy=1e-2, lambda_max_entropy=1e-2, max_temp=1.0, min_temp=1.0, verbose=True,keep_history=False):
        optimiser = torch.optim.Adam(self.parameters(), lr=0.01)
        history = {'balance_loss':[], 'abs_loss': [], 'graph_loss':[], 'topo_loss':[], 'surjective_loss': [], 'entropy_loss': [], 'T':[], 'soft_mask':[]} if keep_history else None
        lambda_entropy = lambda_max_entropy
        rate = calculate_anneal_rate(lambda_max_entropy, lambda_min_entropy, num_epochs)
        for epoch in range(num_epochs):
            lambda_entropy *= rate
            balance_loss, abs_loss, graph_loss, topo_loss, surjective_loss, entropy_loss = self.train_step(optimiser, lambda_abs, lambda_graph, lambda_topo, lambda_surj, lambda_entropy)
            if epoch % 100 == 0 or epoch == num_epochs - 1:
                log(f"""Epoch {epoch+1}/{num_epochs},
                        Balance: {balance_loss:.4f},
                        Abstraction: {abs_loss:.4f}, 
                        Graph: {graph_loss:.4f},
                        Topo: {topo_loss:.4f}
                        Surjective: {surjective_loss:.4f},
                        Entropy: {entropy_loss} """, verbose=verbose)
            if keep_history:
                history['balance_loss'].append(balance_loss)
                history['abs_loss'].append(abs_loss * lambda_abs)
                history['graph_loss'].append(graph_loss * lambda_graph)
                history['topo_loss'].append(topo_loss * lambda_topo)
                history['surjective_loss'].append(surjective_loss * lambda_surj)
                history['entropy_loss'].append(entropy_loss * lambda_entropy)
                history['T'].append(self.get_partition()[1].cpu().detach().numpy())
                history['soft_mask'].append(self.dag_mask()[1].cpu().detach().numpy())
        return history

def process_output(model,min_temp=0.05):
    model.eval()
    with torch.no_grad():
        # 1. Recover T
        T,T_soft, T_gumbel = model.get_partition(min_temp)
        S = model.get_S(T)
        A_high = model.get_A_high(T,S)

    A_high_np = A_high.cpu().detach().numpy()

    # Convert to numpy for your evaluation downstream
    T_pred = T.cpu().detach().numpy()
    S_pred = S.cpu().detach().numpy()
    #A_high_pred = enforce_dag(A_high_pred)
    A_high_pred = A_high_np.copy()
    return S_pred, T_pred, A_high_pred, A_high_np

def diagnose_matrix(name, M):
    print(f"=== Diagnostics for {name} ===")
    print(f"Shape: {M.shape}")
    print(f"Sparsity: {(M == 0).sum() / M.size:.2%} of entries are exactly 0")
    print(f"Range: [Min: {M.min():.4f}, Max: {M.max():.4f}]")
    print(f"Mean (Abs): {np.mean(np.abs(M)):.4f} | Std: {np.std(M):.4f}")
    
    # Condition number (inf if singular/not square, but great for square causal matrices)
    if M.shape[0] == M.shape[1]:
        cond = np.linalg.cond(M)
        print(f"Condition Number: {cond:.4f} (High = numerically unstable)")
        
        # Check eigenvalues
        eigenvalues = np.linalg.eigvals(M)
        print(f"Max Eigenvalue Magnitude: {np.max(np.abs(eigenvalues)):.4f}")
    print("\n")

if __name__ == "__main__":
    from utils.plots import plot_matrix, visualize_causal_graph, plot_loss_landscape, plot_assignment_raster, animate_matrices
    from utils.generate_data import generate_easy_benchmark, generate_generalized_benchmark
    from utils.helper import learn_dag
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
        S_gt = get_S(A_low=A_low, T=T_gt)
        A_high_gt = get_A_high(A_low=A_low, T=T_gt, priority='abstraction')

        
    elif DATA_TYPE=='easy_benchmark':
        seed = None #44
        n_high = 2  #number of high-level variables
        n_low = 10 #number of low-level variables
        cluster_balance = 0.1
        blockiness = 0.5
        A_low, T_gt, S_gt, A_high_gt = generate_easy_benchmark(n_low=n_low,cluster_balance=cluster_balance,blockiness=blockiness,shuffle=False) #generate_generalized_benchmark(n_low=n_low,n_high=n_high,blockiness=1.0,shuffle=True,seed=seed)
    
    elif DATA_TYPE == 'standard_benchmark':
        seed = None #44
        n_high = 50  #number of high-level variables
        n_low = 200 #number of low-level variables
        density = 0.4
        internal = False
        A_low, T_gt, S_gt, A_high_gt = sample_low_level_graph(n_high,n_low,density,seed=seed,shuffle=True,internal=internal)

    # Ground Truth Visualisation & Diagnostics 
    print(f"n_low = {n_low}, n_high = {n_high}")
    E = torch.linalg.inv(torch.eye(n_low) - torch.from_numpy(A_low).float())
    plot_matrix(
        matrix_list=[T_gt],
        title_list=['Endogenous Abstraction T'],
        fig_title='Ground Truth Matrices'
    )
    visualize_causal_graph(A_low, A_high_gt, T_gt, S_gt, title='Ground Truth Causal Abstraction')

    obs_loss_gt = observation_loss(A_low=A_low,A_high=A_high_gt,S=S_gt,T=T_gt)
    intv_loss_gt = intervention_loss(E_low=E.cpu().numpy(), S=S_gt, T=T_gt)
    abs_residual_gt = abstraction_residual(A_low=A_low, T=T_gt, A_high=A_high_gt)
    print(f"Ground Truth Metrics: Observation Loss: {obs_loss_gt:.4f}, Intervention Loss: {intv_loss_gt:.4f}, Abstraction Residual: {abs_residual_gt}, Is Surjective?: {T_gt.sum(axis=0).min() > 0}, Is DAG?: {is_dag(A_high_gt)}")
    if n_high <= 2:
       #plot_loss_landscape(A_low=torch.from_numpy(A_low).float(), T_gt=T_gt, idx1=0, idx2=1, flip_prob=0.5)
       pass

    # Training
    min_temp = 1.
    flip_prop = 0.8
    flip_indices = np.random.choice(np.arange(n_low),size=int(n_low*flip_prop),replace=False)
    T_init = T_gt.copy()
    T_init[flip_indices] = 1 - T_init[flip_indices]
    T_init = None #torch.from_numpy(T_init).float()
    model = HailMary(n_low, n_high, A_low=torch.from_numpy(A_low).float(),logits_init=T_init)
    num_epochs = 1000 * int(np.log(n_low*n_high)+1)
    start_time = time.time()
    history = model.train(num_epochs=num_epochs, min_temp=min_temp,max_temp=min_temp, verbose=True, keep_history=True)
    runtime_hm = time.time() - start_time
    T, T_soft, T_gumbel = model.get_partition(min_temp)
    _,model_abs_loss, _, _, _,_ = model(min_temp)
    print(f"Model Abs Loss: {model_abs_loss.item()}, Row-wise MinMax of T_soft: {torch.min(torch.max(T_soft,dim=1).values).item()}")
    
    #learner, history_dag = learn_dag(A_low=torch.from_numpy(A_low).float(), T=T.cpu().detach(),verbose=True, keep_history=True)
    #dag_mask, _ = learner.dag_mask()

    S_pred, T_pred, A_high_pred, A_high_np = process_output(model)
    #A_high_pred = dag_mask.cpu().detach().numpy() * A_high_pred

    # Evaluation
    obs_loss = observation_loss(A_low=A_low, A_high=A_high_pred, S=S_pred, T=T_pred)
    intv_loss = intervention_loss(E_low=E.cpu().numpy(), S=S_pred, T=T_pred)
    abs_residual = abstraction_residual(A_low=A_low,T=T_pred, A_high=A_high_pred)
    print(f"(Hail Mary) Time: {runtime_hm:.4f}, Observation Loss: {obs_loss:.4f}, Intervention Loss: {intv_loss:.4f}, Abstraction Residual: {abs_residual}, Is Surjective?: {T_pred.sum(axis=0).min() > 0}, Is DAG?: {is_dag(A_high_np)}")

    # Hierarchical Clustering Baseline
    start_time = time.time()
    S_dict,T_dict,A_high_dict = hierarchical_clustering(A_low=A_low, n_high=n_high)
    runtime_hc = time.time() - start_time
    obs_loss = observation_loss(A_low=A_low, A_high=A_high_dict[n_high], S=S_dict[n_high], T=T_dict[n_high])
    intv_loss = intervention_loss(E_low=E.cpu().numpy(), S=S_dict[n_high], T=T_dict[n_high])
    abs_residual = abstraction_residual(A_low=A_low,T=T_dict[n_high], A_high=A_high_dict[n_high])
    print(f"(HC) Time: {runtime_hc:.4f}, Observation Loss: {obs_loss:.4f}, Intervention Loss: {intv_loss:.4f}, Abstraction Residual: {abs_residual}, Is Surjective?: {T_dict[n_high].sum(axis=0).min() > 0}, Is DAG?: {is_dag(A_high_dict[n_high])}")
    
    # Visualising Hail Mary Results
    #plot_assignment_raster(history,T_gt=T_gt)
    animate_matrices(history['soft_mask'], title='dag_mask',cmap='Blues')
    animate_matrices(history['T'],title='abstraction',cmap='Reds')
    visualize_causal_graph(A_low, A_high_pred, T_pred, S_pred, title='Learned Causal Abstraction')
    plot_matrix(
        matrix_list=[T_soft.cpu().detach().numpy()],
        title_list=['Learned T'],
        fig_title='Learned Matrices'
    )
    #visualize_causal_graph(A_low, A_high_pred, T_pred, S_pred, title='Learned Causal Abstraction')
    
    #Plot a graph of loss vs epochs
    fig, ax = plt.subplots(1,2, figsize=(10,8))

    ax[0].plot(range(0,len(history['abs_loss']),10), history['abs_loss'][::10])
    ax[0].plot(range(0,len(history['topo_loss']),10), history['topo_loss'][::10])
    ax[0].plot(range(0,len(history['graph_loss']),10), history['graph_loss'][::10])
    ax[0].legend(['Abstraction Loss', 'Topo Loss', 'Graph Loss'])
    ax[0].set_yscale('log')
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")

    ax[1].plot(range(0,len(history['balance_loss']),10), history['balance_loss'][::10])
    ax[1].plot(range(0,len(history['surjective_loss']),10), history['surjective_loss'][::10])
    ax[1].plot(range(0,len(history['entropy_loss']),10), history['entropy_loss'][::10])
    ax[1].legend(['Balance Loss', 'Surjective Loss', 'Entropy Loss'])
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    plt.show()

    