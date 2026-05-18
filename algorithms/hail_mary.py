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

class HailMary(nn.Module):
    def __init__(self, n_low, n_high, E, T_type='binary', init_assignment=None, threshold=0.5):
        super().__init__()
        self.n_low = n_low
        self.n_high = n_high
        self.E = E
        self.A_low = torch.eye(n_low, device=E.device) - torch.linalg.inv(E)
        self.device = E.device
        self.W_param = nn.Parameter(torch.randn(n_low, n_high,device=self.device))
        self.A_param = nn.Parameter(torch.randn(n_low, n_high,device=self.device))
        self.S = nn.Parameter(torch.randn(n_low, n_high,device=self.device))
        self.threshold = threshold
        self.T_type = T_type
        if T_type == 'binary':
            self.T = torch.ones(n_low, n_high, device=self.device)
        else:
            self.T = nn.Parameter(torch.randn(n_low, n_high, device=self.device))

        A_mask = torch.ones(n_low, n_high,device=self.device)
        if init_assignment:
            assignment_tensor = torch.tensor(init_assignment, device=self.device).long()
            valid_mask = (assignment_tensor < self.n_high) & (assignment_tensor != -1)
            low_level_indices = torch.where(valid_mask)[0]
            high_level_indices = assignment_tensor[valid_mask]
            A_mask[low_level_indices] = 0.0
            A_mask[low_level_indices, high_level_indices] = 1.0
        else:
        #fix the first low-level variable to the first high-level variable
            A_mask[0] = 0
            A_mask[0,0] = 1.0

        self.register_buffer('A_mask', A_mask)
        self.register_buffer('E_high_mask',torch.ones(n_high, n_high,device=self.device) - torch.eye(n_high, device=self.device))
    
    def get_params(self,W=None):
        E = self.E
        if W is None:
            W = torch.sigmoid(self.W_param) #stick_breaking(torch.sigmoid(self.W_param)) * self.W_mask
            A = torch.sigmoid(self.A_param)*self.A_mask
        else:
            A = W.clone()
        

        if self.T_type == 'binary':
            T_soft = W * self.T
        elif self.T_type == 'simplex':
            logits = torch.log(W) # near 0 values should be very negative 
            T_soft = F.softmax(logits + self.T, dim=0)
            T_soft = torch.nan_to_num(T_soft, nan=0.0)
        elif self.T_type == 'sphere':
            T_soft = F.normalize(W * self.T, p=2, dim=0, eps=1e-8)
        
        indices = torch.argmax(W, axis=1)
        mask = torch.zeros_like(W)
        mask[torch.arange(self.n_low), indices] = 1
        T_hard = mask
        T = T_soft #(T_hard - T_soft).detach() + T_soft

        I = torch.eye(self.n_low,device=self.device)
        F_pi = torch.diag_embed(T.T) #(n_high, n_low, n_low)
    
        term = I + F_pi @ (E - I)
        ET = E @ T
        ET_batch = ET.unsqueeze(0).expand(self.n_high,-1,-1)
        T_batch = T.unsqueeze(0).expand(self.n_high, -1, -1)
        term_T = torch.bmm(term, T_batch)  # (n_high, n_low, n_high)
        lhs = torch.bmm(F_pi,ET_batch)
        rhs_col = term_T[torch.arange(self.n_high), :, torch.arange(self.n_high)]  # (n_high, n_low)
        rhs = rhs_col.unsqueeze(-1).expand(-1, -1, self.n_high)  # (n_high, n_low, n_high)

        dot_prod = torch.sum(lhs * rhs, dim=1)
        a_norm = torch.sum(lhs * lhs, dim=1)
        b_norm = torch.sum(rhs * rhs, dim=1)
        safe_b = torch.where(b_norm > 0, b_norm, torch.ones_like(b_norm))
        E_high_ij = torch.where(b_norm > 0, dot_prod / safe_b, torch.zeros_like(dot_prod))
        E_high =  E_high_ij * self.E_high_mask + torch.eye(self.n_high, device=self.device)

        S_soft = T_soft * torch.linalg.solve(E_high.T, (ET).T).T
        S_hard = mask * torch.sum(S_soft, dim=1, keepdim=True)
        S = (S_hard - S_soft).detach() + S_soft
        
        return S_soft, T_soft, E_high, W, A

    def get_S(self,E_high,A,T):
        s_mask = (A > self.threshold)
        S = torch.linalg.solve(E_high.T, (self.E @ T).T).T * s_mask
        return S
    
    def forward(self,W=None):
        S, T, E_high, W, A = self.get_params(W=W) # T,S shape: (n_low, n_high); W,A shape: (n_low, n_high-1)
        E = self.E
        A_low = self.A_low

        # encourage DAG structure, row-wise partition and consistency between S and T
        structural_constraint = torch.det(E_high - torch.eye(self.n_high,device=self.device))**2 + torch.mean(((1-T) * S)**2) #+ torch.mean((W.sum(dim=1) - 1)**2)  

        indices = torch.argmax(T, axis=1)
        mask = torch.zeros_like(T)
        mask[torch.arange(self.n_low), indices] = 1
        T_hard = (mask - T).detach() + T
        col_sum, soft_sum = T_hard.sum(dim=0), T.sum(dim=0)
        surjective_loss = torch.sum(F.relu(1.0 - col_sum)) #+ ((soft_sum - self.n_low/self.n_high)**2).sum()/self.n_low
        constraint_loss = torch.mean((W - F.relu(W + A - self.threshold - 1) - F.relu(W + A - self.threshold))**2)
        try:
            regularisation_loss = F.binary_cross_entropy(W,A)
        except:
            import pdb;pdb.set_trace()
        recon_loss = intra_cluster_dist(A_low,T)/inter_cluster_dist(A_low,T)#torch.log(intra_cluster_dist(A_low,T))  - torch.log(inter_cluster_dist(A_low,T))

        I = torch.eye(self.n_low,device=self.device)
        I_high = torch.eye(self.n_high,device=self.device)
        F_pi = torch.diag_embed(T.T) #(n_high, n_low, n_low)
    
        term = I + F_pi @ (E - I)
        
        T_batch = T.unsqueeze(0).expand(self.n_high, -1, -1)
        S_batch = S.unsqueeze(0).expand(self.n_high, -1, -1)
        lhs = F_pi @ S_batch
        rhs = term @ (F_pi @ T_batch)
        

        intv_loss = torch.mean((lhs-rhs)**2) * self.n_high
        obs_loss = torch.mean((S @ E_high - E @T)**2)
        """"
        T_batch = T.unsqueeze(0).expand(self.n_high, -1, -1) 
        S_batch = S.unsqueeze(0).expand(self.n_high, -1, -1)
        lhs = torch.bmm(F_pi, S_batch)
        rhs = term @ torch.bmm(F_pi, T_batch)
        ET = E @ T
        A_high = torch.linalg.lstsq(ET, ET - S).solution * self.A_high_mask
        """
        
        
        """
        lhs = torch.bmm(F_pi,ET_batch)
        rhs_col = term_T[torch.arange(self.n_high), :, torch.arange(self.n_high)]  # (n_high, n_low)
        rhs = rhs_col.unsqueeze(-1).expand(-1, -1, self.n_high)  # (n_high, n_low, n_high)

        dot_prod = torch.sum(lhs * rhs, dim=1)
        a_norm = torch.sum(lhs * lhs, dim=1)
        b_norm = torch.sum(rhs * rhs, dim=1)
        safe_b = torch.where(b_norm > 0, b_norm, torch.ones_like(b_norm))
        E_high_ij = torch.where(b_norm > 0, dot_prod / safe_b, torch.zeros_like(dot_prod))

        
        E_high_mask = torch.triu(torch.ones_like(E_high_ij), diagonal=1)
        E_high =  E_high_ij * E_high_mask + I_high
        #E_high = (E_high_ij * E_high_mask + I_high - E_high_ij).detach() + E_high_ij
        #gram_loss = torch.mean((a_norm * b_norm) - dot_prod ** 2)
        mse_loss = torch.mean((lhs - E_high.unsqueeze(1) * rhs)**2)
        intv_loss = torch.relu(mse_loss * self.n_high) #torch.relu(mse_loss).mean()

        S = self.get_S(E_high=E_high,A=A,T=T)
        obs_loss = torch.mean((S @ E_high - self.E @ T)**2)
        """

        return recon_loss,obs_loss, intv_loss, surjective_loss, constraint_loss, regularisation_loss, structural_constraint
    
    def train_step(self, optimizer, abs_param=1, reg_param=1.):
        optimizer.zero_grad()
        recon_loss, obs_loss, intv_loss, surjective_loss, constraint_loss, regularisation_loss, structural_constraint = self.forward()
        total_loss = recon_loss + abs_param * (obs_loss + intv_loss) + reg_param * (surjective_loss + constraint_loss + regularisation_loss)
        total_loss.backward()
        optimizer.step()
        return recon_loss.item(), obs_loss.item(), intv_loss.item(), surjective_loss.item(), constraint_loss.item(), regularisation_loss.item(), structural_constraint.item()

    def train(self, num_epochs, abs_param=1, reg_min=1., reg_max=1., verbose=True,keep_history=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        history = {'recon_loss': [], 'obs_loss': [], 'intv_loss': [], 'surjective_loss': [], 'constraint_loss': [], 'regularisation_loss': [], 'structural_constraint': []}
        reg_param = reg_min
        anneal_rate = calculate_anneal_rate(reg_min, reg_max, num_epochs)
        for epoch in range(num_epochs):
            recon_loss, obs_loss, intv_loss, surjective_loss, constraint_loss, regularisation_loss, structural_constraint = self.train_step(optimizer, abs_param, reg_param)

            if epoch % 100 == 0 or epoch == num_epochs - 1:
                log(f'Epoch {epoch+1}/{num_epochs}, Recon Loss: {recon_loss:.4f}, Obs Loss: {obs_loss:.4f}, Intv Loss: {intv_loss:.4f}, Surjective Loss: {surjective_loss:.4f}, Constraint Loss: {constraint_loss:.4f}, Regularisation Loss: {regularisation_loss:.4f}, Structural Constraint: {structural_constraint:.4f} ', verbose=verbose)
            if keep_history:
                history['recon_loss'].append(recon_loss)
                history['obs_loss'].append(obs_loss)
                history['intv_loss'].append(intv_loss)
                history['surjective_loss'].append(surjective_loss)
                history['constraint_loss'].append(constraint_loss)
                history['regularisation_loss'].append(regularisation_loss)
                history['structural_constraint'].append(structural_constraint)
                
            reg_param = reg_param * (1+anneal_rate)
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
    plot_loss_landscape(E, T_gt, idx1=0, idx2=2, flip_prob=0.5)

    model = HailMary(n_low, n_high, E)
    num_epochs = 1000 * int(np.log10(n_low))
    history = model.train(num_epochs=num_epochs, abs_param=1e4, reg_min=1, reg_max=1,verbose=True, keep_history=True)
    S,T, E_high, W, A = model.get_params()
    S_pred, T_pred, A_high_pred = process_output(S,T,E_high)
    
    obs_loss = observation_loss(A_low=A_low, A_high=A_high_pred, S=S_pred, T=T_pred)
    intv_loss = intervention_loss(E_low=E.cpu().numpy(), S=S_pred, T=T_pred)
    print(f"Observation Loss: {obs_loss:.4f}, Intervention Loss: {intv_loss:.4f}")

    visualize_causal_graph(A_low, A_high_pred, T_pred, S_pred, title='Learned Causal Abstraction')
    plot_matrix(
        matrix_list=[W.detach().cpu().numpy(),A.detach().cpu().numpy()],
        title_list=['LearnedW','Learned A'],
        fig_title='Learned Matrices'
    )
    visualize_causal_graph(A_low, A_high_pred, T_pred, S_pred, title='Learned Causal Abstraction')
    
    #Plot a graph of loss vs epochs
    fig, ax = plt.subplots(1,2, figsize=(10,8))

    ax[0].plot(range(0,len(history['recon_loss']),10), history['recon_loss'][::10])
    ax[0].plot(range(0,len(history['obs_loss']),10), history['obs_loss'][::10])
    ax[0].plot(range(0,len(history['intv_loss']),10), history['intv_loss'][::10])
    ax[0].legend(['Reconstruction Loss','Observed Loss','Interventional Loss'])
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")

    ax[1].plot(range(0,len(history['regularisation_loss']),10), history['regularisation_loss'][::10])
    ax[1].plot(range(0,len(history['constraint_loss']),10), history['constraint_loss'][::10])
    ax[1].plot(range(0,len(history['surjective_loss']),10), history['surjective_loss'][::10])
    ax[1].plot(range(0,len(history['structural_constraint']),10), history['structural_constraint'][::10])
    ax[1].legend(['Regularisation Loss','Constraint Loss','Surjective Loss','Structural Constraint'])
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    plt.show()

    