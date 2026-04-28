import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import time

import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))

from utils.plots import plot_matrix, visualize_causal_graph
from utils.generate_data import sample_low_level_graph
from utils.losses import abstraction_loss, reconstruction_loss

class GradSplit(nn.Module):
    def __init__(self,E,partition_mask,split_idx=0,upper=True):
        """
        Parameters:
            E: (n_low, n_low) tensor representing the low-level environment matrix.
            partition_mask: (n_low, n_high-1) binary tensor indicating the current partition of low-level variables into high-level clusters.
            split_idx: The index of the cluster to split
        """
        super().__init__()
        self.E = E
        self.n_low = E.shape[0]
        self.n_high = partition_mask.shape[1] + 1 # we add 1 because we are splitting one cluster into two
        self.device = E.device
        self.thresh = 0.5
        self.split_idx = split_idx if split_idx != -1 else partition_mask.shape[1]-1
        assert -1 <= self.split_idx < partition_mask.shape[1], f"split_idx must be between 0 and the number of clusters - 1. Got {self.split_idx}"
       
        self.partition_mask = partition_mask
        self.a = nn.Parameter(torch.randn(self.n_low),requires_grad=True)
        self.s = nn.Parameter(torch.randn(self.n_low),requires_grad=True)
        
        if upper:
            E_high_mask = torch.triu(torch.ones((self.n_high,self.n_high)),diagonal=1)
        else:
            E_high_mask = torch.tril(torch.ones((self.n_high,self.n_high)),diagonal=-1)
        
    
    def get_new_partition(self):
        """ Inserts a new column to the right of the split_idx column in partition_mask.
            The split_idx column is populated by the learnable parameter s and masked by partition_mask[split_idx] to ensure consistency with current partition.
            The new column is populated by 1 - s and masked by partition_mask[split_idx] to ensure consistency with current partition.
        """
        #we assume T is binary!
        mask = self.partition_mask[self.split_idx]
        a = torch.sigmoid(self.a) * mask
        s = torch.sigmoid(self.s) * mask
        t = torch.sigmoid(self.t) * mask
        one_minus_s = (1 - s) * mask
        one_minus_t = (1 - t) * mask
        
        prefix = self.partition_mask[:,:self.split_idx]
        mid = torch.stack([s, one_minus_s], dim=1)
        suffix = self.partition_mask[:,self.split_idx+1:]

        to_cat = []
        if prefix.shape[1] > 0:
            to_cat.append(prefix)

        to_cat.append(mid)
        
        if suffix.shape[1] > 0:
            to_cat.append(suffix)

        new_partition = torch.cat(to_cat, dim=1)

        return new_partition, s, a

    def get_S(self,partition_mask,E_high,T,impose_sufficiency=True):
        s_mask = partition_mask[:, :]
        if not impose_sufficiency: 
            is_marginalised = torch.all(partition_mask == 0, dim=1).float()
            s_mask = s_mask + is_marginalised * torch.ones_like(T) #this allows marginalised low-level variables to influence all high-level variables.
        
        S = torch.linalg.solve(E_high.T, (self.E @ T).T).T * s_mask
        return S
    
    
    def get_dag_penalty(self,E_high):
        A_high = torch.eye(self.n_high, device=E_high.device) - torch.linalg.inv(E_high)
        M = A_high * A_high
        return torch.trace(torch.matrix_exp(M)) - self.n_high

    def get_reconstruction_penalty(self,T,A_low,E_high):
        A_high = torch.eye(self.n_high, device=E_high.device) - torch.linalg.inv(E_high)
        column_norms = torch.sum(T**2, dim=0)
        T_norm = T / (column_norms + 1e-12)
        A_recon = T @ A_high @ T_norm.T

        adj_recon_loss = torch.mean((A_recon - A_low)**2)
        return adj_recon_loss

    def get_surjectivity_penalty(self,partition_mask_full,target_norm=1.):
        col_sums = partition_mask_full.sum(dim=0)
        target_sums = torch.ones_like(col_sums) * target_norm

        return torch.sum(F.relu(target_sums - col_sums))
    
    def forward(self,loss_type='gram'):
        new_partition, s, a = self.get_new_partition()
        constraint_loss = torch.mean((s + F.relu(s + a - self.thresh - 1) - F.relu(s + a - self.thresh))**2)
        regularisation_loss = F.binary_cross_entropy(s,a)

        prefix = new_partition[:, :self.split_idx]
        mid = (new_partition[:, self.split_idx : self.split_idx + 2] > self.thresh).float()
        suffix = new_partition[:, self.split_idx + 2:]
        to_cat = []
        if prefix.shape[1] > 0:
            to_cat.append(prefix)
        
        to_cat.append(mid) 

        if suffix.shape[1] > 0:
            to_cat.append(suffix)

        new_partition_mask = torch.cat(to_cat, dim=1)

        partition_mask = (new_partition_mask - new_partition).detach() + new_partition #Straight-Through Estimator
        T = new_partition
        I = torch.eye(self.n_low,device=self.device)
        F_pi = torch .diag_embed(new_partition.T) #torch.diag_embed(partition_mask.T) #(n_high, n_low, n_low)
        inv_term = torch.linalg.inv(I + F_pi @ (self.E - I))
        M = (self.E - I) @ inv_term @ F_pi @ self.E
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
        E_high_ij = dot_prod/(b_norm + 1e-6)

        I_high = torch.eye(self.n_high,device=self.device)
        E_high_mask = torch.ones_like(E_high_ij) - I_high #torch.triu(torch.ones_like(E_high_ij), diagonal=1)

        E_high = (E_high_ij * E_high_mask + I_high - E_high_ij).detach() + E_high_ij
        valid_mask = E_high_mask

        if loss_type == 'gram':
            gram_loss = (a_norm * b_norm) - dot_prod ** 2
            intv_loss = torch.relu(gram_loss * valid_mask).mean()

        S = self.get_S(new_partition,E_high,T)
        obs_loss = torch.mean((S @ E_high - self.E @ T)**2)

        return obs_loss, intv_loss, constraint_loss, regularisation_loss, E_high, partition_mask

@dataclass
class TrainingConfig:
    max_epochs: int = 10000
    num_splits: int = 1
    lr: float = 0.01
    acceptable_loss: float = 1e-4
    constraint_pen: float = 1.
    regularisation_pen: float = 1.
    reconstruction_pen: float = 1.
    surjectivity_pen: float = 1.
    verbose: bool = True
    threshold: float = 1e-8
    
def log(msg,verbose=True):
        if verbose: print(msg)

def reconstruction_loss(T,A_low,A_high=None,E_high=None):
    assert not(A_high is None and E_high is None), "at least one of A_high and E_high must be specified"
    if A_high is None:
        A_high = torch.eye(E_high.shape[0],device=E_high.device) - torch.linalg.inv(E_high)

    column_norms = torch.sum(T, dim=0)
    T_norm = T / (column_norms + 1e-12)
    A_recon = T @ A_high @ T_norm.T

    adj_recon_loss = torch.mean((A_recon - A_low)**2)
    return adj_recon_loss

def divisive_clustering(E,train_config):
    T = torch.ones((E.shape[0],1), device=E.device)
    indices_to_split = [0]
    A_low = torch.eye(E.shape[0], device=E.device) - torch.linalg.inv(E)
    T_dict = {1:T.clone()}
    E_high_dict = {1:torch.eye(1, device=E.device)}

    for _ in range(train_config.num_splits):
        best_reconstruction_loss_global = float('inf')
        best_partition_global = None
        best_E_high_global = None
        for split_idx in indices_to_split:
            if T[:,split_idx].sum() <= 1: 
                #skip if this cluster has 1 or fewer variables
                continue
            model = GradSplit(E,T,split_idx=split_idx)
            #import pdb;pdb.set_trace()
            optimiser = torch.optim.Adam(model.parameters(), lr=train_config.lr)
            history = {'obs_loss': [], 'intv_loss':[], 'constraint_loss': [], 'regularisation_loss': [], 'surjectivity_loss' : []}
            best_loss = float('inf')
            best_E_high = None
            best_partition = None
        
            for epoch in range(train_config.max_epochs):
                optimiser.zero_grad()
                obs_loss, intv_loss,constraint_loss, regularisation_loss, E_high, partition_mask = model(loss_type='gram')
                surjectivity_loss = model.get_surjectivity_penalty(partition_mask_full=partition_mask)

                history['obs_loss'].append(obs_loss.item())
                history['intv_loss'].append(intv_loss.item())
                history['constraint_loss'].append(constraint_loss.item())
                history['regularisation_loss'].append(regularisation_loss.item())
                history['surjectivity_loss'].append(surjectivity_loss.item())

                total_loss = (obs_loss + intv_loss 
                + train_config.constraint_pen * constraint_loss 
                + train_config.regularisation_pen * regularisation_loss 
                + train_config.surjectivity_pen * surjectivity_loss)

                total_loss.backward()
                optimiser.step()

                abstraction_loss = obs_loss + intv_loss

                if surjectivity_loss.item() == 0:
                    if abstraction_loss.item() < best_loss:
                        best_loss = abstraction_loss.item()
                        best_E_high = E_high.detach().cpu().clone()
                        best_partition = partition_mask.cpu().clone()
                
                if epoch % 100 == 0:
                    log(f"Epoch {epoch}, Observation Loss: {obs_loss.item()}, Intervention Loss: {intv_loss.item()}, Constraint Loss: {constraint_loss.item()}, Regularisation Loss: {regularisation_loss.item()}, Surjectivity Loss: {surjectivity_loss.item()}",train_config.verbose)
            
            if best_partition is None:
                log(f"No valid partition found for split index {split_idx}.", train_config.verbose)
                #indices_to_split.remove(split_idx)
            else:
                with torch.no_grad():
                    best_reconstruction_loss = reconstruction_loss(T=best_partition,A_low=A_low,E_high=best_E_high).item()
                if best_reconstruction_loss < best_reconstruction_loss_global:
                    best_reconstruction_loss_global = best_reconstruction_loss
                    best_partition_global = best_partition
                    best_E_high_global = best_E_high
        
        if best_partition_global is not None:
            T = best_partition_global
            T_dict[len(T_dict)+1] = T.clone()
            E_high_dict[len(E_high_dict)+1] = best_E_high_global.clone()

            indices_to_split.append(len(indices_to_split))

    return T_dict, E_high_dict, history

if __name__ == "__main__":
    seed = 44
    n_high = 2  #number of high-level variables
    n_low = 6 #number of low-level variables
    density = 0.5

    A_low, T_gt, S_gt, A_high_gt = sample_low_level_graph(n_high,n_low,density,seed=seed)
    plot_matrix(
        matrix_list=[A_high_gt,T_gt,S_gt],
        title_list=['High-Level Adjacency','Endogenous Abstraction T','Exogenous Abstraction S'],
        fig_title='Ground Truth Matrices'
    )
    visualize_causal_graph(A_low, A_high_gt, T_gt, S_gt, title='Ground Truth Causal Abstraction')

    E = torch.linalg.inv(torch.eye(n_low) - torch.from_numpy(A_low).float())
    train_config = TrainingConfig(max_epochs=1000, num_splits=1, lr=0.01, acceptable_loss=1e-4,constraint_pen=1., regularisation_pen=1., surjectivity_pen=1., verbose=True)
    start_time = time.time()
    T_dict, E_high_dict, history = divisive_clustering(E,train_config)
    print(f"Completed in {time.time()-start_time:.4f}s")
    print(f"Learned Abstraction T:\n{T_dict[2].cpu().detach().numpy()}")
    print(f"Learned E_high:\n{E_high_dict[2].cpu().detach().numpy()}")

    T_pred = T_dict[n_high].cpu().detach()
    E_high = E_high_dict[n_high].cpu().detach()
    S_pred = torch.linalg.solve(E_high.T, (E.cpu() @ T_pred).T).T * T_pred
    A_high = torch.eye(E_high.shape[0],device=E_high.device) - torch.linalg.inv(E_high)

    visualize_causal_graph(A_low, A_high.numpy(), T_pred.numpy(), S_pred.numpy(), title='Learned Causal Abstraction')

    #Plot a graph of loss vs epochs
    fig, ax = plt.subplots(1,2, figsize=(10,8))

    ax[0].plot(range(0,len(history['obs_loss']),10), history['obs_loss'][::10])
    ax[0].plot(range(0,len(history['intv_loss']),10), history['intv_loss'][::10])
    ax[0].legend(['Observed Loss','Interventional Loss'])
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")

    ax[1].plot(range(0,len(history['regularisation_loss']),10), history['regularisation_loss'][::10])
    ax[1].plot(range(0,len(history['constraint_loss']),10), history['constraint_loss'][::10])
    ax[1].legend(['Regularisation Loss','Constraint Loss'])
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    plt.show()
