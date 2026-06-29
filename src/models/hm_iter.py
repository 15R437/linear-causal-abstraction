import math
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))

from src.utils.helper import log

def insert_columns(M, N, idx):
    assert N.shape[0] == M.shape[0], f"N must have same number of rows as M. {N.shape[0]} != {M.shape[0]}"
    assert 0 <= idx < M.shape[1], f"idx should be between 0 and {M.shape[1]}. Got {idx}"

    prefix = [M[:,:idx]] if idx > 0 else []
    middle = [N]
    suffix = [M[:,idx+1:]] if idx+1 < M.shape[1] else []

    M_new = torch.cat(prefix + middle + suffix, dim=1)
    return M_new

class HailMaryIter(nn.Module):
    def __init__(self, A_low):
        super().__init__()
        n_low = A_low.shape[0]
        self.device = A_low.device
        self.n_low = n_low
        
        self.register_buffer('A_low', A_low)  # (n_low, n_low) low-level weight matrix
        self.register_buffer('A_low_binary', (A_low != 0).float())
        self.register_buffer('T', torch.ones(n_low, 1))
        self.clusters = [[i for i in range(n_low)]]

    def split(self,cluster_idx):
        cluster_to_split = self.clusters[cluster_idx]
        cluster_mask = torch.zeros(self.n_low,2)
        cluster_mask[cluster_to_split] = 1
        
        self.logits = nn.Parameter(torch.cat([torch.zeros(self.n_low,1), torch.ones(self.n_low,1)],dim=1))
        self.cluster_mask = cluster_mask
        self.cluster_idx = cluster_idx
    
    def get_partition(self):
        T_soft = F.softmax(self.logits, dim=1) * self.cluster_mask
        indices = torch.argmax(self.logits.detach(), dim=1)
        T_hard = F.one_hot(indices, 2) * self.cluster_mask

        T_soft = insert_columns(M=self.T,N=T_soft,idx=self.cluster_idx)
        T_hard = insert_columns(M=self.T,N=T_hard,idx=self.cluster_idx)

        return T_hard, T_soft
    
    def train(self,target_k):
        gen = 0
        while len(self.clusters) < target_k:
            # breadth first search
            new_split = False
            for _ in range(2**gen):
                # split from left to right
                split_idx = 0
                if split_idx >= len(self.clusters) or len(self.clusters) >= target_k:
                    break
                self.split(split_idx)
                num_epochs = 1000 * int(math.log10(self.n_low)+1)
                T = self.optimise_T(num_epochs=num_epochs,verbose=False)
                left_nz = torch.nonzero(T[:, split_idx])
                right_nz = torch.nonzero(T[:, split_idx + 1])

                if left_nz.shape[0] > 0 and right_nz.shape[0] > 0:
                    left_cluster = left_nz.squeeze(1).tolist()
                    right_cluster = right_nz.squeeze(1).tolist()
                    self.T = T.clone()
                    self.clusters[split_idx] = left_cluster
                    self.clusters.insert(split_idx+1, right_cluster)
                    split_idx += 2
                    new_split = True
                else:
                    #no update to cluster assignments
                    split_idx += 1
            gen += 1
            if not new_split:
                break

        # Pad with empty columns if we couldn't reach target_k
        shortfall = target_k - self.T.shape[1]
        if shortfall > 0:
            padding = torch.zeros(self.n_low, shortfall, device=self.T.device)
            self.T = torch.cat([self.T, padding], dim=1)
    
    
    def reconstruction_loss(self, T):
        A_low = self.A_low
        W_out = torch.mean((A_low[None, :, :] - A_low[:,None,:])**2,dim=-1)
        W_in = torch.mean((A_low.T[None,:,:] - A_low.T[:, None, :])**2,dim=-1)
        co_assign = T @ T.T                                   # (n_low, n_low)
        diag_mask = 1 - torch.eye(A_low.shape[0], device=T.device)
        mask = co_assign * diag_mask
        recon_loss = (mask * (W_out + W_in)).mean()

        return recon_loss
    
    def graph_loss(self, A_low_binary, T):
        H_soft = torch.max(A_low_binary[:, None, :] * T.T[None, :, :], dim=-1)[0]
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
        graph_loss = (mask * H_pen).mean()

        return graph_loss

    def forward(self, compute_graph=True, compute_topo=False):
        T_hard, T_soft = self.get_partition()
        p,k = T_hard.shape
        A_low_binary = (self.A_low != 0).float()
        dag_mask = torch.triu(torch.ones(k, k), diagonal=1)

        if compute_topo:
            M = T_soft.T @ A_low_binary @ T_soft # M[i,j] is the number of edges between cluster i and cluster j
            topo_loss = torch.sum((1 - dag_mask - torch.eye(k)) * M ** 2)
        else:
            topo_loss = torch.tensor(0.0, device=self.device)

        recon_loss = self.reconstruction_loss(T_soft) # recon_loss_tensor(self.A_low_normalised, T_soft, A_high) # self.graph_loss(A_low_binary.T, T_soft)  #inter_block_recon_penalty(A_low_binary,T,A_high) #+ intra_cluster_penalty(A_low_binary,T)  inter_block_recon_penalty(self.A_low,T,A_high) #
        entropy_loss = -torch.sum(T_soft * torch.log(T_soft + 1e-8), dim=1).mean()

        top_k,_ = torch.topk(T_soft,k=2,dim=0)
        col_deficit = F.relu(0.5 - top_k.mean(dim=0))
        surjective_loss = (col_deficit * (-torch.log(T_soft + 1e-8).mean(dim=0))).sum() #torch.sum(T.mean(dim=0) * torch.log(T.mean(dim=0) + 1e-8)) + math.log(self.n_high)

        graph_loss = self.graph_loss(self.A_low_binary,T_soft) if compute_graph else torch.tensor(0.0, device=self.device)
  

        return recon_loss, graph_loss, topo_loss, surjective_loss, entropy_loss

    def optimise_step(self, optimiser, lambda_graph=1., lambda_topo=1., compute_graph=True, compute_topo=True):
        optimiser.zero_grad()
        recon_loss, graph_loss, topo_loss, surjective_loss, entropy_loss = self.forward(compute_graph=compute_graph, compute_topo=compute_topo)
        total_loss = (recon_loss + 
                      (lambda_graph) * graph_loss +
                      (lambda_topo) * topo_loss
                      )
        total_loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        return recon_loss.item(), graph_loss.item(), topo_loss.item(),  surjective_loss.item(), entropy_loss.item()
    

    def optimise_T(self, num_epochs, lambda_graph=10, lambda_topo=100., verbose=True, keep_history=False, compute_graph=True, compute_topo=True):
        optimiser = torch.optim.Adam([{'params':[self.logits],'lr':1e-2}],betas=(0.8,0.99))
        history = {'recon_loss':[], 'graph_loss':[], 'topo_loss':[],  'surjective_loss': [], 'entropy_loss': [], 'T':[]} if keep_history else None
        
        for epoch in range(num_epochs):
            recon_loss, graph_loss, topo_loss, surjective_loss, entropy_loss = self.optimise_step(optimiser, lambda_graph, lambda_topo, compute_graph=compute_graph, compute_topo=compute_topo)
            if epoch % 100 == 0 or epoch == num_epochs - 1:
                log(f"""Epoch {epoch+1}/{num_epochs},
                        Reconstruction: {recon_loss},
                        Graph: {graph_loss},
                        Topo: {topo_loss:.6f},
                        Surjective: {surjective_loss:.6f},
                        Entropy: {entropy_loss} """, verbose=verbose)
            if keep_history:
                history['recon_loss'].append(recon_loss)
                history['graph_loss'].append(graph_loss * lambda_graph)
                history['topo_loss'].append(topo_loss * lambda_topo)
                history['surjective_loss'].append(surjective_loss)
                history['entropy_loss'].append(entropy_loss)
                history['T'].append(self.get_partition()[1].cpu().detach().numpy())

        
        T_hard, _ = self.get_partition()
        return T_hard

if __name__ == "__main__":
    import time
    from src.utils.plots import plot_matrix, visualize_causal_graph, plot_probability_path, plot_assignment_raster, animate_matrices
    from src.utils.generate_data import sample_low_level_graph, generate_easy_benchmark, sample_by_noise_adjustment_any_order
    from src.utils.losses import intervention_loss, observation_loss, abstraction_residual, graph_loss, graph_loss_entropy, reconstruction_loss, reconstruction_loss_with_data, reconstruction_loss_binary
    from src.utils.helper import process_output, get_S, get_A_high, is_dag, reconstruct_adj
    from src.models.wrapper import hail_mary_graph, hm_iter, hc_fast, random

    seed = None #44
    n_low = 100 #number of low-level variables
    n_high = 30 #number of high-level variables
    k = 30
    density = 0.4
    internal_density = 0.
    blockiness = 0.5
    heteregoneity = 1.
    A_low, T_gt, S_gt, A_high_gt = sample_low_level_graph(n_high,n_low,density,seed=seed,shuffle=True,blockiness=blockiness, heterogeneity=heteregoneity, internal_density=internal_density)
    T_gt_binary = (T_gt != 0) * 1
    S_gt, T_gt, A_high_gt = process_output(T_hard=T_gt_binary, A_low=A_low)

    X_test = sample_by_noise_adjustment_any_order(A_low,n_samples=10000,gennorm_param=1.8,seed=seed)
    E_low = np.linalg.inv(np.eye(n_low) - A_low)
    

    METHODS = {'ground_truth':lambda **kwds: (T_gt_binary, None), 'hm_graph':hail_mary_graph, 'hm_iter':hm_iter, 'hc_fast':hc_fast, 'random':random}

    for method_name, method_fn in METHODS.items():
        start_time = time.time()
        T_hard,_ = method_fn(A_low=A_low, k=k)
        runtime = time.time() - start_time
        p,k = T_hard.shape
        S_pred, T_pred, A_high_pred = process_output(T_hard=T_hard, A_low=A_low)
        E_high_pred = np.linalg.inv(np.eye(k) - A_high_pred)

        recon_loss = reconstruction_loss_binary(A_low=A_low, T=T_pred, A_high=A_high_pred)
        recon_loss_data = reconstruction_loss_with_data(A_low=A_low, T=T_pred, A_high=A_high_pred, X=X_test)
        graph_loss_ = graph_loss(A_low=A_low, T_hard=T_hard)
        obs_loss = observation_loss(E_low=E_low, E_high=E_high_pred, S=S_pred, T=T_pred)
        intv_loss = intervention_loss(E_low=E_low, E_high=E_high_pred, S=S_pred, T=T_pred)
        abs_residual = abstraction_residual(A_low=A_low,T=T_pred, A_high=A_high_pred)
        print(f"{method_name} Time: {runtime:.4f}s, Reconstruction Loss: {recon_loss:.4f}, Reconstruction (Data): {recon_loss_data:.4f}, Graph Loss: {graph_loss_:.4f}, Observation Loss: {obs_loss}, Intervention Loss: {intv_loss}, Is Surjective?: {T_hard.sum(axis=0).min() > 0}, Is DAG?: {is_dag(A_high_pred)}")
