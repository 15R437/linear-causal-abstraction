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

        # Precompute pairwise row/column distance matrix — constant across all splits
        W_out = torch.mean((A_low[None, :, :] - A_low[:, None, :]) ** 2, dim=-1)
        W_in  = torch.mean((A_low.T[None, :, :] - A_low.T[:, None, :]) ** 2, dim=-1)
        self.register_buffer('W_recon', W_out + W_in)

    def split(self, cluster_idx):
        cluster_to_split = self.clusters[cluster_idx]
        n = len(cluster_to_split)
        cluster_mask = torch.zeros(self.n_low, 2)
        cluster_mask[cluster_to_split] = 1

        # Balanced random initialisation: half the cluster members prefer left, half prefer right
        perm = torch.randperm(n)
        logits = torch.zeros(self.n_low, 2)
        left_nodes  = [cluster_to_split[i] for i in perm[:n // 2]]
        right_nodes = [cluster_to_split[i] for i in perm[n // 2:]]
        logits[left_nodes,  0] = 1.0
        logits[right_nodes, 1] = 1.0

        self.logits = nn.Parameter(logits)
        self.cluster_mask = cluster_mask
        self.cluster_idx = cluster_idx
    
    def get_partition(self):
        T_soft = F.softmax(self.logits, dim=1) * self.cluster_mask
        indices = torch.argmax(self.logits.detach(), dim=1)
        T_hard = F.one_hot(indices, 2) * self.cluster_mask

        T_soft = insert_columns(M=self.T,N=T_soft,idx=self.cluster_idx)
        T_hard = insert_columns(M=self.T,N=T_hard,idx=self.cluster_idx)

        return T_hard, T_soft
    
    def train_bfs(self, target_k, verbose=True):
        gen = 0
        splittable = {0}  # set of cluster indices (in current self.clusters) to attempt splitting

        while len(self.clusters) < target_k and splittable:
            new_split = False
            log(f"Generation {gen}. Cluster sizes: {[len(c) for c in self.clusters]}", verbose=verbose)

            n_clusters_start = len(self.clusters)
            next_splittable = set()
            out_cursor = 0  # index of each cluster in next-gen self.clusters
            offset = 0      # cumulative insertions so far this generation

            for orig_idx in range(n_clusters_start):
                actual_idx = orig_idx + offset  # true index after previous insertions this generation

                if orig_idx in splittable and len(self.clusters) < target_k:
                    self.split(actual_idx)
                    num_epochs = 1000 * int(np.log(self.n_low) + 1)
                    T, history = self.optimise_T(num_epochs=num_epochs, verbose=False, keep_history=True)
                    left_nz = torch.nonzero(T[:, actual_idx])
                    right_nz = torch.nonzero(T[:, actual_idx + 1])

                    if left_nz.shape[0] > 0 and right_nz.shape[0] > 0:
                        left_cluster = left_nz.squeeze(1).tolist()
                        right_cluster = right_nz.squeeze(1).tolist()
                        self.T = T.clone()
                        self.clusters[actual_idx] = left_cluster
                        self.clusters.insert(actual_idx + 1, right_cluster)
                        next_splittable.update({out_cursor, out_cursor + 1})
                        out_cursor += 2
                        offset += 1
                        new_split = True
                        log(f"  Split cluster {actual_idx}. Recon: {history['recon_loss'][-1]:.4f}, Graph: {history['graph_loss'][-1]:.4f}, Topo: {history['topo_loss'][-1]:.4f}", verbose=verbose)
                    else:
                        out_cursor += 1
                        log(f"  Cluster {actual_idx} cannot be split further.", verbose=verbose)
                else:
                    out_cursor += 1  # fixed cluster passes through unchanged

            splittable = next_splittable
            gen += 1
            if not new_split:
                break
        log(f"Final Cluster sizes: {[len(c) for c in self.clusters]}", verbose=verbose)
        # Pad with empty columns if we couldn't reach target_k
        shortfall = target_k - self.T.shape[1]
        if shortfall > 0:
            padding = torch.zeros(self.n_low, shortfall, device=self.T.device)
            self.T = torch.cat([self.T, padding], dim=1)
    
    
    def train_dfs(self, target_k, min_cluster_size=1, verbose=True):
        split_idx = 0
        min_cluster_size = max(1,int(self.n_low/(2* target_k))) if min_cluster_size is None else min_cluster_size
        while len(self.clusters) < target_k:
            if split_idx >= len(self.clusters):
                break

            self.split(split_idx)
            num_epochs = 1000 * int(math.log10(self.n_low) + 1)
            T, history = self.optimise_T(num_epochs=num_epochs, verbose=False, keep_history=True)
            left_nz = torch.nonzero(T[:, split_idx])
            right_nz = torch.nonzero(T[:, split_idx + 1])

            if left_nz.shape[0] >= min_cluster_size and right_nz.shape[0] >= min_cluster_size:
                self.T = T.clone()
                self.clusters[split_idx] = left_nz.squeeze(1).tolist()
                self.clusters.insert(split_idx + 1, right_nz.squeeze(1).tolist())
                log(f"  Split cluster {split_idx} → {len(self.clusters[split_idx])}/{len(self.clusters[split_idx+1])} nodes. Recon: {history['recon_loss'][-1]:.4f}, Graph: {history['graph_loss'][-1]:.4f}, Topo: {history['topo_loss'][-1]:.4f}", verbose=verbose)
                # split_idx stays: next iteration drills into the left child
            else:
                log(f"  Cluster {split_idx} cannot be split (min size {min_cluster_size}). Moving right.", verbose=verbose)
                split_idx += 1

        log(f"Final cluster sizes: {[len(c) for c in self.clusters]}", verbose=verbose)
        shortfall = target_k - self.T.shape[1]
        if shortfall > 0:
            padding = torch.zeros(self.n_low, shortfall, device=self.T.device)
            self.T = torch.cat([self.T, padding], dim=1)

    def cluster_scores(self):
        """Average within-cluster pairwise W_recon dissimilarity — higher = more heterogeneous = better to split."""
        scores = []
        for c in self.clusters:
            n = len(c)
            if n <= 1:
                scores.append(0.0)
                continue
            idx = torch.tensor(c, dtype=torch.long, device=self.device)
            sub = self.W_recon[idx][:, idx]
            diag_mask = 1 - torch.eye(n, device=self.device)
            scores.append(((sub * diag_mask).sum() / (n * (n - 1))).item())
        return scores
    
    def calc_cluster_graph(self):
        scores = []
        C = (self.A_low_binary @ self.T).clamp(max=1.0)
        C = C * (1 - self.T)  # exclude within-cluster edges: only compare cross-cluster connectivity profiles
        CC = C @ C.T
        CC_norm = (C ** 2).sum(dim=1)
        C_diff_sq = CC_norm[:, None] + CC_norm[None, :] - 2 * CC

        for c in self.clusters:
            n = len(c)
            if n <= 1:
                scores.append(0.0)
                continue
            idx = torch.tensor(c, dtype=torch.long, device=self.device)
            sub = C_diff_sq[idx][:, idx]
            diag_mask = 1 - torch.eye(n, device=self.device)
            scores.append(((sub * diag_mask).sum() / 2).item())

        return scores

            
    def train_best_first(self, target_k, min_cluster_size=1, verbose=True):
        min_cluster_size = max(1, int(self.n_low / (2 * target_k))) if min_cluster_size is None else min_cluster_size
        failed = set()  # frozensets of node indices for clusters we've given up on

        while len(self.clusters) < target_k:
            scores = self.calc_cluster_graph()

            best_idx, best_score = -1, -1.0
            for i, (c, s) in enumerate(zip(self.clusters, scores)):
                if frozenset(c) in failed or len(c) < 2 * min_cluster_size:
                    continue
                if s > best_score:
                    best_score, best_idx = s, i

            if best_idx == -1:
                log("No splittable clusters remaining.", verbose=verbose)
                break

            log(f"  Splitting cluster {best_idx} (size={len(self.clusters[best_idx])}, score={best_score:.4f})", verbose=verbose)
            self.split(best_idx)
            num_epochs = 1000 * int(math.log10(self.n_low) + 1)
            T, history = self.optimise_T(num_epochs=num_epochs, verbose=False, keep_history=True)
            left_nz = torch.nonzero(T[:, best_idx])
            right_nz = torch.nonzero(T[:, best_idx + 1])

            if left_nz.shape[0] >= min_cluster_size and right_nz.shape[0] >= min_cluster_size:
                self.T = T.clone()
                self.clusters[best_idx] = left_nz.squeeze(1).tolist()
                self.clusters.insert(best_idx + 1, right_nz.squeeze(1).tolist())
                log(f"    → {len(self.clusters[best_idx])}/{len(self.clusters[best_idx+1])} nodes. Recon: {history['recon_loss'][-1]:.4f}, Graph: {history['graph_loss'][-1]:.4f}, Topo: {history['topo_loss'][-1]:.4f}", verbose=verbose)
            else:
                failed.add(frozenset(self.clusters[best_idx]))
                log(f"    → Rejected (would produce cluster < {min_cluster_size} nodes).", verbose=verbose)

        log(f"Final cluster sizes: {[len(c) for c in self.clusters]}", verbose=verbose)
        shortfall = target_k - self.T.shape[1]
        if shortfall > 0:
            padding = torch.zeros(self.n_low, shortfall, device=self.T.device)
            self.T = torch.cat([self.T, padding], dim=1)

    def reconstruction_loss(self, T):
        co_assign = T @ T.T                                   # (n_low, n_low)
        diag_mask = 1 - torch.eye(self.n_low, device=T.device)
        mask = co_assign * diag_mask
        recon_loss = (mask * self.W_recon).mean()

        return recon_loss
    
    def graph_loss(self, A_low_binary, T):
        H_soft = torch.max(A_low_binary[:, None, :] * T.T[None, :, :], dim=-1)[0] * (1 - T)
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
        dag_mask = torch.triu(torch.ones(k, k), diagonal=1)

        if compute_topo:
            M = T_soft.T @ self.A_low_binary @ T_soft # M[i,j] is the number of edges between cluster i and cluster j
            topo_loss = torch.sum((1 - dag_mask - torch.eye(k)) * M ** 2)
        else:
            topo_loss = torch.tensor(0.0, device=self.device)

        recon_loss = self.graph_loss(self.A_low_binary.T, T_soft)
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
    

    def optimise_T(self, num_epochs, lambda_graph=100, lambda_topo=10., verbose=True, keep_history=False, compute_graph=True, compute_topo=False):
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
        return T_hard, history

if __name__ == "__main__":
    import time
    from src.utils.plots import plot_matrix, visualize_causal_graph, plot_probability_path, plot_assignment_raster, animate_matrices
    from src.utils.generate_data import sample_low_level_graph, generate_easy_benchmark, sample_by_noise_adjustment_any_order
    from src.utils.losses import intervention_loss, observation_loss, abstraction_residual, graph_loss, kl_loss, reconstruction_loss, reconstruction_loss_with_data, reconstruction_loss_binary
    from src.utils.helper import process_output, get_S, get_A_high, is_dag, reconstruct_adj, enforce_dag
    from src.models.wrapper import hail_mary_graph, hail_mary_intv, hail_mary_graph_intv, hm_iter_bfs, hm_iter_dfs, hm_iter_best_first, hc_fast, random, uca

    seed = None #44
    n_low = 500 #number of low-level variables
    n_high = 50 #number of high-level variables
    k_target = n_high
    density = 0.4
    internal_density = 0.
    blockiness = 0.5
    heteregoneity = 1.
    A_low, T_gt, S_gt, A_high_gt = sample_low_level_graph(n_high,n_low,density,seed=seed,shuffle=True,blockiness=blockiness, heterogeneity=heteregoneity, internal_density=internal_density, graph_type='one_child')
    T_gt_binary = (T_gt != 0) * 1
    S_gt, T_gt, A_high_gt = process_output(T_hard=T_gt_binary, A_low=A_low)

    X_test = sample_by_noise_adjustment_any_order(A_low,n_samples=10000,gennorm_param=1.8,seed=seed)
    E_low = np.linalg.inv(np.eye(n_low) - A_low)
    

    METHODS = {'ground_truth':lambda **kwds: T_gt_binary, 'uca':uca, 'hm_graph':hail_mary_graph, 'hc_fast':hc_fast,'random':random}

    A_recon_list = [A_low]
    T_pred_list = []

    for method_name, method_fn in METHODS.items():
        start_time = time.time()
        T_hard = method_fn(A_low=A_low, k=k_target)
        runtime = time.time() - start_time
        p,k = T_hard.shape
        if method_name == "uca":
            T_pred = T_hard[:,:]
            T_hard = (np.abs(T_pred) > 1e-6).astype(int)
            S_pred = get_S(A_low=A_low, T=T_pred) 
            A_high_pred = enforce_dag(get_A_high(A_low=A_low, T=T_pred))
            
        else:
            S_pred, T_pred, A_high_pred = process_output(T_hard=T_hard, A_low=A_low)

        E_high_pred = np.linalg.inv(np.eye(k) - A_high_pred)
        A_recon = reconstruct_adj(A_high=A_high_pred, T=T_pred)

        if method_name != "ground_truth":
            A_recon_list.append(A_recon)

        T_pred_list.append(T_pred)

        recon_loss = reconstruction_loss(A_low=A_low, T=T_pred, A_high=A_high_pred, A_hat=A_recon)
        recon_loss_data = reconstruction_loss_with_data(A_low=A_low, T=T_pred, A_high=A_high_pred, X=X_test)
        graph_loss_ = graph_loss(A_low=A_low, T_hard=T_hard)
        obs_loss = observation_loss(E_low=E_low, E_high=E_high_pred, S=S_pred, T=T_pred)
        intv_loss = intervention_loss(E_low=E_low, E_high=E_high_pred, S=S_pred, T=T_pred)
        abs_residual = abstraction_residual(A_low=A_low,T=T_pred, A_high=A_high_pred)
        #kl_loss_ = kl_loss(E_high=E_high_pred, A_low=A_low, T=T_pred)
        T_active = (np.abs(T_pred) > 1e-6).astype(int)
        print(f"{method_name} Time: {runtime:.4f}s, Recon Data: {recon_loss_data:.4f}, Graph: {graph_loss_:.4f}, Observation: {obs_loss}, Intervention: {intv_loss}, Clusters Used: {np.sum((T_active.sum(axis=0) > 0))}/{k}, Marginalisation: {np.sum((T_active.sum(axis=1) == 0))}/{p}, DAG: {is_dag(A_high_pred)}")

        #visualize_causal_graph(A_low, A_high_pred, T_pred, S_pred, title=f"{method_name} abstracton")
    
    plot_matrix(
        matrix_list= T_pred_list,
        title_list= list(METHODS.keys()),
        fig_title='Abstraction Matrix'
    )

    plot_matrix(
        matrix_list= A_recon_list,
        title_list= list(METHODS.keys()),
        fig_title='Adjacency Matrix'
    )
