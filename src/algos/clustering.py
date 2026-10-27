import numpy as np
import copy
import random
import itertools
import heapq
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))

from src.utils.generate_data import sample_low_level_graph
from src.utils.plots import visualize_causal_graph
from src.utils.losses import reconstruction_loss, abstraction_loss
from src.utils.helper import get_S, get_A_high, reconstruct_adj

def get_T(assignment_array, **kwds):
    n_low = len(assignment_array)
    unique, col_indices = np.unique(assignment_array, return_inverse=True)
    n_high = len(unique)
    
    T = np.zeros((n_low, n_high))
    T[np.arange(n_low), col_indices] = 1.0
    
    return T

def is_consistent(init_assignment, clusters_list, c1, c2, assignment_array=None):
    c1, c2 = min(c1, c2), max(c1, c2)

    # Check consistency with initialization
    assigned1 = init_assignment[clusters_list[c1]]
    assigned2 = init_assignment[clusters_list[c2]]
    assigned = np.concatenate([assigned1[assigned1 != -1], assigned2[assigned2 != -1]])

    return len(np.unique(assigned)) <= 1
    

def heuristic_loss(A_low, clusters_list, c1, c2, assignment_array=None):
    c1, c2 = min(c1, c2), max(c1, c2)
            
    c1_nodes = clusters_list[c1]
    c2_nodes = clusters_list[c2]
    
    out_diff = np.mean(A_low[c1_nodes, :], axis=0) - np.mean(A_low[c2_nodes, :], axis=0)
    in_diff  = np.mean(A_low[:, c1_nodes], axis=1) - np.mean(A_low[:, c2_nodes], axis=1)

    loss = np.sum(out_diff**2) + np.sum(in_diff**2)
    return loss

def abstraction_residual(A_low, clusters_list, c1, c2, assignment_array):
        T = get_T(assignment_array)
        n_high = T.shape[1]
        n_low = T.shape[0]
        I = np.eye(n_low)

        S_list = []
        for j in range(n_high):
            t_j = T[:, j]
            A_jj = t_j[:, None] * A_low * t_j[None, :]
            s_j = np.linalg.solve(I - A_jj, t_j)
            S_list.append(s_j)
        S = np.stack(S_list, axis=1)  # (n_low, n_high)

        influence_matrix = A_low @ (T * S)  # (n_low, n_high)

        T_sq = T ** 2                                          # (n_low, n_high)
        t_i_norms = np.clip(T_sq.sum(axis=0), a_min=1e-8, a_max=None)                # (n_high,) effective block sizes

        # Weighted mean: a_star[i,j] = sum_r w[r,i] * influence[r,j] / sum_r w[r,i]
        a_star = (T_sq.T @ influence_matrix) / t_i_norms[:,None]  # (n_high, n_high)
        diff = influence_matrix[None, :, :] - a_star[:, None, :]

        weights = T_sq.T[:, :, None]                               # (n_high, n_low, 1)
        residuals = (weights * diff ** 2).sum(axis=1) / t_i_norms[:,None]  # (n_high, n_high)
        diag_mask = 1.0 - np.eye(n_high)
        return (residuals * diag_mask).sum()


def hierarchical_clustering(A_low, n_high=1,init_assignment=None,loss_list=[heuristic_loss], compute_all_levels=False, A_low_binary=None):
    if A_low_binary is None:
        A_low_binary = A_low.copy()

    n_low = A_low.shape[0]
    E_low = np.linalg.inv(np.eye(len(A_low)) - A_low)
    clusters_list = [[i] for i in range(n_low)]
    assignment_array = np.arange(n_low)

    if init_assignment is None:
        init_assignment = np.ones(n_low) * -1
    
    active_clusters = set(range(n_low))
    cluster_version = {i:0 for i in range(n_low)}
    heap = []

    def push_merge_candidate(c1):
        for c2 in active_clusters:
            if c2 == c1: continue
            c1, c2 = min(c1, c2), max(c1, c2)
            temp_assignment_array = assignment_array.copy()
            temp_assignment_array[clusters_list[c2]] = c1
            losses = [1 - int(is_consistent(A_low, clusters_list, c1, c2))] + [loss_fn(A_low=A_low_binary,clusters_list=clusters_list,c1=c1,c2=c2,assignment_array=temp_assignment_array) 
                                                 for loss_fn in loss_list]
            heapq.heappush(heap, tuple(losses + [c1, c2, cluster_version[c1], cluster_version[c2]]))
            

    for i in range(n_low):
        push_merge_candidate(i)

    S_dict = {}
    T_dict = {}
    A_high_dict = {}
    k = n_low
    while k >= n_high and heap:
        if compute_all_levels or k == n_high:
            T = get_T(assignment_array)
            A_high = get_A_high(A_low=A_low, T=T)
            S = get_S(E_low=E_low, T=T, A_low=A_low)
            S_dict[k] = S
            T_dict[k] = T
            A_high_dict[k] = A_high

        *losses, i, j, v_i, v_j = heapq.heappop(heap)
        if i not in active_clusters or j not in active_clusters:
            continue
        if v_i != cluster_version[i] or  v_j != cluster_version[j]:
            continue
        
        # commit 
        clusters_list[i].extend(clusters_list[j])
        assignment_array[clusters_list[j]] = i
        active_clusters.remove(j)
        cluster_version[i] += 1
        push_merge_candidate(i)
        k -= 1
    
    return S_dict, T_dict, A_high_dict


    
if __name__ == "__main__":
    seed = 23 #23 sometimes gives an example of zero loss but graphical violations
    n_low = 60
    n_high = 10
    marg_prob = 0.
    T_lower = (-2,1)
    T_upper = (-1,2)
    edge_density = 0.4

    A_low,T_gt,S_gt,A_high_gt = sample_low_level_graph(n_high=n_high,n_low=n_low,density=edge_density,T_lower=T_lower,T_upper=T_upper,marg_prob=marg_prob,seed=seed,shuffle=True,internal=False)
    E_low = np.linalg.inv(np.eye(n_low) - A_low)

    #assignment,A_high = find_valid_alignment(A_low,n_high, init_clusters, seed=seed)
    start_time = time.time()
    S_dict, T_dict, A_high_dict = hierarchical_clustering(A_low,loss_list=[heuristic_loss],dag_learner='learner',compute_all_levels=True)
    runtime = time.time() - start_time
    S_pred = S_dict[n_high]
    T_pred = T_dict[n_high]
    A_high = A_high_dict[n_high]
    A_hat = reconstruct_adj(A_high,T_pred)

    print(f"Runtime: \n{runtime:.4f}s")
    print(f"Reconstruction Loss: \n {reconstruction_loss(A_low,A_hat):4f}")
    print(f"Abstraction Loss: \n {abstraction_loss(E_low,A_low,A_high,S_pred,T_pred):4f}")

    visualize_causal_graph(A_low, A_high, T_pred, S_pred,title='Low-Level DAG with Valid Clusters')
                        


