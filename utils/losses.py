import numpy as np
import torch
import sys
from pathlib import Path

# 1. Get the directory of uca.py (/Users/issy/Projects/linear-causal-abstraction/algorithms)
current_script_dir = Path(__file__).resolve().parent
# 2. Go up TWO levels to reach /Users/issy/Projects/
projects_root = current_script_dir.parent.parent
# 3. Point to the folder that CONTAINS the 'unsupervised_causal_reduction' package
target_project_dir = projects_root / 'targeted-causal-reduction-unsupervisedAbstraction'

sys.path.append(str(target_project_dir))
from unsupervised_causal_reduction.causal_model.high_level import HighLevelCausalModel #type: ignore
from unsupervised_causal_reduction.data_generators.linear import LinearCausalModel #type: ignore
from unsupervised_causal_reduction.abstraction.causal_abstraction import CausalAbstraction #type: ignore
from unsupervised_causal_reduction.lightning.targeted_causal_reduction import UnsupervisedCausalAbstractionModel #type: ignore
"""
loss_fn should take as input:
    A_high, A_low, A_hat, T, X
"""
def reconstruction_loss(A_low,A_hat,**kwds):
    baseline = np.linalg.norm(A_low, ord='fro')
    return np.linalg.norm((A_low - A_hat), ord='fro') / baseline

def reconstruction_loss_with_data(A_hat,X, A_low,**kwds):
    """
    Params
        A_hat (np.ndarray): (n_low x n_low) adjacency matrix assumed to be upper-triangular under topological ordering
        X (np.ndarray): (n_samples x n_low) data matrix
        A_low (np.ndarray): (n_low x n_low) true adjacency matrix
    """
    
    I = np.eye(A_hat.shape[0])
    residual = (I-A_hat.T)@X.T #pxN matrix
    residual_oracle = (I-A_low.T)@X.T
    rmse = np.sqrt(np.mean(residual**2))
    rmse_oracle = np.sqrt(np.mean(residual_oracle**2))
    return rmse -  rmse_oracle

def kl_loss(A_high,A_low,T,**kwds):
    """
    Parameters
        A_high (numpy.ndarray): Adjacency matrix for high-level model. X_tilde (k x 1) = E_high @ U_tilde. 
        A_low (numpy.ndarray): Adjacency matrix for low-level model. X (p x 1) = E_low @ U.
        T (numpy.ndrray): Linear abstraction matrix. X_tilde (k x 1) = T @ X
    Returns
        results (dict): 
    """
    if T is None or A_high is None:
        return np.nan
    T = T.T
    k = A_high.shape[0]
    p = A_low.shape[0]
    E_high = np.linalg.solve((np.eye(k)-A_high),np.eye(k))
    Model_H = HighLevelCausalModel(n_vars=k,E_macro=E_high)
    Model_L = LinearCausalModel(n_vars=p,low_level_scm=A_low)
    CA = CausalAbstraction(low_level=Model_L,high_level=Model_H)
    CA.set_weights(torch.Tensor(T))

    uca_model = UnsupervisedCausalAbstractionModel(abstraction=CA)
    _,results = uca_model._training_step()

    return results['consistency_loss'].item()

def observation_loss(E_low,E_high,S,T,**kwds):
    I_low = np.eye(E_low.shape[0])
    I_high = np.eye(E_high.shape[0])

    
    lhs = S @ E_high #(I_low - A_low) @ S
    rhs = E_low @ T #T @ (I_high - A_high)
    obs_loss = np.mean((lhs - rhs)**2)
    return obs_loss

def intervention_loss(E_low,S,T,**kwds):
    n_low,n_high = T.shape
    I_low = np.eye(n_low)
    
    F_pi = T.T[...,None] * I_low
    term = I_low + F_pi @ (E_low - I_low)

    lhs = F_pi @ S[None,...]
    rhs = term @ F_pi @ T[None,...]
    intv_loss = np.mean(np.mean((lhs - rhs)**2,axis=(-2,-1)))

    return intv_loss

def intervention_loss_direct(E_low, E_high, S, T, **kwds):
    n_low, n_high = T.shape
    I_low = np.eye(n_low)
    I_high = np.eye(n_high)

    F_ii = np.zeros((n_high, n_high, n_high))
    idx = np.arange(n_high)
    F_ii[idx, idx, idx] = 1.0
    F_pi = ((T.T != 0).astype(float))[..., None] * I_low


    E_high_do = E_high[None, ...] - (E_high - I_high)[None, ...] @ F_ii @ E_high[None, ...]

    inv_term_low = np.linalg.inv(I_low + F_pi @ (E_low - I_low))
    E_low_do = E_low - (E_low - I_low) @ inv_term_low @ F_pi @ E_low

    loss = 0.
    for i in range(n_high):
        LHS = E_low_do[i] @ T          # (n_low, n_high)
        RHS = (S + (T - S) @ F_ii[i]) @ E_high_do[i]  # (n_low, n_high)
        loss += np.mean((LHS - RHS) ** 2)

    return loss / n_high

def abstraction_residual(A_low, T, A_high, **kwds):
        n_low,n_high = T.shape
        I = np.eye(n_low)

        S_list = []
        for j in range(n_high):
            t_j = T[:, j]
            A_jj = t_j[:, None] * A_low * t_j[None, :]
            s_j = np.linalg.solve(I - A_jj, t_j)
            S_list.append(s_j)
        S = np.stack(S_list, axis=1)  # (n_low, n_high)

        influence_matrix = A_low @ S # (n_low, n_high)

        T_sq = T ** 2                                          # (n_low, n_high)
        t_i_norms = np.clip(T_sq.sum(axis=0), a_min=1e-8, a_max=None)                # (n_high,) effective block sizes

        # Weighted mean: a_star[i,j] = sum_r w[r,i] * influence[r,j] / sum_r w[r,i]
        #a_star = (T_sq.T @ influence_matrix) / t_i_norms[:,None]  # (n_high, n_high)
        diff = influence_matrix[None, :, :] - A_high[:, None, :]* T.T[:, :, None]

        weights = T_sq.T[:, :, None]                               # (n_high, n_low, 1)
        residuals = (weights * diff ** 2).sum(axis=1) / t_i_norms[:,None]  # (n_high, n_high)
        diag_mask = 1.0 - np.eye(n_high)
        return (residuals * diag_mask).sum()

def graph_loss(A_low, T, **kwds):
    n_low = A_low.shape[0]
    A_low_binary = (A_low != 0).astype(float)
    
    # H[i,j] = number of edges between low-level node i and high-level cluster j
    H = A_low_binary @ T 
    
    # 2. Replicate the forward pass of: 1 - relu(1 - H)
    # This clamps the upper bound of H to 1.0 without affecting values below it
    C = 1.0 - np.maximum(0, 1.0 - H) 
    
    # 3. Compute number of shared cluster connections between low-level nodes
    CC = C @ C.T 
    
    # 4. Pairwise squared Euclidean distances: ||C[i] - C[j]||^2 using broadcasting
    CC_norm = (C ** 2).sum(axis=1)
    C_diff_sq = CC_norm[:, None] + CC_norm[None, :] - 2 * CC  
    
    # 5. Continuous co-assignment matrix
    co_assign = T @ T.T                                   
    diag_mask = 1.0 - np.eye(n_low)
    
    # 6. Calculate total masked mean loss
    graph_loss = (co_assign * diag_mask * C_diff_sq).sum()
    
    return graph_loss


def topo_loss(A_low, T, A_high, **kwargs):
    n_high = T.shape[1]
    
    # 1. Binarize low-level graph connections
    A_low_binary = (A_low != 0).astype(float)
    dag_mask = (A_high != 0).astype(float)
    
    # M[i,j] represents the aggregate structural connection from cluster i to cluster j
    M = T.T @ A_low_binary @ T 
        
    violation_mask = 1.0 - dag_mask - np.eye(n_high)
    
    topo_loss = np.mean(violation_mask * (M ** 2))
    
    return topo_loss

def abstraction_loss(E_low,A_low,A_high,S,T,**kwds):
    obs_loss = observation_loss(A_low=A_low, A_high=A_high, S=S, T=T)
    intv_loss = intervention_loss(E_low=E_low, S=S, T=T)
    return obs_loss + intv_loss

def heuristic_cost(A_low, c1_nodes, c2_nodes, **kwds):
    out_diff = np.mean(A_low[c1_nodes, :], axis=0) - np.mean(A_low[c2_nodes, :], axis=0)
    in_diff  = np.mean(A_low[:, c1_nodes], axis=1) - np.mean(A_low[:, c2_nodes], axis=1)
    
    return np.sum(out_diff**2) + np.sum(in_diff**2)

def frob_norm(pred,gt):
    if pred is None:
        return np.nan
    return np.linalg.norm(pred-gt,ord='fro')/np.linalg.norm(gt,ord='fro')

def rand_index(T_gt,T_pred,**kwds):
    p,k = T_gt.shape
    a=b=c=d=0
    for i in range(p-1):
        for j in range(i+1,p):
            SAME_CLST_gt = set(np.where(T_gt[i,:]!=0)[0]).intersection(set(np.where(T_gt[j,:]!= 0)[0]))
            SAME_CLST_pred = set(np.where(T_pred[i,:]!=0)[0]).intersection(set(np.where(T_pred[j,:]!= 0)[0]))
            if SAME_CLST_gt and SAME_CLST_pred:
                a += 1
            elif not SAME_CLST_gt and not SAME_CLST_pred:
                b += 1

    rand_index = 2*(a+b)/(p*(p-1)) 
    return rand_index

def abstraction_distance(E_low,T_pred,E_high,**kwds):
    """
    Maximises the distance (over u and all valid interventions) between T_pred.T @ E_low @ u (under intervention on the low-level model) and E_high @ S_pred.T @ u (under the corresponding intervention in the high-level model)
    """
    n_low, n_high = T_pred.shape
    S_pred = E_low @ T_pred @ np.linalg.inv(E_high)
    alignment = {j: np.where(T_pred[:,j]!=0)[0] for j in range(n_high)}
    max_dist = 0
    dist = [None for _ in range(n_high)]
    sim_interv = torch.random.rand((n_low,n_high))
    sim_interv.requires_grad = True
    T_mask = (T_pred != 0) * 1.0
    for j in range(n_high):
        #go through each possible intervention on the high-level model (intervening on variable j)
        y = np.random.rand(len(alignment[j])) #learnable parameter
        u_init = np.random.rand(n_low)
        u_init[alignment[j]] = y

        E_do = E_low - (E_low-np.eye(n_low)) @ (np.eye(n_low))

        E_high_do = E_high - (E_high-np.eye(n_high)) @ (np.eye(n_high))
    pass

def find_lost_weights(A_low, clusters):
    # 1. Create a mask of "preserved" edges (initially all False)
    n = A_low.shape[0]
    active_nodes = []
    for clst, nodes in clusters.items():
        if clst != -1:
            active_nodes.extend(nodes)
            
    all_edges_mask = (A_low != 0)
    
    survived_mask = np.zeros((n, n), dtype=bool)
    for clst, nodes in clusters.items():
        if clst == -1: continue
        for other_clst, other_nodes in clusters.items():
            if other_clst == -1 or clst == other_clst:
                continue
            # These are the edges that A_high actually represents
            survived_mask[np.ix_(nodes, other_nodes)] = True
            
    # Lost edges are those that were non-zero but didn't survive
    lost_mask = all_edges_mask & ~survived_mask
    return A_low[lost_mask]