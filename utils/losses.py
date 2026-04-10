import numpy as np
import torch
#from unsupervised_causal_reduction.causal_model.high_level import HighLevelCausalModel #type: ignore
#from unsupervised_causal_reduction.data_generators.linear import LinearCausalModel #type: ignore
#from unsupervised_causal_reduction.abstraction.causal_abstraction import CausalAbstraction #type: ignore
#from unsupervised_causal_reduction.lightning.targeted_causal_reduction import UnsupervisedCausalAbstractionModel #type: ignore

def reconstruction_rmse(A_hat,X):
    """
    Params
        A_hat (np.ndarray): (n_low x n_low) adjacency matrix assumed to be upper-triangular under topological ordering
        X (np.ndarray): (n_samples x n_low) data matrix
    """
    if A_hat is None:
        return np.nan
    else:
        residual = (np.eye(A_hat.shape[0])-A_hat.T)@X.T #pxN matrix
    RMSE = np.sqrt(np.mean(residual**2))
    return RMSE

def frob_norm(pred,gt):
    if pred is None:
        return np.nan
    return np.linalg.norm(pred-gt,ord='fro')/np.linalg.norm(gt,ord='fro')

"""
def kl_loss(A_high,A_low,T):
    
    Parameters
        E_high (numpy.ndarray): Total treatment effect matrix for high-level model. X_tilde (k x 1) = E_high @ U_tilde. 
        E_low (numpy.ndarray): Total treatment effect matrix for low-level model. X (p x 1) = E_low @ U.
        T (numpy.ndrray): Linear abstraction matrix. X_tilde (k x 1) = T @ X
    Returns
        results (dict): 

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
"""

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