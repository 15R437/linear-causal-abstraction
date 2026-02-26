import numpy as np

def reconstruction_rmse(E_pred,X,E_gt=None,eps=1e-6,**kwds):
    p = X.shape[1]
    U_pred = np.linalg.solve(E_pred,X)
    RMSE = np.linalg.norm(U_pred)/np.sqrt(p)
    if E_gt is not None:
        RMSE_gt = np.linalg.norm(np.linalg.solve(E_gt,X.T))/np.sqrt(p)
        return RMSE/(RMSE_gt+eps)
    
    return RMSE

def frob_norm_A(A_pred,A_gt,**kwds):
    return np.linalg.norm(A_pred-A_gt,ord='fro')/np.linalg.norm(A_gt,ord='fro')

def frob_norm_T(T_pred,T_gt,**kwds):
    return np.linalg.norm(T_pred - T_gt, ord='fro') / np.linalg.norm(T_gt, ord='fro')

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
    for j in range(n_high):
        #go through each possible intervention on the high-level model (intervening on variable j)
        y = np.random.rand(len(alignment[j])) #learnable parameter
        u_init = np.random.rand(n_low)
        u_init[alignment[j]] = y

        E_do = E_low - (E_low-np.eye(n_low)) @ (np.eye(n_low))

        E_high_do = E_high - (E_high-np.eye(n_high)) @ (np.eye(n_high))
    pass

def find_lost_weights(A_low, clusters):
    lost_weights = []
    
    marg_nodes = clusters.get(-1, [])
    if len(marg_nodes) > 0:
        lost_weights.append(A_low[marg_nodes, :].flatten())
        lost_weights.append(A_low[:, marg_nodes].flatten())

    for clst, nodes in clusters.items():
        if clst == -1 or len(nodes) < 2: 
            continue
        internal_block = A_low[np.ix_(nodes, nodes)]
        lost_weights.append(internal_block.flatten())

    if not lost_weights:
        return np.array([])
        
    combined = np.concatenate(lost_weights)

    return combined[combined != 0]
    