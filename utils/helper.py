import numpy as np

def clusters_to_matrix(clusters):
    n_low = sum((len(clst) for clst in clusters.values()))
    n_high = len({k:v for k,v in clusters.items() if k != -1})

    T = np.zeros((n_low,n_high))
    for k,v in clusters.items():
        if k != -1:
            T[v,k] = 1
    
    return T

def matrix_to_clusters(T):

    pass

def reconstruct_adj(A_high,T):
    """
    This function takes a high-level adjacency matrix A_high,
    an abstraction matrix T and reconstructs a low-level
    adjacency matrix.
    """

    with np.errstate(divide='ignore', invalid='ignore'):
        T_norm = T / np.sum(T**2, axis=0)
    T_norm = np.nan_to_num(T_norm)

    return T @ A_high @ T_norm.T