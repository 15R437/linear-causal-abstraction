import numpy as np

def clusters_to_matrix(clusters,n_low):
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
    Numpy based!
    This function takes a high-level adjacency matrix A_high,
    an abstraction matrix T and reconstructs a low-level
    adjacency matrix by distributing the high-level edges according to normalised T.
    """

    with np.errstate(divide='ignore', invalid='ignore'):
        T_norm = T / np.sum(T**2, axis=0)
    
    T_norm = np.nan_to_num(T_norm)

    return T @ A_high @ T_norm.T

def enforce_dag(A):
    """
    Greedily removes edges with the smallest absolute weights that participate in cycles.
    This guarantees an acyclic graph while roughly minimizing reconstruction error.
    """
    n = A.shape[0]
    A_dag = np.zeros_like(A)
    
    # Extract all non-zero edges and sort by magnitude descending
    edges = []
    for i in range(n):
        for j in range(n):
            if A[i, j] != 0:
                edges.append((abs(A[i, j]), A[i, j], i, j))
    
    edges.sort(key=lambda x: x[0], reverse=True)
    
    # DFS to check if adding an edge u->v creates a cycle (i.e. path v->u already exists)
    def has_path(adj, u, v):
        stack = [u]
        visited = set()
        while stack:
            curr = stack.pop()
            if curr == v:
                return True
            if curr not in visited:
                visited.add(curr)
                for nxt in range(n):
                    if adj[curr, nxt] != 0:
                        stack.append(nxt)
        return False

    for _, w, i, j in edges:
        if not has_path(A_dag, j, i):
            A_dag[i, j] = w
            
    return A_dag