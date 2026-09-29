import numpy as np
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp
import math
import torch
import torch.nn as nn
from typing import Literal
from scipy.optimize import minimize
import time
from src.utils.losses import abstraction_residual, observation_loss
from collections import deque

def get_device(cpu=True):
    """Best available torch device: CUDA, then Apple Silicon MPS, then CPU."""
    if cpu:
        return torch.device('cpu')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def clusters_to_matrix(clusters,n_low):
    n_high = len({k:v for k,v in clusters.items() if k != -1})

    T = np.zeros((n_low,n_high))
    for k,v in clusters.items():
        if k != -1:
            T[v,k] = 1
    
    return T

def log(msg,verbose=True):
        if verbose:
            print(msg)


def reconstruct_adj(A_high, T):
    R = np.linalg.pinv(T.T)
    return T @ A_high @ R.T

def enforce_dag(A):
    """
    Greedily removes edges with the smallest absolute weights that participate in cycles.
    This guarantees an acyclic graph while roughly minimizing reconstruction error.
    """
    n = A.shape[0]
    is_torch = hasattr(A, "detach")
    if is_torch:
        device = A.device
        dtype = A.dtype
        A_np = A.detach().cpu().numpy()
    else:
        A_np = np.array(A)
    
        A_dag = np.zeros_like(A_np)
    
    
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

    if is_torch:
        return torch.tensor(A_dag, device=device, dtype=dtype)
    return A_dag


def is_dag(A):
    """
    Returns True if the adjacency matrix A represents a DAG.
    A should be a square, n x n matrix.
    """
    n = A.shape[0]
    # Calculate in-degree of each node
    # Summing columns gives the number of incoming edges for each node
    in_degree = np.sum(A != 0, axis=0)
    
    # Queue for nodes with 0 in-degree
    queue = [i for i in range(n) if in_degree[i] == 0]
    visited_count = 0
    
    while queue:
        u = queue.pop(0)
        visited_count += 1
        
        # Look at neighbors (outgoing edges from u)
        for v in range(n):
            if A[u, v] != 0:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
                    
    # If we visited all nodes, there is no cycle
    return visited_count == n

def get_S(T, A_low):
    partition = (T != 0) * 1.
    n_low = A_low.shape[0]
    A_intra = partition @ partition.T * A_low
    I_low = np.eye(n_low)

    S = np.linalg.solve(I_low - A_intra, T)
    return S

def get_S(T, A_low, T_hard=None):
    if T_hard is None:
        T_hard = (T != 0).astype(int)
    idx = cluster_indices(T_hard)
    S = np.zeros_like(T)
    for j in range(T.shape[1]):
        if len(idx[j]) == 0:
            continue
        W_jj = A_low[np.ix_(idx[j], idx[j])]
        S[idx[j], j] = np.linalg.solve(np.eye(len(idx[j])) - W_jj, T[idx[j], j])
    return S

def get_A_high(T, A_low, priority='abstraction'):
    n_low,n_high = T.shape
    if priority == 'recon':
        with np.errstate(divide='ignore', invalid='ignore'):
            T_norm = T / np.sum(T**2, axis=0)
        
        T_norm = np.nan_to_num(T_norm)
        A_high = T_norm.T @ A_low @ T

    elif priority == 'abstraction':
        S = get_S(T,A_low)
        AS = A_low @ S                     

        A_high = T.T @ AS                                
        t_norms = np.sum(T**2, axis=0)
        t_norms = np.clip(t_norms, 1e-8, None)

        A_high = A_high / t_norms[:, None]
        A_high = A_high * (1 - np.eye(n_high))
    
    return A_high



def regularised_surjective_init(n_low, n_high, noise_std=0.1):
    if n_low < n_high:
        raise ValueError(
            f"Surjectivity impossible: n_low ({n_low}) must be >= n_high ({n_high}) "
            f"to allow each column to have a softmax value > 0.5."
        )
        
    # 1. Start with a baseline of low-variance random noise
    logits = np.random.randn(n_low, n_high) * noise_std

    # 2. Build the guaranteed target map
    targets = np.zeros(n_low, dtype=int)
    targets[:n_high] = np.arange(n_high)
    
    # Fill any remaining rows with random column assignments
    if n_low > n_high:
        targets[n_high:] = np.random.randint(0, n_high, size=(n_low - n_high,))
    
    # Shuffle row assignments so guaranteed coverage isn't locked to the top rows
    # np.random.permutation returns a shuffled copy of the target array
    targets = np.random.permutation(targets)
    
    # 3. Calculate the exact structural margin needed
    min_margin = math.log(max(1, n_high - 1)) 
    
    # Add a 5-sigma buffer to ensure noise doesn't pull the softmax below 0.5
    margin = min_margin + (5 * noise_std) + 0.1
    
    # 4. Inject the margin into the targeted positions using advanced integer indexing
    logits[np.arange(n_low), targets] += margin
    
    return logits


def find_sinks_modulo_undirected_connected(A_high):
    """
    Finds the sink nodes of a DAG and groups them based on whether 
    they belong to the same undirected connected component.
    
    Parameters:
    A_high (np.ndarray): Adjacency matrix where A[i, j] != 0 implies a directed edge i -> j.
    
    Returns:
    list of lists: Groups of sink node indices clustered by undirected connectedness.
    """
    # Ensure binary matrix representation
    A = (A_high != 0).astype(int)
    
    # 1. Identify sink nodes (out-degree == 0)
    out_degrees = np.sum(A, axis=1)
    sink_indices = np.where(out_degrees == 0)[0]
    
    if len(sink_indices) == 0:
        return []
    
    # 2. Construct the undirected adjacency matrix for the entire graph
    A_undirected = (A | A.T).astype(int)
    
    # 3. Find the connected components of the entire undirected graph
    _, labels = connected_components(A_undirected, directed=False)
    
    # 4. Group the isolated sink indices by their global component labels
    grouped_sinks = {}
    for idx in sink_indices:
        component_label = labels[idx]
        if component_label not in grouped_sinks:
            grouped_sinks[component_label] = []
        grouped_sinks[component_label].append(idx)
        
    return list(grouped_sinks.values())

def normalize_massidda_numpy(W, T_hard, epsilon=1e-12):
    """
    Normalizes the low-level adjacency matrix W to satisfy Massidda's
    causal abstraction equations for a binary partition T_hard.
    
    W: ndarray of shape (P, P)
    T_hard: Binary ndarray of shape (P, K)
    """
    K = T_hard.shape[1]
    W_norm = W.copy()
    
    # Extract row indices for each cluster
    cluster_indices = [np.where(T_hard[:, k] == 1)[0] for k in range(K)]
    
    # Step 1: Compute internal propagation vectors (v_j)
    v = []
    for j in range(K):
        idx_j = cluster_indices[j]
        if len(idx_j) == 0:
            v.append(np.array([]))
            continue
            
        # Extract internal block W_jj
        W_jj = W[np.ix_(idx_j, idx_j)]
        
        I = np.eye(len(idx_j))
        # v_j = (I - W_jj)^-1 @ 1
        v_j = np.linalg.inv(I - W_jj) @ np.ones(len(idx_j))
        v.append(v_j)
        
    # Step 2: Rescale cross-cluster row projections
    for j in range(K):
        idx_j = cluster_indices[j]
        if len(idx_j) == 0:
            continue
            
        v_j = v[j]
        
        for i in range(K):
            if i == j:
                continue
                
            idx_i = cluster_indices[i]
            if len(idx_i) == 0:
                continue
                
            # Extract cross-cluster block W_ij
            W_ij = W_norm[np.ix_(idx_i, idx_j)]
            
            # S: Current row projections (shape: |C_i|)
            S = W_ij @ v_j
            
            # Process only if there is mass flowing from j to i
            if np.sum(S) > epsilon:
                # Target macro-weight: mean of active (non-zero) row projections
                active_mask = S > epsilon
                m_ij = np.mean(S[active_mask]) if np.any(active_mask) else 0.0
                
                # Calculate row-wise scale factors. Bypass rows with zero projection.
                scale_factors = np.where(active_mask, m_ij / (S + 1e-24), 1.0)
                
                # Apply scaling and update the matrix block
                W_norm[np.ix_(idx_i, idx_j)] = W_ij * scale_factors[:, None]
                
    return W_norm

def solve_massidda_cosine(A_low, T_hard, eps=1e-12, dag_mask=None):
    """
    Solves the Massidda equations using a scale-free cosine similarity objective.
    
    Parameters:
    - A_low: Micro-level adjacency matrix (p x p).
    - T_hard: Initial hard clustering matrix (p x k).
    - A_high_mask: Boolean/Binary matrix indicating valid DAG edges (k x k).
    - sink_anchors: Dictionary mapping cluster_index (int) -> 1D numpy array of anchor values.
    """
    row, col = np.nonzero(T_hard)
    T_hard = T_hard.copy()
    T_hard[row, col] = 1
    p, k = T_hard.shape

    # 1. Precompute static cluster properties
    cluster_sizes = np.sum(T_hard, axis=0).astype(int)
    clusters = [np.nonzero(T_hard[:, i])[0] for i in range(k)]
    offset = [0]
    for N in cluster_sizes:
        offset.append(offset[-1] + N)

    inv_I_A = [np.linalg.inv(np.eye(cluster_sizes[j]) - A_low[np.ix_(clusters[j], clusters[j])]) for j in range(k)]

    A_low_normalised = normalize_massidda_numpy(W=A_low,T_hard=T_hard)

    
    A_high = get_A_high(T_hard, A_low_normalised)
    A_high = A_high * dag_mask if dag_mask is not None else enforce_dag(A_high)
    A_high_mask = (A_high != 0) * 1.

    sink_indices_list = find_sinks_modulo_undirected_connected(A_high)
    sink_anchors = {}
    for component in sink_indices_list:
        sink = component[0]
        cluster_size = np.count_nonzero(T_hard[:, sink])
        t_sink = np.ones(cluster_size, dtype=float)
        t_sink /= np.linalg.norm(t_sink)

        sink_anchors[sink] = t_sink


    # 2. Build Vectorized Sparse Block Operator (M_sparse)
    edges = np.argwhere(A_high_mask == 1)
    n_edges = len(edges)

    row_to_edge = []
    row_to_node_i = []
    rows, cols, data = [], [], []

    current_row = 0
    for edge_idx, (i, j) in enumerate(edges):
        N_i = cluster_sizes[i]
        M_ij = A_low[np.ix_(clusters[i], clusters[j])] @ inv_I_A[j]

        for r_idx in range(N_i):
            orig_i = clusters[i][r_idx]
            for c_idx in range(cluster_sizes[j]):
                orig_j = clusters[j][c_idx]
                rows.append(current_row + r_idx)
                cols.append(orig_j)
                data.append(M_ij[r_idx, c_idx])

            row_to_edge.append(edge_idx)
            row_to_node_i.append(orig_i)

        current_row += N_i

    M_sparse = sp.csr_matrix((data, (rows, cols)), shape=(current_row, p))
    row_to_edge = np.array(row_to_edge, dtype=int)
    row_to_node_i = np.array(row_to_node_i, dtype=int)

    # 3. Parameter Masking (Separating learnable vs anchored variables)
    fixed_indices = []
    fixed_values = []
    for sink_c, anchor_v in sink_anchors.items():
        fixed_indices.extend(clusters[sink_c])
        fixed_values.extend(anchor_v)

    fixed_indices = np.array(fixed_indices, dtype=int)
    fixed_values = np.array(fixed_values, dtype=float)

    learn_mask = np.ones(p, dtype=bool)
    learn_mask[fixed_indices] = False
    learn_indices = np.where(learn_mask)[0]

    def get_full_t(x):
        t = np.zeros(p)
        t[fixed_indices] = fixed_values
        t[learn_indices] = x
        return t

    # 4. Objective and Gradient
    def objective_and_grad(x):
        t = get_full_t(x)

        V = M_sparse @ t
        U = t[row_to_node_i]

        V_dot_U = np.bincount(row_to_edge, weights=V * U, minlength=n_edges)
        V_dot_V = np.bincount(row_to_edge, weights=V * V, minlength=n_edges)
        V_dot_V_safe = V_dot_V + eps

        # Cosine mismatch penalty per edge
        f_e = 1.0 - (V_dot_U**2) / V_dot_V_safe
        f = np.sum(f_e)

        # Gradients per row
        V_dot_U_per_row = V_dot_U[row_to_edge]
        V_dot_V_per_row = V_dot_V_safe[row_to_edge]

        # Partial wrt U (t_i)
        grad_U_per_row = -2.0 * (V_dot_U_per_row / V_dot_V_per_row) * V
        grad_t_from_U = np.bincount(row_to_node_i, weights=grad_U_per_row, minlength=p)

        # Partial wrt V (M_ij t_j)
        grad_V_per_row = -2.0 * (V_dot_U_per_row / V_dot_V_per_row) * U + \
                          2.0 * ((V_dot_U_per_row**2) / (V_dot_V_per_row**2)) * V
        grad_t_from_V = M_sparse.T @ grad_V_per_row

        grad_t_full = grad_t_from_U + grad_t_from_V

        return f, grad_t_full[learn_indices]

    def objective(x):
        return objective_and_grad(x)[0]

    def grad(x):
        return objective_and_grad(x)[1]

    # 5. Constraints: Unit norm for non-sink clusters
    constraints = []
    
    def make_con(mask):
        return lambda x: np.sum(x[mask]**2) - 1.0

    def make_jac(mask):
        return lambda x: 2.0 * x * mask

    for c in range(k):
        if c in sink_anchors:
            continue
        c_mask_in_x = np.isin(learn_indices, clusters[c])
        constraints.append({
            'type': 'eq',
            'fun': make_con(c_mask_in_x),
            'jac': make_jac(c_mask_in_x)
        })

    # Initialization: Ones on the hypersphere
    x0 = np.ones(len(learn_indices))
    for c in range(k):
        if c in sink_anchors:
            continue
        c_mask_in_x = np.isin(learn_indices, clusters[c])
        norm = np.linalg.norm(x0[c_mask_in_x])
        if norm > 0:
            x0[c_mask_in_x] /= norm

    # 6. Optimization
    start_time = time.time()
    res = minimize(objective, x0, jac=grad, constraints=constraints, method='SLSQP', tol=1e-9, options={'maxiter': 1000})
    runtime = time.time() - start_time
    print(f"res.fun={res.fun:.6e}  time={runtime:.4f}s  res.message={res.message}")

    # 7. Final Reconstructions
    t_final = get_full_t(res.x)
    A_high = np.zeros((k, k))
    T_final = np.zeros((p, k))

    for c in range(k):
        T_final[clusters[c], c] = t_final[clusters[c]]
    
    A_high = get_A_high(T_final, A_low) * A_high_mask

    return A_high, T_final

def process_output(T_hard, A_low, E_low=None, n_restarts=10, eps=1e-9, verbose=False):
    A_high_pred, T_pred = fit_abstraction_alternative(A_low=A_low, E_low=E_low, T_hard=T_hard, n_restarts=n_restarts) #alternating_optimisation(A_low=A_low,T_hard=T_hard, dag_mask=dag_mask, n_iters=n_iters)
    S_pred = get_S(T=T_pred, A_low=A_low, T_hard=T_hard)
    
    log(f"Min T value: {np.min(np.abs(T_pred[T_pred != 0]))}.",verbose=verbose)
    
    return S_pred, T_pred, A_high_pred

def solve_massidda_cosine_new(A_low, T_hard, eps=1e-12, dag=True):
    """
    Solves the Massidda equations using a scale-free cosine similarity objective.
    
    Parameters:
    - A_low: Micro-level adjacency matrix (p x p).
    - T_hard: Initial hard clustering matrix (p x k).
    """
    row, col = np.nonzero(T_hard)
    T_hard = T_hard.copy()
    T_hard[row, col] = 1
    p, k = T_hard.shape
    active_c = np.where((T_hard.sum(axis=0) > 0))[0]

    # 1. Precompute static cluster properties
    cluster_sizes = np.sum(T_hard, axis=0).astype(int)
    clusters = [np.nonzero(T_hard[:, i])[0] for i in range(k)]
    offset = [0]
    for N in cluster_sizes:
        offset.append(offset[-1] + N)

    inv_I_A = [np.linalg.inv(np.eye(cluster_sizes[j]) - A_low[np.ix_(clusters[j], clusters[j])]) for j in range(k)]

    A_low_normalised = normalize_massidda_numpy(W=A_low,T_hard=T_hard)

    
    A_high = get_A_high(T_hard, A_low_normalised)
    A_high = enforce_dag(A_high) if dag else A_high
    A_high_mask = (A_high != 0) * 1.

    sink_indices_list = find_sinks_modulo_undirected_connected(A_high)
    sink_anchors = [component[0] for component in sink_indices_list if component[0] in active_c]

    # 2. Build Vectorized Sparse Block Operator (M_sparse)
    edges = np.argwhere(A_high_mask == 1)
    n_edges = len(edges)

    row_to_edge = []
    row_to_node_i = []
    rows, cols, data = [], [], []

    current_row = 0
    for edge_idx, (i, j) in enumerate(edges):
        N_i = cluster_sizes[i]
        M_ij = A_low[np.ix_(clusters[i], clusters[j])] @ inv_I_A[j]

        for r_idx in range(N_i):
            orig_i = clusters[i][r_idx]
            for c_idx in range(cluster_sizes[j]):
                orig_j = clusters[j][c_idx]
                rows.append(current_row + r_idx)
                cols.append(orig_j)
                data.append(M_ij[r_idx, c_idx])

            row_to_edge.append(edge_idx)
            row_to_node_i.append(orig_i)

        current_row += N_i

    M_sparse = sp.csr_matrix((data, (rows, cols)), shape=(current_row, p))
    row_to_edge = np.array(row_to_edge, dtype=int)
    row_to_node_i = np.array(row_to_node_i, dtype=int)

    # 3. Parameter Masking (Separating learnable vs anchored variables)
    fixed_indices = []
    fixed_values = []
    for sink_c in sink_anchors:
        fixed_indices.append(clusters[sink_c][0])
        fixed_values.append(1)

    fixed_indices = np.array(fixed_indices, dtype=int)
    fixed_values = np.array(fixed_values, dtype=float)

    learn_mask = np.ones(p, dtype=bool)
    learn_mask[fixed_indices] = False
    learn_indices = np.where(learn_mask)[0]

    def get_full_t(x):
        t = np.zeros(p)
        t[fixed_indices] = fixed_values
        t[learn_indices] = x
        return t

    # 4. Objective and Gradient
    def objective_and_grad(x):
        t = get_full_t(x)

        V = M_sparse @ t
        U = t[row_to_node_i]

        V_dot_U = np.bincount(row_to_edge, weights=V * U, minlength=n_edges)
        V_dot_V = np.bincount(row_to_edge, weights=V * V, minlength=n_edges)
        V_dot_V_safe = V_dot_V + eps

        # Cosine mismatch penalty per edge
        f_e = 1.0 - (V_dot_U**2) / V_dot_V_safe
        f = np.sum(f_e)

        # Gradients per row
        V_dot_U_per_row = V_dot_U[row_to_edge]
        V_dot_V_per_row = V_dot_V_safe[row_to_edge]

        # Partial wrt U (t_i)
        grad_U_per_row = -2.0 * (V_dot_U_per_row / V_dot_V_per_row) * V
        grad_t_from_U = np.bincount(row_to_node_i, weights=grad_U_per_row, minlength=p)

        # Partial wrt V (M_ij t_j)
        grad_V_per_row = -2.0 * (V_dot_U_per_row / V_dot_V_per_row) * U + \
                          2.0 * ((V_dot_U_per_row**2) / (V_dot_V_per_row**2)) * V
        grad_t_from_V = M_sparse.T @ grad_V_per_row

        grad_t_full = grad_t_from_U + grad_t_from_V

        return f, grad_t_full[learn_indices]

    def objective(x):
        return objective_and_grad(x)[0]

    def grad(x):
        return objective_and_grad(x)[1]

    # 5. Constraints: Unit norm for non-sink clusters
    constraints = []
    
    def make_con(mask):
        return lambda x: np.sum(x[mask]**2) - 1.0

    def make_jac(mask):
        # Returns a vector of length n_learn
        def jacobian(x):
            grad = np.zeros(len(learn_indices))
            grad[mask] = 2.0 * x[mask]
            return grad
        return jacobian

    for c in active_c:
        c_mask_in_x = np.isin(learn_indices, clusters[c])
        if c in sink_anchors:
            continue
        else:
            constraints.append({
                'type': 'eq',
                'fun': make_con(c_mask_in_x),
                'jac': make_jac(c_mask_in_x)
            })

    # Initialization: Ones on the hypersphere
    #np.random.seed(42)
    x0 = np.random.randn(len(learn_indices))
    for c in active_c:
        c_mask_in_learn = np.isin(learn_indices, clusters[c])
        cluster_x0_slice = x0[c_mask_in_learn]
        norm = np.linalg.norm(cluster_x0_slice)
        
        if norm > 0:
            if c in sink_anchors:
                continue
            else:
                x0[c_mask_in_learn] = cluster_x0_slice / norm

    # 6. Optimization
    start_time = time.time()
    res = minimize(objective, x0, jac=grad, constraints=constraints, method='SLSQP', tol=1e-9, options={'maxiter': 1000})
    runtime = time.time() - start_time
    #print(f"res.fun={res.fun:.6e}  time={runtime:.4f}s  res.message={res.message}")

    # 7. Final Reconstructions
    t_final = get_full_t(res.x)
    A_high = np.zeros((k, k))
    T_final = np.zeros((p, k))

    for c in active_c:
        t_c = t_final[clusters[c]]
        norm_c = np.linalg.norm(t_c)
        if norm_c > 0:
            t_c = t_c/norm_c
        T_final[clusters[c], c] = t_c
    
    A_high = get_A_high(T_final, A_low) * A_high_mask

    return A_high, T_final

def cluster_indices(T_hard):
    """List of low-level node indices belonging to each cluster."""
    K = T_hard.shape[1]
    return [np.where(T_hard[:, k] == 1)[0] for k in range(K)]
 
 
def high_level_mask(A_low, T_hard):
    """Edge (i, j) exists iff some member of cluster i has a nonzero direct
    A_low connection to some member of cluster j (i.e. i causes j)."""
    idx = cluster_indices(T_hard)
    k = len(idx)
    A_high = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j or len(idx[i]) == 0 or len(idx[j]) == 0:
                continue
            W_ij = A_low[np.ix_(idx[i], idx[j])]
            if not np.any(W_ij):
                continue
            W_jj = A_low[np.ix_(idx[j], idx[j])]
            M_ij = W_ij @ np.linalg.inv(np.eye(len(idx[j])) - W_jj)
            A_high[i, j] = np.linalg.norm(M_ij, "fro")
    
    A_high_dag = enforce_dag(A_high)
    mask = (A_high_dag != 0)
    return mask
 
 
def reverse_topo_order(mask):
    """Order clusters so every node appears after all of its children."""
    K = mask.shape[0]
    children = [np.where(mask[k, :])[0] for k in range(K)]
    visited = [False] * K
    order = []
 
    def visit(n):
        if visited[n]:
            return
        visited[n] = True
        for c in children[n]:
            visit(c)
        order.append(n)
 
    for k in range(K):
        visit(k)
    return order 
 

 
def fit_abstraction(A_low, T_hard, E_low=None,n_restarts=10, seed=0, maxiter=100):
    """
    Given fixed A_low and a binary partition T_hard, construct the continuous
    abstraction T and A_high by optimising sink t-vectors and propagating
    bottom-up via the Massidda equations W_ij(I-W_jj)^{-1}t_j = m_ij t_i.

    Returns:
        A_high: (n_high, n_high) abstract adjacency matrix
        T: (n_low, n_high) continuous abstraction matrix
        diagnostics: dict with 'penalty' (multi-child disagreement at optimum)
    """
    idx = cluster_indices(T_hard)
    mask = high_level_mask(A_low, T_hard)
    order = reverse_topo_order(mask)
    sinks = [X for X in order if mask[X, :].sum() == 0]
    sizes = [len(idx[X]) for X in sinks]
    split = np.cumsum([0] + sizes)

    # Precompute M[(X,c)] = W_Xc @ (I - W_cc)^{-1} once for every edge.
    # These are constant across all optimisation iterations.
    M_edge = {}
    for i in order:
        for j in np.where(mask[i, :])[0]:
            W_jj = A_low[np.ix_(idx[j], idx[j])]
            W_ij = A_low[np.ix_(idx[i], idx[j])]
            M_edge[(i, int(j))] = W_ij @ np.linalg.inv(np.eye(len(idx[j])) - W_jj)

    def propagate(dirs):
        t = [None] * len(idx)
        penalty = 0.0
        sink_iter = iter(dirs)
        for X in order:
            children = np.where(mask[X, :])[0]
            if len(children) == 0:
                t[X] = next(sink_iter)
                continue
            votes = []
            for c in children:
                v = M_edge[(X, int(c))] @ t[c]
                n = np.linalg.norm(v)
                votes.append(v / n if n > 1e-12 else v)
            if len(votes) == 1:
                t[X] = votes[0]
            else:
                V = np.stack(votes)
                for r in range(1, len(V)):
                    if V[r] @ V[0] < 0:
                        V[r] = -V[r]
                mean_v = V.mean(axis=0)
                n = np.linalg.norm(mean_v)
                mean_v = mean_v / n if n > 1e-12 else mean_v
                t[X] = mean_v
                # ||v - (v·m)m||² = 1 - cos²(v,m) for unit vectors
                for v in V:
                    penalty += 1.0 - (v @ mean_v) ** 2
        return t, penalty

    def unpack(x):
        dirs = []
        for k in range(len(sinks)):
            seg = x[split[k]:split[k + 1]]
            n = np.linalg.norm(seg)
            dirs.append(seg / n if n > 1e-12 else seg)
        return dirs

    n_params = int(split[-1]) if len(split) else 0
    if n_params == 0:
        best_x, best_val = np.array([]), 0.0
    else:
        rng = np.random.RandomState(seed)
        best_x, best_val = None, np.inf
        for _ in range(n_restarts):
            x0 = rng.randn(n_params)
            res = minimize(lambda x: propagate(unpack(x))[1], x0,
                           method="SLSQP", tol=1e-9, options={"maxiter": maxiter})
            if res.fun < best_val:
                best_val, best_x = res.fun, res.x

    t, penalty = propagate(unpack(best_x) if n_params > 0 else [])

    p, K = A_low.shape[0], len(idx)
    T = np.zeros((p, K))
    A_high = np.zeros((K, K))
    for X in range(K):
        if t[X] is not None and len(idx[X]) > 0:
            T[idx[X], X] = t[X]
        for c in np.where(mask[X, :])[0]:
            vote_unnorm = M_edge[(X, int(c))] @ t[c]
            A_high[X, int(c)] = t[X] @ vote_unnorm  # t[X] is unit-norm so denominator = 1

    return A_high, T

def fit_abstraction_alternative(A_low, E_low, T_hard, n_restarts=10, seed=None, maxiter=100):
    """
    Given fixed E_low and binary partition matrix T_hard, optimise the non-zero values of T
    so that with LHS = (I + F_pi @ (E_low - I))^{-1} @ F_pi @ E @ T , the columns of LHS
    lie in the span of T[:,i] for each cluster i.
    """
    p,k = T_hard.shape
    if E_low is None:
        E_low = np.linalg.inv(np.eye(p) - A_low)

    idx = cluster_indices(T_hard)
    row_indices = np.concatenate(idx) if k else np.array([], dtype=int)
    col_indices = np.repeat(np.arange(k), [len(ix) for ix in idx])

    # term = I + F_pi @ (E_low - I) depends only on T_hard and E_low (both fixed for the
    # whole optimisation), never on the free vector t -- so it's solved for ONCE here rather
    # than refactorised on every objective call. LHS(T) = term^{-1} @ F_pi @ E_low @ T is
    # linear in T, so folding the (constant) left-hand factor into B = term^{-1} @ F_pi @ E_low
    # turns each subsequent evaluation into a cheap matmul B @ T instead of an O(p^3) solve.
    I = np.eye(p)
    F_pi = (T_hard.T)[..., None] * I                      # (n_high, n_low, n_low)
    term = I + F_pi @ (E_low - I)                          # (n_high, n_low, n_low)
    B = np.linalg.solve(term, F_pi @ E_low)                # (n_high, n_low, n_low), constant
    B_t = torch.from_numpy(B)
    row_idx_t = torch.from_numpy(row_indices).long()
    col_idx_t = torch.from_numpy(col_indices).long()

    def objective_and_grad(t_np):
        t = torch.tensor(t_np, dtype=torch.float64, requires_grad=True)
        T = torch.zeros((p, k), dtype=torch.float64).index_put((row_idx_t, col_idx_t), t)

        LHS = B_t @ T                                                      # (n_high, n_low, n_high)

        # proj_coeff[i,j] = T[:,i]^T LHS[i,:,j] / ||T[:,i]||^2
        t_norms_sq = (T ** 2).sum(dim=0).clamp(min=1e-8)                   # (n_high,)
        proj_coeff = torch.einsum('ij,ijk->ik', T.T, LHS) / t_norms_sq.unsqueeze(1)  # (n_high, n_high)

        # rank1[i] = P_i LHS[i]  where P_i = T[:,i] T[:,i]^T / ||T[:,i]||^2
        rank1 = T.T.unsqueeze(2) * proj_coeff.unsqueeze(1)                  # (n_high, n_low, n_high)

        # ||(I - P_i) LHS[i]||_F^2 summed over i
        residual = LHS - rank1                                             # (n_high, n_low, n_high)
        f = (residual ** 2).mean()

        f.backward()
        return f.item(), t.grad.numpy().astype(np.float64)

    # Without a scale constraint, shrinking any column of T toward 0 strictly decreases
    # this residual (T=0 gives exactly 0), so SLSQP needs the same per-cluster unit-norm
    # equality constraint used in solve_massidda_cosine to avoid the degenerate optimum.
    cluster_sizes = [len(ix) for ix in idx]
    offsets = np.cumsum([0] + cluster_sizes)
    constraints = []

    def make_con(lo, hi):
        return lambda x: np.sum(x[lo:hi] ** 2) - 1.0

    def make_jac(lo, hi, n):
        def jacobian(x):
            g = np.zeros(n)
            g[lo:hi] = 2.0 * x[lo:hi]
            return g
        return jacobian

    for j in range(k):
        lo, hi = offsets[j], offsets[j + 1]
        if hi == lo:
            continue
        constraints.append({'type': 'eq', 'fun': make_con(lo, hi), 'jac': make_jac(lo, hi, p)})

    rng = np.random.RandomState(seed)
    best_t, best_val = None, np.inf
    for _ in range(n_restarts):
        t0 = rng.randn(p)
        for j in range(k):
            lo, hi = offsets[j], offsets[j + 1]
            if hi > lo:
                n = np.linalg.norm(t0[lo:hi])
                if n > 1e-12:
                    t0[lo:hi] /= n
        res = minimize(objective_and_grad, t0, jac=True, method='SLSQP', constraints=constraints, tol=1e-9, options={"maxiter": maxiter})
        if res.fun < best_val:
            best_t, best_val = res.x, res.fun

    T = np.zeros((p,k))
    T[row_indices,col_indices] = best_t
    A_high = get_A_high(T=T, A_low=A_low)

    return A_high, T


def is_non_sink_subgraph_strongly_connected(weight_matrix: np.ndarray) -> bool:
    """
    Checks whether the induced subgraph of non-sink nodes is strongly connected.
    
    Assumes weight_matrix[i, j] != 0 indicates a directed edge X_i -> X_j.
    A sink node is defined as having an out-degree of 0 (an all-zero row).

    Parameters
    ----------
    weight_matrix : np.ndarray
        A square 2D NumPy array representing the adjacency/weight matrix.

    Returns
    -------
    bool
        True if the subgraph of non-sinks is strongly connected (or trivial),
        False otherwise.
    """
    A = np.asarray(weight_matrix)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("weight_matrix must be a square 2D array.")
    
    # Out-degree corresponds to non-zero entries along rows (X_i -> X_j)
    out_degrees = np.count_nonzero(A, axis=1)
    non_sinks = np.where(out_degrees > 0)[0]
    
    k = len(non_sinks)
    if k <= 1:
        return True  # Graph with 0 or 1 node is trivially strongly connected
    
    # Extract induced submatrix for non-sink nodes
    A_sub = A[np.ix_(non_sinks, non_sinks)]
    
    # Initialize reachability matrix (with self-loops)
    reachability = (A_sub != 0).copy()
    np.fill_diagonal(reachability, True)
    
    # Vectorized Floyd-Warshall transitive closure
    for idx in range(k):
        reachability |= reachability[:, idx:idx+1] & reachability[idx:idx+1, :]
        
    return bool(np.all(reachability))