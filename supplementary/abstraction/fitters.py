"""Abstraction fitters: propagation (`fit_abstraction`), the rank-one form
(`fit_abstraction_K`), and the router `fit_abstraction_auto`.

The closed-form eigen fitter lives in `eigen_solver.py`.
"""
import numpy as np
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp
import math
import torch
import torch.nn as nn
from typing import Literal
from scipy.optimize import minimize
import time
from .losses import abstraction_residual, observation_loss
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

def get_S(T, E_low=None, T_hard=None, A_low=None):
    """S_i = (I - A_low[pi_i, pi_i])^-1 t_i, supported on cluster i.

    FIXED 2026-09-23. The previous version returned E_low[pi_i, pi_i] @ t_i. That is only
    correct when the QUOTIENT is acyclic. The abstraction equation (I - A_low) S = T (I-A_high)
    has diagonal block (I - A_low[pi_i, pi_i]) S_ii = t_i -- every cross term dies because T
    and S are both block-supported, and the only A_high entry that could appear is the
    diagonal, which is zero -- so S_ii is the LOCAL inverse. E_low's diagonal block instead
    also counts paths that leave cluster i and come back, and with a cyclic quotient the two
    differ by O(1): the generating abstraction scored RAE = 0.02-0.21 against ITSELF (squared
    scale, i.e. 0.14-0.46 under the ratio-of-norms RAE in use since 2026-09-24). Random
    partitions induce a cyclic quotient 87-100% of the time even on acyclic graphs, so control
    arms were affected too. See experiments/abstraction_study/REPORT.md.

    Equivalently S_i = e_ii^-1 E_low[pi_i, pi_i] t_i, from the (pi_i, i) block of
    E_low T = S E_high. That form is a VECTOR identity in the direction t_i holding only under
    exactness -- as matrices e_ii^-1 E_ii and (I - A_ii)^-1 differ by up to 4.0 on cyclic
    instances -- so this uses the local inverse, which needs no such assumption and is correct
    on approximate fits too.

    Pass A_low. E_low is accepted only so older call sites keep working; A_low is then
    recovered from it at the cost of one extra O(p^3) inverse.
    """
    if A_low is None:
        if E_low is None:
            raise ValueError("get_S requires A_low (preferred) or E_low")
        A_low = np.eye(E_low.shape[0]) - np.linalg.inv(E_low)
    if T_hard is None:
        T_hard = (T != 0).astype(int)
    idx = cluster_indices(T_hard)
    S = np.zeros_like(T)
    for i in range(T.shape[1]):
        if len(idx[i]) == 0:
            continue
        A_ii = A_low[np.ix_(idx[i], idx[i])]
        S[idx[i], i] = np.linalg.solve(np.eye(len(idx[i])) - A_ii, T[idx[i], i])
    return S

def get_A_high(T, A_low, priority='abstraction'):
    n_low,n_high = T.shape
    if priority == 'recon':
        with np.errstate(divide='ignore', invalid='ignore'):
            T_norm = T / np.sum(T**2, axis=0)
        
        T_norm = np.nan_to_num(T_norm)
        A_high = T_norm.T @ A_low @ T

    elif priority == 'abstraction':
        E_low = np.linalg.inv(np.eye(n_low) - A_low)
        S = get_S(T, E_low, A_low=A_low)
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

def process_output(T_hard, A_low, E_low=None, n_restarts=5, maxiter=500, seed=0, eps=1e-9,
                   faithful=True, verbose=False):
    if E_low is None:
        E_low = np.linalg.inv(np.eye(A_low.shape[0]) - A_low)
    # faithful=True (default since 2026-09-23): maximise ||A_high|| subject to the residual,
    # rather than minimising the residual and taking whichever restart happened to have the
    # largest ||A_high||. The old default is T-UNFAITHFUL -- it attenuates real high-level
    # edges by up to 7 orders of magnitude while still achieving RAE = 0, and reconstructs
    # A at 0.42-0.96 relative error where the faithful fit reaches ~1e-12. Costs ~2-6x more.
    # Pass faithful=False to reproduce results generated before this date.
    A_high_pred, T_pred = fit_abstraction_auto(A_low=A_low, T_hard=T_hard, E_low=E_low,
                                                      n_restarts=n_restarts, maxiter=maxiter,
                                                      seed=seed, faithful=faithful)[:2]
    S_pred = get_S(T=T_pred, E_low=E_low, T_hard=T_hard, A_low=A_low)
    
    log(f"Min T value: {np.min(np.abs(T_pred[T_pred != 0]))}.",verbose=verbose)
    
    return S_pred, T_pred, A_high_pred
   

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
 
 
def quotient_mask(A_low, T_hard):
    """Boolean (k, k) cluster graph: mask[i, j] iff some node of cluster j has a direct
    A_low edge into cluster i.

    UNLIKE `high_level_mask` this does NOT run `enforce_dag`, so a cyclic quotient stays
    cyclic. That is the whole point: `high_level_mask` always looks acyclic, so it cannot be
    used to decide whether the quotient actually is.
    """
    idx = cluster_indices(T_hard)
    k = len(idx)
    mask = np.zeros((k, k), dtype=bool)
    for i in range(k):
        for j in range(k):
            if i == j or len(idx[i]) == 0 or len(idx[j]) == 0:
                continue
            mask[i, j] = bool(np.any(A_low[np.ix_(idx[i], idx[j])]))
    return mask


def is_quotient_acyclic(A_low, T_hard):
    """Is the cluster graph induced by T_hard on A_low a DAG? O(p^2 + k^3)."""
    return is_dag(quotient_mask(A_low, T_hard))


def fit_abstraction_auto(A_low, T_hard, E_low=None, n_restarts=10, seed=None, maxiter=100,
                         verbose=False, faithful=True, return_method=False):
    """Fit T and A_high with whichever solver is right for THIS quotient.

    Neither solver is best everywhere, and the deciding property is whether the quotient
    (cluster graph) is acyclic -- which happens for ground-truth partitions on DAGs, and for
    All-In + one-child partitions, but NOT for bad/random partitions (87-100% cyclic) or for
    genuinely cyclic models.

        acyclic quotient -> fit_abstraction        (propagation + faithful)
        cyclic  quotient -> fit_abstraction_K      (faithful=True)

    Measured reconstruction on ground-truth partitions at blockiness=1, heterogeneity=0
    (the regime where reconstruction can discriminate -- see FITTER_SWITCH.md):

                     p=24     p=40     p=60     p=100
        acyclic  fit_abstraction   1.9e-07  6.9e-07  1.3e-07  2.6e-07
                 _alternative      6.5e-07  6.4e-07  1.7e-05  7.6e-01
                 _K faithful       2.6e-06  1.2e-06  1.4e-06  6.6e-01
        cyclic   fit_abstraction   2.1e-01  2.6e-01  2.4e-01  2.6e-01
                 _K faithful       9.4e-06  7.4e-06  1.2e-06  1.2e-06

    `fit_abstraction` is the only solver that still works at p=100, and it is the fastest on
    acyclic quotients because only the SINK t-vectors are free. It cannot do cyclic quotients
    at all: `high_level_mask` runs `enforce_dag`, deleting the cycle edges before the fit
    starts. `fit_abstraction_K(faithful=True)` handles those at every size tested.

    Returns (A_high, T), or (A_high, T, method) when return_method=True.
    """
    p = A_low.shape[0]
    if E_low is None:
        E_low = np.linalg.inv(np.eye(p) - A_low)

    mask = quotient_mask(A_low, T_hard)
    n_edges = int(mask.sum())

    def _out(A_high, T, method):
        log(f"fit_abstraction_auto: quotient has {n_edges} cross-cluster edges -> {method}",
            verbose=verbose)
        return (A_high, T, method) if return_method else (A_high, T)

    if is_dag(mask):
        try:
            A_high, T = fit_abstraction(A_low=A_low, T_hard=T_hard, E_low=E_low,
                                        n_restarts=n_restarts, seed=(0 if seed is None else seed),
                                        maxiter=maxiter, faithful=faithful)
            # Guard against a silently degenerate result: an edgeless A_high on a quotient
            # that demonstrably has edges is the annihilating solution, not an abstraction.
            if np.all(np.isfinite(A_high)) and not (n_edges > 0 and np.linalg.norm(A_high) < 1e-9):
                return _out(A_high, T, 'propagation')
            log("fit_abstraction_auto: propagation returned a degenerate A_high -- using _K",
                verbose=verbose)
        except Exception as e:
            log(f"fit_abstraction_auto: propagation raised {type(e).__name__} -- using _K",
                verbose=verbose)

    A_high, T = fit_abstraction_K(A_low=A_low, E_low=E_low, T_hard=T_hard,
                                  n_restarts=n_restarts, seed=seed, maxiter=maxiter,
                                  faithful=faithful, verbose=verbose)
    return _out(A_high, T, 'K_faithful')


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
 

 
def fit_abstraction(A_low, T_hard, E_low=None,n_restarts=10, seed=0, maxiter=100,
                    faithful=True, ptol=1e-9, delta=1e-9):
    """
    Given fixed A_low and a binary partition T_hard, construct the continuous
    abstraction T and A_high by optimising sink t-vectors and propagating
    bottom-up via the Massidda equations W_ij(I-W_jj)^{-1}t_j = m_ij t_i.

    faithful=False reproduces the historical behaviour exactly (minimise the multi-child
    disagreement penalty). DEFAULT CHANGED TO True on 2026-09-23.

    With faithful=True it instead MAXIMISES ||A_high||^2 subject to that penalty staying
    below ptol. Minimising the penalty alone leaves the sink directions underdetermined, and
    the solutions it happens to pick attenuate real high-level edges: measured on the
    ground-truth partition, the default reconstructs A at 0.05-0.51 relative error while the
    faithful variant reaches 1e-11 to 1e-12 at every density. Cost is ~0.8-2.5s against
    ~0.02s, because the penalty objective is nearly free while this one searches.

    Only the SINK t-vectors are free (everything upstream propagates deterministically), so
    the search is over sum(|sink clusters|) parameters, not p.

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

    def _build(t):
        p_, K_ = A_low.shape[0], len(idx)
        T_ = np.zeros((p_, K_))
        A_ = np.zeros((K_, K_))
        for X in range(K_):
            if t[X] is not None and len(idx[X]) > 0:
                T_[idx[X], X] = t[X]
            for c in np.where(mask[X, :])[0]:
                A_[X, int(c)] = t[X] @ (M_edge[(X, int(c))] @ t[c])
        return A_, T_

    n_params = int(split[-1]) if len(split) else 0
    if n_params == 0:
        best_x, best_val = np.array([]), 0.0
    else:
        rng = np.random.RandomState(seed)
        best_x, best_val = None, np.inf
        if faithful:
            # Same objective as faithful=False -- the multi-child disagreement penalty --
            # differing only by the T-faithfulness constraint, stated on the quantity this
            # solver actually propagates:
            #
            #     ||M_Xc t_c|| / ||M_Xc||_F >= delta,   M_Xc = A_low[pi_X, pi_c](I - A_cc)^-1
            #
            # which is exactly the vector _build turns into A_high[X, c]. This replaces the
            # pre-2026-09-25 device (maximise ||A_high||^2 subject to the penalty staying
            # under ptol); see fit_abstraction_K for why that inversion was the wrong shape.
            #
            # HONEST NOTE ON WHAT THIS BUYS HERE: almost nothing. Unlike the K solver, this
            # one CANNOT easily annihilate -- propagate() renormalises every vote, so every
            # upstream t is unit by construction and only the sink directions are free.
            # Measured over 36 fits (12 ground-truth, 12 AOC Closure, 12 OC Closure) NO arm
            # killed a single edge: 0 of 761 quotient edges under faithful=True, under this
            # constraint, and under faithful=False alike, with the smallest ratio reached
            # being 1.6e-05. The constraint is carried for consistency with
            # fit_abstraction_K, not because it is load-bearing.
            #
            # The real gain over the old faithful path is dropping the ||A_high||^2 search:
            # that scored 21/36 against 23/36 for faithful=False at ~10x the cost (8.66s
            # against 0.88s), i.e. it was slower AND slightly worse.
            edges = [(X, int(c)) for X in range(len(idx)) for c in np.where(mask[X, :])[0]]
            nM = {e: float(np.linalg.norm(M_edge[e])) for e in edges}
            live_e = [e for e in edges if nM[e] > 1e-12]

            def _penalty(x):
                return propagate(unpack(x))[1]

            def _faith(x):
                t_ = propagate(unpack(x))[0]
                out = np.empty(len(live_e))
                for c_, e in enumerate(live_e):
                    tc = t_[e[1]]
                    nt = np.linalg.norm(tc) if tc is not None else 0.0
                    if nt <= 1e-12:
                        out[c_] = -delta ** 2
                    else:
                        out[c_] = (np.linalg.norm(M_edge[e] @ tc) / (nM[e] * nt)) ** 2 - delta ** 2
                return out

            # No analytic jacobian: propagate() renormalises and sign-aligns its votes, so
            # the chain is differentiable only piecewise. SLSQP falls back to finite
            # differences, as the previous faithful path also did.
            cons = [{'type': 'ineq', 'fun': _faith}] if live_e else []
            # Candidates from the constrained AND the unconstrained solve: the unconstrained
            # optimum is usually already feasible here (nothing annihilates), and adding a
            # constraint can leave SLSQP stuck somewhere worse. Best feasible wins.
            for _ in range(n_restarts):
                x0 = rng.randn(n_params)
                for cons_ in ((cons, ()) if cons else ((),)):
                    try:
                        res = minimize(_penalty, x0, method="SLSQP", constraints=cons_,
                                       tol=1e-9, options={"maxiter": maxiter})
                    except Exception:
                        continue
                    if live_e and _faith(res.x).min() < -((delta * 0.01) ** 2):
                        continue
                    if res.fun < best_val:
                        best_val, best_x = res.fun, res.x
            if best_x is None:          # constraint infeasible: fall back to the default
                log("fit_abstraction(faithful): no feasible restart -- falling back to the "
                    "unconstrained penalty path", verbose=False)
                for _ in range(n_restarts):
                    x0 = rng.randn(n_params)
                    res = minimize(_penalty, x0, method="SLSQP", tol=1e-9,
                                   options={"maxiter": maxiter})
                    if res.fun < best_val:
                        best_val, best_x = res.fun, res.x
        else:
            for _ in range(n_restarts):
                x0 = rng.randn(n_params)
                res = minimize(lambda x: propagate(unpack(x))[1], x0,
                               method="SLSQP", tol=1e-9, options={"maxiter": maxiter})
                if res.fun < best_val:
                    best_val, best_x = res.fun, res.x

    t, penalty = propagate(unpack(best_x) if n_params > 0 else [])
    A_high, T = _build(t)   # t[X] is unit-norm, so the projection denominator is 1
    return A_high, T


def fit_abstraction_alternative(A_low, E_low, T_hard, n_restarts=10, seed=None, maxiter=100,
                                faithful=True, rtol=1e-9):
    """
    Given fixed E_low and binary partition matrix T_hard, optimise the non-zero values of T
    so that with LHS = (I + F_pi @ (E_low - I))^{-1} @ F_pi @ E @ T , the columns of LHS
    lie in the span of T[:,i] for each cluster i.

    faithful=False (the pre-2026-09-23 default) MINIMISES that residual and then, among restarts that tie for
    lowest residual, prefers the largest ||A_high||. That is selection among whatever the
    random starts happened to find, not search: measured on the ground-truth partition it
    drops 62-89% of the true cross-cluster edges and reconstructs A at 0.42-0.96 relative
    error, while still achieving RAE = 0. Behaviour is unchanged from before.

    faithful=True swaps the roles -- MAXIMISE ||A_high||^2 subject to the residual staying
    below rtol, keeping the same per-cluster unit-norm equality constraints (without which
    T -> 0 is optimal). This recovers the generating abstraction: reconstruction 1e-11 to
    1e-13 with every true edge retained. Cost is ~0.6-6.9s against ~0.7-1.1s.

    The residual enters as an INEQUALITY, not an equality: a sum of squares has vanishing
    gradient on its own solution manifold, which SLSQP handles poorly as an equality.
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

    def _A_high_from_coefficients(T, B=B, idx=idx):
        p, k = T.shape
        active = [c for c in range(k) if len(idx[c]) > 0]
        if not active:
            return np.zeros((k, k))
        R = np.eye(len(active))
        for a, i in enumerate(active):
            L_i = B[i] @ T                        # (p, k)
            t_i = T[:, i]
            proj = (t_i @ L_i) / (t_i @ t_i)       # (k,)
            R[a, :] = proj[active]
        R_inv = np.linalg.inv(R)
        d = np.diag(R_inv).copy()
        E_high_inv_active = R_inv / d[None, :]     # column-normalise so diag -> 1
        A_high = np.zeros((k, k))
        A_high[np.ix_(active, active)] = np.eye(len(active)) - E_high_inv_active
        return A_high

    def _A_high_torch(T):
        """Differentiable twin of _A_high_from_coefficients, for the faithful objective."""
        active = [c for c in range(k) if len(idx[c]) > 0]
        rows = []
        for i in active:
            L_i = B_t[i] @ T
            t_i = T[:, i]
            rows.append(((t_i @ L_i) / (t_i @ t_i))[active])
        R = torch.stack(rows)
        R_inv = torch.linalg.inv(R)
        d = torch.diagonal(R_inv)
        return torch.eye(len(active), dtype=torch.float64) - R_inv / d.unsqueeze(0), active

    def neg_norm_and_grad(t_np):
        t = torch.tensor(t_np, dtype=torch.float64, requires_grad=True)
        T = torch.zeros((p, k), dtype=torch.float64).index_put((row_idx_t, col_idx_t), t)
        try:
            A_act, _ = _A_high_torch(T)
        except Exception:
            return 0.0, np.zeros_like(t_np)
        f = -(A_act ** 2).sum()
        if not torch.isfinite(f):
            return 0.0, np.zeros_like(t_np)
        f.backward()
        return f.item(), np.nan_to_num(t.grad.numpy().astype(np.float64))

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

    # The residual alone cannot tell a T-faithful abstraction from a T-unfaithful one: both
    # are exact zeros of this objective (e.g. Y2 = X2 - X3 cancelling X1's effect is just as
    # valid a root as the faithful completion). So every restart is kept, not just the best,
    # and among those that tie for lowest residual we break the tie toward the more
    # informative (larger ||A_high||) completion -- preferring a faithful abstraction
    # whenever one coexists with an unfaithful one, and falling back to whatever's best when
    # a faithful completion doesn't exist for this T_hard at all.
    if faithful:
        # Memoise: SLSQP hits `fun` and `jac` at the same x, and each would otherwise run its
        # own forward+backward. See the same note in fit_abstraction_K.
        _rc = {'x': None, 'v': None}

        def _resid(x):
            if _rc['x'] is None or not np.array_equal(x, _rc['x']):
                _rc['x'] = x.copy()
                _rc['v'] = objective_and_grad(x)
            return _rc['v']

        cons_f = [{'type': 'ineq',
                   'fun': lambda x: rtol - _resid(x)[0],
                   'jac': lambda x: -_resid(x)[1]}] + constraints
        rng_f = np.random.RandomState(seed)
        best_x_f, best_f = None, np.inf
        for _ in range(n_restarts):
            t0 = rng_f.randn(p)
            for j in range(k):
                lo, hi = offsets[j], offsets[j + 1]
                if hi > lo:
                    n = np.linalg.norm(t0[lo:hi])
                    if n > 1e-12:
                        t0[lo:hi] /= n
            try:
                res_f = minimize(neg_norm_and_grad, t0, jac=True, method='SLSQP',
                                 constraints=cons_f, tol=1e-9, options={"maxiter": maxiter})
            except Exception:
                continue
            # Feasibility must be checked on BOTH constraints. The residual alone is not
            # enough: it divides by ||t_i||^2, so it is scale-invariant, and a solution with
            # ||t_i|| = 1e10 passes it while being wildly infeasible. Since ||A_high|| grows
            # without bound as T inflates, SLSQP will happily walk off the norm surface --
            # measured: column norms up to 5.9e10 and ||A_high|| = 6.6e13 before this check.
            norm_err = max((abs(np.sum(res_f.x[offsets[j]:offsets[j + 1]] ** 2) - 1.0)
                            for j in range(k) if offsets[j + 1] > offsets[j]), default=0.0)
            if (objective_and_grad(res_f.x)[0] <= max(rtol * 1e3, 1e-6)
                    and norm_err <= 1e-6 and res_f.fun < best_f):
                best_f, best_x_f = res_f.fun, res_f.x
        if best_x_f is not None:
            T_f = np.zeros((p, k))
            T_f[row_indices, col_indices] = best_x_f
            try:
                A_high_f = _A_high_from_coefficients(T=T_f, B=B, idx=idx)
                if np.all(np.isfinite(A_high_f)):
                    return A_high_f, T_f
            except np.linalg.LinAlgError:
                pass
        # no feasible / finite faithful solution: fall through to the default path

    rng = np.random.RandomState(seed)
    candidates = []
    for _ in range(n_restarts):
        t0 = rng.randn(p)
        for j in range(k):
            lo, hi = offsets[j], offsets[j + 1]
            if hi > lo:
                n = np.linalg.norm(t0[lo:hi])
                if n > 1e-12:
                    t0[lo:hi] /= n
        res = minimize(objective_and_grad, t0, jac=True, method='SLSQP', constraints=constraints, tol=1e-9, options={"maxiter": maxiter})

        T_cand = np.zeros((p, k))
        T_cand[row_indices, col_indices] = res.x
        try:
            A_high_cand = _A_high_from_coefficients(T=T_cand, B=B, idx=idx)
            faithfulness = np.linalg.norm(A_high_cand) if np.all(np.isfinite(A_high_cand)) else -1.0
        except np.linalg.LinAlgError:
            A_high_cand, faithfulness = np.zeros((k, k)), -1.0
        candidates.append((res.fun, res.x, A_high_cand, faithfulness))

    best_val = min(c[0] for c in candidates)
    tie_tol = best_val * 1e-3 + 1e-8
    tied = [c for c in candidates if c[0] <= best_val + tie_tol]
    _, best_t, A_high, _ = max(tied, key=lambda c: c[3])

    T = np.zeros((p,k))
    T[row_indices,col_indices] = best_t

    return A_high, T


def fit_abstraction_K(A_low, E_low, T_hard, n_restarts=10, seed=None, maxiter=100,
                      faithful=True, rtol=1e-9, verbose=False, delta=1e-9):
    """Rank-one form of fit_abstraction_alternative.

    Stack K^(i) with j-th column E_ij t_j, where E_ij := E[pi(i), pi(j)]. Because t_j is
    supported on pi(j), that column is just (E T e_j) restricted to pi(i). The abstraction
    condition R^(i) T e_j = (e_ij/e_ii) T e_i is equivalent to E_ij t_j || E_ii t_i -- the
    E_ii^{-1} cancels from both sides of the proportionality -- i.e. rank(K^(i)) <= 1 with
    the i-th column as the designated direction.

    Two consequences over the R^(i)/B_i form:
      * only SUBMATRICES of E are needed, so the O(k p^3) precompute (k dense p x p solves)
        collapses to the single O(p^3) inverse, and a step costs O(p^2) rather than O(k^2 p^2);
      * the residual is measured in un-inverted coordinates, so the loss geometry is not
        distorted by cond(E_ii)^2. Empirically this converges several orders of magnitude
        closer to an exact abstraction on partitions that admit one.

    Same zero set as fit_abstraction_alternative, but different values and gradients off it.

    faithful=True (DEFAULT since 2026-09-24) maximises ||A_high||^2 subject to the residual
    staying below rtol and the per-cluster unit norms, exactly as fit_abstraction_alternative
    does. faithful=False is the historical path: minimise the residual, then take the largest
    ||A_high|| among restarts tying at tie_tol = best*1e-3 + 1e-8.

    That tie-break is not the problem people assume -- with best_res ~ 1e-13 the window is
    ~1e-8 ABSOLUTE and 7-10 of 10 restarts qualify. The problem is that picking the best of
    the restarts that happened to be drawn is SELECTION, NOT SEARCH. Measured on ground-truth
    partitions at blockiness=1, heterogeneity=0 (the only regime where reconstruction can
    discriminate -- at blockiness=0.5 the ORACLE itself reconstructs at 0.87-0.95, so nothing
    can be told apart):

        acyclic p=24   reconstruction 0.496 -> 1.0e-06
        acyclic p=40   reconstruction 0.667 -> 1.2e-06,  recall 0.850 -> 1.000
        cyclic  p=40   reconstruction 8.5e-07 -> 9.5e-06

    So the default path returns every edge PRESENT and WRONGLY SCALED (RAE 7e-15 on the then-
    squared scale, 8e-8 under the ratio-of-norms RAE in use since 2026-09-24; recall
    1.000, reconstruction 0.50), and at p=40 it loses ground-truth edges outright. Cost of
    faithful=True is 5-7x: 1.1-3.1s against 0.21-0.41s.

    CAVEAT: feasibility is regime-dependent. At blockiness=0.5 no restart clears the residual
    constraint at p=40 and this falls back to the default path, returning identical output for
    the extra runtime. Normalising the residual does not fix it -- the residual AT THE OPTIMUM
    reaches 4e-3 to 75, i.e. SLSQP leaves the feasible manifold chasing norm. Pass verbose=True
    to see when the fallback fires rather than having it hidden.
    """
    p, k = T_hard.shape
    if E_low is None:
        E_low = np.linalg.inv(np.eye(p) - A_low)

    idx = cluster_indices(T_hard)
    row_indices = np.concatenate(idx) if k else np.array([], dtype=int)
    col_indices = np.repeat(np.arange(k), [len(ix) for ix in idx])
    active = [c for c in range(k) if len(idx[c]) > 0]

    # E[:, pi(j)] per cluster: multiplying these by t_j avoids running T's zeros through a
    # dense matmul, which is what keeps a step at O(p^2) instead of O(p^2 k).
    E_cols = [torch.from_numpy(E_low[:, idx[j]].copy()) if len(idx[j]) else None for j in range(k)]
    seg = [torch.from_numpy(ix.copy()).long() for ix in idx]

    def _ET_columns(t_parts):
        """(p, k) matrix whose j-th column is E[:, pi(j)] @ t_j."""
        cols = []
        for j in range(k):
            if E_cols[j] is None:
                cols.append(torch.zeros(p, dtype=torch.float64))
            else:
                cols.append(E_cols[j] @ t_parts[j])
        return torch.stack(cols, dim=1)

    def _split(t):
        out, off = [], 0
        for j in range(k):
            n = len(idx[j])
            out.append(t[off:off + n])
            off += n
        return out

    def _A_high_from_K(t_np):
        """M[i,j] = (k_i . K^(i)_:,j)/||k_i||^2 = e_ij/e_ii, then A_high = I - colnorm(M^-1)."""
        A_high = np.zeros((k, k))
        if not active:
            return A_high
        ET = _ET_columns(_split(torch.from_numpy(t_np))).numpy()
        M = np.eye(len(active))
        for a, i in enumerate(active):
            Ki = ET[idx[i], :]
            ki = Ki[:, i]
            denom = ki @ ki
            if denom > 1e-30:
                M[a, :] = (ki @ Ki)[active] / denom
        M_inv = np.linalg.inv(M)
        d = np.diag(M_inv).copy()
        A_high[np.ix_(active, active)] = np.eye(len(active)) - M_inv / d[None, :]
        return A_high

    def objective_and_grad(t_np):
        t = torch.tensor(t_np, dtype=torch.float64, requires_grad=True)
        ET = _ET_columns(_split(t))
        f = torch.zeros((), dtype=torch.float64)
        for i in active:
            Ki = ET[seg[i], :]                       # (|pi(i)|, k)
            ki = Ki[:, i]                            # designated direction E_ii t_i
            denom = (ki * ki).sum().clamp(min=1e-12)
            coef = (ki @ Ki) / denom
            f = f + ((Ki - ki.unsqueeze(1) * coef.unsqueeze(0)) ** 2).sum()
        f.backward()
        return f.item(), t.grad.numpy().astype(np.float64)

    # ||E_ii t_i|| >= sigma_min(E_ii) > 0 (E_ii must be invertible for R^(i) to exist at all),
    # so the designated column cannot collapse; only the OFF-diagonal columns can, which is
    # the T-unfaithful case. The per-cluster unit-norm constraint is still needed to stop the
    # whole of T shrinking to zero.
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

    def _draw(rng_):
        t0 = rng_.randn(p)
        for j in range(k):
            lo, hi = offsets[j], offsets[j + 1]
            if hi > lo:
                nrm = np.linalg.norm(t0[lo:hi])
                if nrm > 1e-12:
                    t0[lo:hi] /= nrm
        return t0

    if faithful and active:
        # faithful=True minimises THE SAME residual as faithful=False, and differs only by
        # the T-faithfulness constraint
        #
        #     ||E_ij t_j|| / ||E_ij||_F  >=  delta     for every block with E_ij != 0.
        #
        # This replaces the pre-2026-09-25 device (maximise ||A_high||^2 subject to the
        # residual staying under rtol), which inverted objective and constraint. Two reasons
        # it was wrong. It barely bought the faithfulness it existed for -- measured over 64
        # fits it left 10 annihilated blocks against 11 for faithful=False, i.e. essentially
        # none -- and inverting the problem cost both accuracy and time: 25/64 fits reached
        # RAE < 1e-6 against 44/64 here, at 12.4s against 9.6s, and this wins 50 of the 64
        # paired comparisons against 14. On AOC Closure partitions the two tie on accuracy
        # (15/16) and this is 45x faster: 0.14s against 6.29s.
        #
        # WHY THE CONSTRAINT IS NEEDED AT ALL. The residual is not scale-invariant in t_j:
        # the j-th column of K^(i) is E_ij t_j, so shrinking that block's image shrinks its
        # own residual term for free. The unit-norm constraint only stops t_j vanishing
        # GLOBALLY -- a unit t_j lying in the null space of one particular E_ij still kills
        # that block at no cost. Minimising the residual alone is therefore ATTRACTED to
        # T-unfaithful solutions.
        #
        # WHERE IT BINDS. min ||E_ij t_j|| over unit t_j is sigma_min(E_ij), so on a block of
        # full column rank annihilation is impossible and the constraint is vacuous. Only
        # column-rank-deficient blocks -- forced whenever |pi(j)| > |pi(i)|, and 40-46% of
        # blocks at every p measured -- can annihilate, and only there does delta do work.
        #
        # CHOICE OF DELTA. With exact arithmetic delta -> 0 would be right; all that is
        # wanted is non-zero. In floating point the ratio cannot be told from rounding noise
        # below ~eps*cond(I-A_low), measured at 1.4e-14 to 6.4e-14 for p = 24..100, while the
        # ground-truth T itself sits at 3.9e-05 (p=100) to 4.2e-03 (p=24). That window is
        # 9-11 orders wide and its geometric centre is 7e-10 to 1.4e-08 at every size, so
        # 1e-9 is ~5 orders clear of the noise floor and ~4 below where truth lives.
        #
        # WHAT DELTA DOES NOT DO. It guarantees non-zero, not non-negligible. Above k ~ 15
        # the optimiser drives rank-deficient blocks onto the floor and rests there: the
        # constraint is satisfied and the block is still effectively annihilated. A delta
        # large enough to forbid that (~1e-5) would risk excluding the true solution. So
        # above k ~ 15 this path is WORSE than the old one (4/12 against 7/12) and a
        # solution resting on the floor should be read as a failure signal -- measured over
        # 76 fits, 1/28 of those reached RAE < 1e-6 against 34/48 that came to rest in the
        # interior.
        nE = np.zeros((k, k))
        for a_ in active:
            for b_ in active:
                nE[a_, b_] = np.linalg.norm(E_low[np.ix_(idx[a_], idx[b_])])
        # RELATIVE cutoff: block norms span ~1e16 at p=100 (min 1.05e-12, max 2.5e4), so an
        # absolute floor would sit inside the live range and call real blocks vacuous.
        cut = max(nE.max() * 1e-10, np.finfo(float).tiny)
        live = [(a_, b_) for a_ in active for b_ in active if nE[a_, b_] > cut]
        blk = [(offsets[b_], offsets[b_ + 1], E_low[np.ix_(idx[a_], idx[b_])], nE[a_, b_] ** 2)
               for (a_, b_) in live]
        d2 = delta ** 2

        def _faith_f(x):
            out = np.empty(len(blk))
            for c_, (lo, hi, Eb, n2) in enumerate(blk):
                v = Eb @ x[lo:hi]
                out[c_] = (v @ v) / n2 - d2
            return out

        def _faith_j(x):
            J = np.zeros((len(blk), p))
            for c_, (lo, hi, Eb, n2) in enumerate(blk):
                J[c_, lo:hi] = 2.0 * (Eb.T @ (Eb @ x[lo:hi])) / n2
            return J

        def _norm_f(x):
            return np.array([np.sum(x[offsets[c_]:offsets[c_ + 1]] ** 2) - 1.0
                             for c_ in range(k) if offsets[c_ + 1] > offsets[c_]])

        def _norm_j(x):
            live_c = [c_ for c_ in range(k) if offsets[c_ + 1] > offsets[c_]]
            J = np.zeros((len(live_c), p))
            for r_, c_ in enumerate(live_c):
                J[r_, offsets[c_]:offsets[c_ + 1]] = 2.0 * x[offsets[c_]:offsets[c_ + 1]]
            return J

        # Both constraint sets go in as ONE vector-valued dict each rather than one per block.
        # At k=20 that is 400 blocks; as separate dicts SLSQP spends its time in Python.
        cons_f = [{'type': 'eq', 'fun': _norm_f, 'jac': _norm_j}]
        if blk:
            cons_f.append({'type': 'ineq', 'fun': _faith_f, 'jac': _faith_j})

        # CONDITIONING. E = (I - A_low)^-1 amplifies with path length (|E|max grows 7.0 ->
        # 1.0e4 from p=24 to p=100), so the raw residual starts around 3e7 at p=100 against
        # O(1) constraints and SLSQP's line search dies outright with "positive directional
        # derivative", stalling with the norm constraints violated by 3.8. A constant factor
        # cannot move the optimum, only the conditioning.
        rng_f = np.random.RandomState(seed)
        starts_f = [_draw(rng_f) for _ in range(n_restarts)]
        r0 = objective_and_grad(starts_f[0])[0] if starts_f else 1.0
        rscale = 1.0 / max(abs(r0), 1.0)

        def _scaled(x):
            v, g = objective_and_grad(x)
            return v * rscale, g * rscale

        def _accept(x):
            """Feasible for BOTH constraint sets, and yields a finite A_high."""
            norm_err = max((abs(np.sum(x[offsets[c_]:offsets[c_ + 1]] ** 2) - 1.0)
                            for c_ in range(k) if offsets[c_ + 1] > offsets[c_]), default=0.0)
            if norm_err > 1e-6:
                return False
            if blk and _faith_f(x).min() < -((delta * 0.01) ** 2):
                return False
            try:
                return bool(np.all(np.isfinite(_A_high_from_K(x))))
            except np.linalg.LinAlgError:
                return False

        # The UNCONSTRAINED minimiser is also a candidate whenever it already satisfies the
        # constraint, which it usually does -- only rank-deficient blocks can annihilate, and
        # most runs never approach one. This is not redundant: adding the constraint changes
        # the search landscape, and SLSQP can get stuck on it at a point far worse than the
        # one it reaches without. Measured on cyclic p=60, the constrained search returned
        # residual 1.8e-01 (RAE 0.199, resting on the floor) while the unconstrained search
        # found residual 2.1e-11 (RAE 1.5e-06) at min ratio 3.8e-03 -- comfortably feasible,
        # simply never visited. Taking the best feasible candidate from both is strictly
        # better than either and costs one extra unconstrained solve per restart.
        best_x_f, best_val_f, n_feas = None, np.inf, 0
        for t0 in starts_f:
            for cons_ in (cons_f, [{'type': 'eq', 'fun': _norm_f, 'jac': _norm_j}]):
                try:
                    res_f = minimize(_scaled, t0, jac=True, method='SLSQP', constraints=cons_,
                                     tol=1e-14, options={"maxiter": max(maxiter, 1000)})
                except Exception:
                    continue
                if not _accept(res_f.x):
                    continue
                n_feas += 1
                val = objective_and_grad(res_f.x)[0]
                if val < best_val_f:
                    best_val_f, best_x_f = val, res_f.x
        if best_x_f is not None:
            T_f = np.zeros((p, k))
            T_f[row_indices, col_indices] = best_x_f
            ratios = np.sqrt(_faith_f(best_x_f) + d2) if blk else np.array([np.inf])
            log(f"fit_abstraction_K(faithful): {n_feas}/{2 * n_restarts} feasible candidates, "
                f"residual {best_val_f:.3e}, min ||E_ij t_j||/||E_ij|| {ratios.min():.3e}"
                + (" -- RESTING ON THE DELTA FLOOR, treat as unfaithful"
                   if ratios.min() < 1.5 * delta else ""), verbose=verbose)
            return _A_high_from_K(best_x_f), T_f
        # Infeasible: say so rather than hiding it. The constraint can be infeasible when a
        # cluster must satisfy several rank-deficient blocks at once.
        log(f"fit_abstraction_K(faithful): no feasible restart out of {n_restarts} "
            f"-- falling back to the unconstrained residual path", verbose=verbose)

    rng = np.random.RandomState(seed)
    candidates = []
    for _ in range(n_restarts):
        t0 = rng.randn(p)
        for j in range(k):
            lo, hi = offsets[j], offsets[j + 1]
            if hi > lo:
                nrm = np.linalg.norm(t0[lo:hi])
                if nrm > 1e-12:
                    t0[lo:hi] /= nrm
        res = minimize(objective_and_grad, t0, jac=True, method='SLSQP',
                       constraints=constraints, tol=1e-9, options={"maxiter": maxiter})
        try:
            A_cand = _A_high_from_K(res.x)
            faithfulness = np.linalg.norm(A_cand) if np.all(np.isfinite(A_cand)) else -1.0
        except np.linalg.LinAlgError:
            A_cand, faithfulness = np.zeros((k, k)), -1.0
        candidates.append((res.fun, res.x, A_cand, faithfulness))

    best_val = min(c[0] for c in candidates)
    tie_tol = best_val * 1e-3 + 1e-8
    tied = [c for c in candidates if c[0] <= best_val + tie_tol]
    _, best_t, A_high, _ = max(tied, key=lambda c: c[3])

    T = np.zeros((p, k))
    T[row_indices, col_indices] = best_t

    # Zero the entries that cannot mean anything. A_high[i, j] is only interpretable if the
    # low-level block A_low[pi_i, pi_j] actually contains an edge; anything else is solver
    # residue. This solver leaves such residue (measured on the Leontief graph: entries up to
    # 1.6e-10 on pairs with NO low-level edge between them) while the propagation solver
    # returns exact zeros there. That residue is not harmless: it is invisible to a
    # magnitude threshold whenever the whole A_high is small -- on a fit whose largest entry
    # was 5e-03, noise at 1e-05 sits two orders below the maximum and survives any sane
    # relative cut -- so downstream code drew it as real high-level structure, including
    # spurious NEGATIVE edges. Masking is exact and needs no tuned threshold.
    A_high = A_high * quotient_mask(A_low, T_hard)

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