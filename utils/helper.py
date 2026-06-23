import numpy as np
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp
import math
import torch
import torch.nn as nn
from typing import Literal
from scipy.optimize import minimize
import time

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

class DAGTopologyMask(nn.Module):
    def __init__(self, n_high):
        super().__init__()
        self.n_high = n_high
        # Learnable, continuous positions for each macro-node
        self.node_positions = nn.Parameter(torch.linspace(0, n_high, n_high))

    def dag_mask(self,temp=1.):
        diff = self.node_positions.unsqueeze(0) - self.node_positions.unsqueeze(1)
        hard_mask = (diff > 0).float()
        soft_mask = torch.sigmoid(diff/temp)
        dag_mask = soft_mask + (hard_mask - soft_mask).detach()
        return dag_mask, soft_mask

def learn_dag(A_low, T, num_epochs=1000, min_temp=0.01, max_temp=1.0, verbose=False, keep_history=False):
    if isinstance(A_low, np.ndarray):
        A_low = torch.from_numpy(A_low).float()
    if isinstance(T, np.ndarray):
        T = torch.from_numpy(T).float()

    def improved_abstraction_residual(dag_mask, A_low=A_low, T=T):
        n_low, n_high = T.shape
        I = torch.eye(n_low, device=T.device)

        S_list = []
        for j in range(n_high):
            t_j = T[:, j]
            A_jj = t_j.unsqueeze(1) * A_low * t_j.unsqueeze(0)
            s_j = torch.linalg.solve(I - A_jj, t_j)
            S_list.append(s_j)
        S = torch.stack(S_list, dim=1)  # (n_low, n_high)

        influence_matrix = A_low @ (T * S)  # (n_low, n_high)

        T_sq = T ** 2                                          # (n_low, n_high)
        t_i_norms = T_sq.sum(dim=0).clamp(min=1e-8)                # (n_high,) effective block sizes
        a_star = dag_mask * (T_sq.T @ influence_matrix) / t_i_norms.unsqueeze(1)  # (n_high, n_high)

        diff = influence_matrix.unsqueeze(0) - a_star.unsqueeze(1)
        weights = T_sq.T.unsqueeze(2)                               # (n_high, n_low, 1)
        residuals = (weights * diff ** 2).sum(dim=1) / t_i_norms.unsqueeze(1)  # (n_high, n_high)

        diag_mask = 1.0 - torch.eye(n_high, device=T.device)

        return (residuals * diag_mask).sum()
    
    n_low, n_high = T.shape
    learner = DAGTopologyMask(n_high)
    optimiser = torch.optim.Adam(learner.parameters(), lr=0.001)
    history = {'loss':[], 'dag_mask':[], 'soft_mask':[]}
    for epoch in range(num_epochs):
        temp = max(min_temp, max_temp * (min_temp**(epoch/num_epochs)))
        dag_mask, soft_mask = learner.dag_mask(temp)
        loss = improved_abstraction_residual(dag_mask)
        loss.backward()
        optimiser.step()
        if epoch % (num_epochs//10) == 0 or epoch == num_epochs - 1:
            log(f"Epoch {epoch+1}/{num_epochs}: Loss = {loss.item()}",verbose=verbose)
        if keep_history:
            history['loss'].append(loss.item())
            history['dag_mask'].append(dag_mask.cpu().detach().numpy())
            history['soft_mask'].append(soft_mask.cpu().detach().numpy())
    
    return learner.dag_mask()[0].cpu().detach().numpy(), history

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

def process_output(T_hard, A_low, E_low=None, dag_mask=None, n_iters=10, eps=1e-6, verbose=False):
    if E_low is None:
        E_low = np.linalg.inv(np.eye(A_low.shape[0]) - A_low)

    valid_solution = False
    best_loss = np.inf
    for _ in range(n_iters):
        A_high_pred, T_pred, loss = solve_massidda_cosine_new(A_low=A_low, T_hard=T_hard, dag_mask=dag_mask) #alternating_optimisation(A_low=A_low,T_hard=T_hard, dag_mask=dag_mask, n_iters=n_iters)
        if loss < eps:
            valid_solution = True
            best_A_high_pred, best_T_pred= A_high_pred.copy(), T_pred.copy()
            break
        elif loss < best_loss:
            best_loss = loss
            best_A_high_pred, best_T_pred= A_high_pred.copy(), T_pred.copy()
    
    if not valid_solution:
        log(f"Failed to find valid solution after {n_iters} attempts.",verbose=verbose)
    S_pred = get_S(A_low=A_low, T=best_T_pred)
    
    return S_pred, best_T_pred, best_A_high_pred

def solve_massidda_cosine_new(A_low, T_hard, eps=1e-12, dag_mask=None):
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
    A_high = A_high * dag_mask if dag_mask is not None else enforce_dag(A_high)
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

    return A_high, T_final, res.fun

def solve_massidda_cosine_unanchored(A_low, T_hard, eps=1e-12, dag_mask=None):
    """
    Solves the Massidda equations using a scale-free cosine similarity objective.
    All clusters are learnable and constrained to the unit hypersphere.
    """
    row, col = np.nonzero(T_hard)
    T_hard = T_hard.copy()
    T_hard[row, col] = 1
    p, k = T_hard.shape

    # 1. Precompute static cluster properties
    cluster_sizes = np.sum(T_hard, axis=0).astype(int)
    clusters = [np.nonzero(T_hard[:, i])[0] for i in range(k)]

    inv_I_A = [np.linalg.inv(np.eye(cluster_sizes[j]) - A_low[np.ix_(clusters[j], clusters[j])]) for j in range(k)]

    A_low_normalised = normalize_massidda_numpy(W=A_low,T_hard=T_hard)
    
    A_high = get_A_high(T_hard, A_low_normalised)
    A_high = A_high * dag_mask if dag_mask is not None else enforce_dag(A_high)
    A_high_mask = (A_high != 0) * 1.

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

    # 3. Objective and Gradient (All nodes are learnable)
    def objective_and_grad(t):
        V = M_sparse @ t
        U = t[row_to_node_i]

        V_dot_U = np.bincount(row_to_edge, weights=V * U, minlength=n_edges)
        V_dot_V = np.bincount(row_to_edge, weights=V * V, minlength=n_edges)
        V_dot_V_safe = V_dot_V + eps

        f_e = 1.0 - (V_dot_U**2) / V_dot_V_safe
        f = np.sum(f_e)

        V_dot_U_per_row = V_dot_U[row_to_edge]
        V_dot_V_per_row = V_dot_V_safe[row_to_edge]

        grad_U_per_row = -2.0 * (V_dot_U_per_row / V_dot_V_per_row) * V
        grad_t_from_U = np.bincount(row_to_node_i, weights=grad_U_per_row, minlength=p)

        grad_V_per_row = -2.0 * (V_dot_U_per_row / V_dot_V_per_row) * U + \
                          2.0 * ((V_dot_U_per_row**2) / (V_dot_V_per_row**2)) * V
        grad_t_from_V = M_sparse.T @ grad_V_per_row

        grad_t_full = grad_t_from_U + grad_t_from_V

        return f, grad_t_full

    def objective(x):
        return objective_and_grad(x)[0]

    def grad(x):
        return objective_and_grad(x)[1]

    # 4. Constraints: Unit norm for ALL clusters
    constraints = []
    
    def make_con(mask):
        return lambda x: np.sum(x[mask]**2) - 1.0

    def make_jac(mask):
        return lambda x: 2.0 * x * mask

    for c in range(k):
        c_mask_in_x = np.zeros(p, dtype=bool)
        c_mask_in_x[clusters[c]] = True
        constraints.append({
            'type': 'eq',
            'fun': make_con(c_mask_in_x),
            'jac': make_jac(c_mask_in_x)
        })

    # 5. Initialization: Random normal on the hypersphere
    np.random.seed(42) # Remove or change for production
    x0 = np.random.randn(p)
    for c in range(k):
        c_mask_in_x = np.zeros(p, dtype=bool)
        c_mask_in_x[clusters[c]] = True
        norm = np.linalg.norm(x0[c_mask_in_x])
        if norm > 0:
            x0[c_mask_in_x] /= norm

    # 6. Optimization
    start_time = time.time()
    res = minimize(objective, x0, jac=grad, constraints=constraints, method='SLSQP', tol=1e-9, options={'maxiter': 1000})
    runtime = time.time() - start_time
    print(f"res.fun={res.fun:.6e}  time={runtime:.4f}s  res.message={res.message}")

    # 7. Final Reconstructions
    t_final = res.x
    A_high = np.zeros((k, k))
    T_final = np.zeros((p, k))

    for c in range(k):
        T_final[clusters[c], c] = t_final[clusters[c]]

    A_high = get_A_high(T_final, A_low)

    return A_high, T_final