import numpy as np
from scipy.sparse.csgraph import connected_components
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


def alternating_optimization(E_low, T_hard, num_iterations=10, max_min_ratio=10):
    """
    Optimizes for E_high and T using the P and K matrix construction via 
    alternating optimization.
    
    Parameters:
    -----------
    E_low : ndarray of shape (p, p)
        Total treatment effects matrix of the low-level SCM.
    T_hard : ndarray of shape (p, k)
        Binary partition matrix
    num_iterations : int
        Number of alternating iterations to perform.
        
    Returns:
    --------
    E_high : ndarray of shape (k, k)
        Optimized high-level total treatment effects matrix.
    T : ndarray of shape (p, k)
        Optimized orthonormal block-structured abstraction matrix.
    """
    p,k = T_hard.shape
    I_p = np.eye(p)
    E_minus_I = E_low - I_p
    pi = [np.where(T_hard[:,j] != 0)[0] for j in range(k)]

    min_t = 1/(max_min_ratio * math.sqrt(p))

    # 1. Precompute M_i for each high-level variable i
    M = []
    for i in range(k):
        F_i = np.zeros((p, p))
        for idx in pi[i]:
            F_i[idx, idx] = 1.0
        
        # M_i = (I + F_i * (E - I))^-1 * F_i * E
        inv_term = np.linalg.inv(I_p + F_i @ E_minus_I)
        M_i = inv_term @ F_i @ E_low
        M.append(M_i)
        
    # 2. Construct the partition embedding matrices P_j and track column offsets
    P = []
    offsets = [0]
    for j in range(k):
        indices = pi[j]
        P_j = np.zeros((p, len(indices)))
        for idx, val in enumerate(indices):
            P_j[val, idx] = 1.0
        P.append(P_j)
        offsets.append(offsets[-1] + len(indices))
        
    # Initialize the joint vector v and apply block-normalization
    v = np.ones(p)
    for j in range(k):
        v_j = v[offsets[j]:offsets[j+1]]
        v[offsets[j]:offsets[j+1]] = v_j / np.linalg.norm(v_j)
        
    E_high = np.eye(k)
    
    # Alternating Optimization Loop
    for iteration in range(num_iterations):
        
        # --- Step 1: Optimize E_high given v (and thus T) ---
        T = np.zeros((p, k))
        for j in range(k):
            T[:, j] = P[j] @ v[offsets[j]:offsets[j+1]]
            
        for i in range(k):
            denom_vec = T[:, i]
            denominator = np.linalg.norm(denom_vec)**2
            if denominator < 1e-9:
                denominator = 1e-9  # Prevent division by zero
                
            for j in range(k):
                if i == j:
                    continue
                numerator = T[:, i].T @ M[i] @ T[:, j]
                E_high[i, j] = numerator / denominator
                
        # --- Step 2: Optimize T via v given E_high (Constrained Quadratic Form) ---
        # Construct the global homogeneous system matrix K of shape (k^2 * p, p)
        K = np.zeros((k * k * p, p))
        row_idx = 0
        for i in range(k):
            for j in range(k):
                if i == j:
                    block = M[i] @ P[i] - E_high[i, i] * P[i]
                    K[row_idx:row_idx+p, offsets[i]:offsets[i+1]] = block
                else:
                    K[row_idx:row_idx+p, offsets[j]:offsets[j+1]] = M[i] @ P[j]
                    K[row_idx:row_idx+p, offsets[i]:offsets[i+1]] = -E_high[i, j] * P[i]
                row_idx += p
        
        # Precompute the quadratic kernel matrix
        KtK = K.T @ K
        
        # Define the objective function for the full vector v
        def objective(v_vars):
            return v_vars.T @ KtK @ v_vars
        
        # Construct explicit equality constraints for every independent block
        constraints = []
        for j in range(k):
            start_idx = offsets[j]
            end_idx = offsets[j+1]
            # Use default arguments to capture the current bounds within the loop scope
            constraints.append({
                'type': 'eq',
                'fun': lambda v_vars, s=start_idx, e=end_idx: np.sum(v_vars[s:e]**2) - 1.0
            })
        
        # Run the constrained optimization using the current v as a true warm-start
        from scipy.optimize import minimize, Bounds
        bounds = Bounds(min_t,1.0)
        res = minimize(objective, v, method='SLSQP', constraints=constraints, tol=1e-9)
        
        # Update v with the constrained solution
        v = res.x
                
    # Final reconstruction of the structural matrix T
    T_final = np.zeros((p, k))
    for j in range(k):
        T_final[:, j] = P[j] @ v[offsets[j]:offsets[j+1]]
        
    return E_high, T_final

def alternating_optimisation_old(A_low, T_hard, dag_mask, n_iters=10, min_t= 1e-3):
    p,k = T_hard.shape
    row, col = np.nonzero(T_hard)
    T_hard[row, col] = 1
    cluster_sizes = np.sum(T_hard,axis=0)
    clusters = [np.nonzero(T_hard[:,i])[0] for i in range(k)]
    offset = [0]
    for N in cluster_sizes:
        offset.append(offset[-1] + int(N))

    dagify = lambda A, dag_mask: A * dag_mask if dag_mask is not None else enforce_dag(A)

    t_vec = np.sum(T_hard/np.linalg.norm(T_hard,axis=0),axis=1)
    
    for _ in range(n_iters):
        T = np.zeros((p,k))
        for j in range(k):
            T[clusters[j],j] = t_vec[clusters[j]]

        A_high = dagify(A = get_A_high(T, A_low), dag_mask=dag_mask)
        K = np.zeros((p*k,p))
        global_row = 0
        for i in range(k):
            N_i = int(cluster_sizes[i])
            for j in range(k):
                N_j = int(cluster_sizes[j])
                A_ij = A_low[np.ix_(clusters[i],clusters[j])]
                I_j = np.eye(N_j)
                A_jj = A_low[np.ix_(clusters[j],clusters[j])]
                M =  A_ij @ np.linalg.inv(I_j - A_jj)

                K[global_row: global_row + N_i, offset[j]: offset[j] + N_j] += M
                K[global_row: global_row + N_i, offset[i]: offset[i] + N_i] += -A_high[i,j] * np.eye(N_i)

                global_row += N_i
        KtK = K.T @ K

        def objective(t_vec):
            return t_vec.T @ KtK @ t_vec
        
        # Construct explicit equality constraints for every independent block
        constraints = []
        for j in range(k):
            constraints.append({
                'type': 'eq',
                'fun': lambda t_vec, idx=j : np.sum(t_vec[clusters[idx]]**2) - 1.0
            })
        
        # Run the constrained optimization using the current v as a true warm-start
        from scipy.optimize import minimize, Bounds
        #bounds = Bounds(min_t,1.0)
        res = minimize(objective, t_vec, method='SLSQP', constraints=constraints, tol=1e-9)
        t_vec = res.x
    
    T_final = np.zeros((p,k))
    for j in range(k):
        T_final[clusters[j],j] = t_vec[clusters[j]]

    A_high_final = dagify(A = get_A_high(T_final, A_low), dag_mask=dag_mask)
    
    return A_high_final, T_final


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

def alternating_optimisation(A_low, T_hard, dag_mask, n_iters=10, min_t=1e-3):
    row, col = np.nonzero(T_hard)
    T_hard[row, col] = 1
    p, k = T_hard.shape
    
    # 1. Precompute static cluster properties
    cluster_sizes = np.sum(T_hard, axis=0).astype(int)
    clusters = [np.nonzero(T_hard[:, i])[0] for i in range(k)]
    offset = [0]
    for N in cluster_sizes:
        offset.append(offset[-1] + N)
        
    #row_sum = np.sum(A_low,axis=1)
    #row_sum[row_sum == 0] = 1
    A_low_normalised = normalize_massidda_numpy(W=A_low,T_hard=T_hard)

    
    A_high = get_A_high(T_hard, A_low_normalised)
    A_high = A_high * dag_mask if dag_mask is not None else enforce_dag(A_high)
    A_high_mask = (A_high != 0) * 1.

    sink_indices_list = find_sinks_modulo_undirected_connected(A_high)
    sink_indices = sorted([l[0] for l in sink_indices_list])

    A_high_code = A_high_mask.copy()
    A_high_code[:,sink_indices] = -1
    A_high_code *= A_high_mask
    A_high_code = A_high_code.reshape(-1) # 0: no edge, 1: edge weight to be learnt and -1: fixed edge weight

    A_high_indices_to_learn = np.where(A_high_code == 1)[0]
    A_high_learn = np.ones_like(A_high_indices_to_learn)

    A_high_sink = np.zeros((k,k))
    A_high_sink[:,sink_indices] = np.ones((k,len(sink_indices)))
    A_high_sink *= A_high_mask

    def clst_to_flat_idx(high_idx):
        indices = [offset[high_idx] + i for i in range(cluster_sizes[high_idx])]
        return indices

    flat_sink_indices = [item for high_idx in sink_indices for item in clst_to_flat_idx(high_idx)]
    
    t_sink = np.zeros(p)
    t_sink[flat_sink_indices] = 1
    flat_indices_to_learn = np.where(t_sink == 0)[0]
    t_learn = np.ones(len(flat_indices_to_learn))

    if len(flat_indices_to_learn) == 0:
        # All high-level variables are sinks and A_high = 0
        return A_high, T_hard
    

    # Precompute indexing vectors for vectorizing T matrix construction
    T_rows = np.concatenate(clusters)
    T_cols = np.repeat(np.arange(k), cluster_sizes)
    
    # 2. Precompute static components of Matrix K
    K_static = np.zeros((p * k, p))
    inv_I_A = [np.linalg.inv(np.eye(cluster_sizes[j]) - A_low[np.ix_(clusters[j], clusters[j])]) for j in range(k)]
    
    # Track slices for dynamic updates to avoid nested loops later
    global_row = 0
    dynamic_slices = []
    
    for i in range(k):
        N_i = cluster_sizes[i]
        for j in range(k):
            N_j = cluster_sizes[j]
            A_ij = A_low[np.ix_(clusters[i], clusters[j])]
            M = A_ij @ inv_I_A[j]
            
            K_static[global_row : global_row + N_i, offset[j] : offset[j] + N_j] = M
            dynamic_slices.append((global_row, global_row + N_i, offset[i], offset[i] + N_i, i, j))
            global_row += N_i

    # 4. Alternating Optimisation Loop
    for _ in range(n_iters):

        # Analytical objective and Jacobian functions
        def objective(t_A_high_learn):
            t_learn = t_A_high_learn[:len(flat_indices_to_learn)]
            A_high_learn = t_A_high_learn[len(flat_indices_to_learn):]

            A_high = np.zeros(k**2)
            A_high[A_high_indices_to_learn] = A_high_learn
            A_high = A_high.reshape(k,k) + A_high_sink

            K = K_static.copy()
            for r_start, r_end, c_start, c_end, i, j in dynamic_slices:
                val = A_high[i, j]
                if val != 0.0:
                    # Direct diagonal subtraction bypassing np.eye allocation overhead
                    diag_indices = np.arange(r_end - r_start)
                    K[r_start + diag_indices, c_start + diag_indices] -= val
                    
            KtK = K.T @ K

            t_whole = t_sink.copy()
            t_whole[flat_indices_to_learn] = t_learn
            return t_whole.T @ KtK @ t_whole
        
        
        # SLSQP optimization with exact gradients
        t_A_high_learn = np.concatenate([t_learn,A_high_learn],axis=0)
        start_time = time.time()
        res = minimize(objective, t_A_high_learn, method='SLSQP', tol=1e-9)
        runtime = time.time() - start_time
        print(f"Time taken to minimse: {runtime:.4f}s")
        
        t_A_high_learn = res.x
        t_learn = t_A_high_learn[:len(flat_indices_to_learn)]
        A_high_learn = t_A_high_learn[len(flat_indices_to_learn):]
        A_high = np.zeros(k**2)
        A_high[A_high_indices_to_learn] = A_high_learn
        A_high = A_high.reshape(k,k) + A_high_sink

        t_whole = t_sink.copy()
        t_whole[flat_indices_to_learn] = t_learn
    
        #print(f"res.fun: {res.fun}")
        #print(f"res.status: {res.status}")
        #print(f"res.message: {res.message}")
    # Final reconstruction
    T_final = np.zeros((p, k))
    T_final[T_rows, T_cols] = t_whole
    
    return A_high, T_final

def process_output(T_hard, A_low, E_low=None, dag_mask=None, n_iters=1):
    if E_low is None:
        E_low = np.linalg.inv(np.eye(A_low.shape[0]) - A_low)

    A_high_pred, T_pred = alternating_optimisation(A_low=A_low,T_hard=T_hard, dag_mask=dag_mask, n_iters=n_iters)
    #A_high_pred = get_A_high(T=T_pred, A_low=A_low) #np.eye(model.n_high) - np.linalg.inv(E_high_pred)
    S_pred = get_S(A_low=A_low, T=T_pred)
    
    return S_pred, T_pred, A_high_pred

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