import numpy as np
import math
import torch
import torch.nn as nn

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
    n_low = A_low.shape[0]
    A_intra = T @ T.T * A_low
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
        ATS = A_low @ (T * S)                        

        A_high = T.T @ ATS                                
        t_norms = np.sum(T, axis=0)
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