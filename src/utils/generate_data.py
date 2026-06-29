import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))
import numpy as np
from collections import deque
from scipy.stats import gennorm
from causabs.sampling import sample_linear_anm,sample_linear_abstraction,sample_linear_concretization
from src.utils.helper import get_S, get_A_high
import numpy as np

def generate_generalized_benchmark(n_low=90, n_high=3, blockiness=1.0, shuffle=True, seed=None, **kwds):
    """
    Generates an n-dimensional, singularity-free layered graph 
    designed to form a perfectly convex optimization basin for hail_mary.
    
    Params:
        n_low (int): Total low-level nodes (must be divisible by n_high).
        n_high (int): Number of high-level clusters/blocks.
        blockiness (float): 1.0 for a perfect, local-minima-free cascade.
                            Lower values introduce structural noise.
    """
    assert n_low % n_high == 0, f"n_low ({n_low}) must be perfectly divisible by n_high ({n_high})."
    nodes_per_block = n_low // n_high
    rng = np.random.default_rng(seed)

    # 1. Construct Ground Truth T Matrix
    T_gt = np.zeros((n_low, n_high))
    for b in range(n_high):
        start_idx = b * nodes_per_block
        end_idx = start_idx + nodes_per_block
        T_gt[start_idx:end_idx, b] = 1.0

    # 2. Build the Ideal Cascade Matrix
    A_ideal = np.zeros((n_low, n_low))
    for b in range(n_high - 1):
        # Source block boundaries
        src_start = b * nodes_per_block
        src_end = src_start + nodes_per_block
        
        # Immediate downstream target block boundaries
        tgt_start = (b + 1) * nodes_per_block
        tgt_end = tgt_start + nodes_per_block
        
        # Fully connect every node in block b to every node in block b+1
        A_ideal[src_start:src_end, tgt_start:tgt_end] = 1.0

    # 3. Create Background Noise Matrix (density = 0.1)
    A_noise = np.random.binomial(1, p=0.1, size=(n_low, n_low))
    np.fill_diagonal(A_noise, 0)
    
    # 4. Blend Ideal Structure with Noise using the Knob
    mask = np.random.rand(n_low, n_low) < blockiness
    A_low = np.where(mask, A_ideal, A_noise)
    
    # At perfect blockiness, wipe out all intra-block and backward edges to guarantee 0 loss
    if blockiness == 1.0:
        A_low = A_ideal
        """
        for b in range(n_high):
            start = b * nodes_per_block
            end = start + nodes_per_block
            # Wipe out internal block connections
            A_low[start:end, start:end] = 0
            # Wipe out backward connections to previous blocks
            if start > 0:
                A_low[start:end, :start] = 0
        """
    if shuffle:
        perm = rng.permutation(n_low)
        A_low = A_low[perm][:,perm]
        T_gt = T_gt[perm][:,:]

    return A_low, T_gt



def generate_easy_benchmark(n_low=60, blockiness=1.0, cluster_balance=0.5, shuffle=True, seed=None):
    """
    Generates a low-dimensional, singularity-free graph engineered 
    to have a convex loss landscape for the hail_mary algorithm.
    
    Params:
        n_low (int): Total low-level nodes (must be even for equal clusters).
        blockiness (float): Value from 0.0 to 1.0. 
                            1.0 = Perfect block structure (all nodes in C0 connected to all nodes in C1).
                            0.0 = Each node in C0 connected to just one node in C1
    """
    rng = np.random.default_rng(seed)
    cluster_0_size = max(min(int(n_low * cluster_balance),n_low-1),1)
    cluster_1_size = n_low - cluster_0_size
    
    # 1. Construct Ground Truth T (Hard Assignments)
    T_gt = np.zeros((n_low, 2))
    T_gt[:cluster_0_size, 0] = 1.0 
    T_gt[cluster_0_size:, 1] = 1.0  

    A_low = np.zeros((n_low,n_low))

    # ensure at least one connection from each node in C0 to satisfy the abstraction residual requirement
    targets = rng.integers(cluster_0_size, n_low, size=cluster_0_size)
    A_low[np.arange(cluster_0_size), targets] = 1.0
    
    if blockiness > 0:
        random_mask = rng.random((cluster_0_size, cluster_1_size)) < blockiness
        A_low[:cluster_0_size, cluster_0_size:] = np.maximum(
            A_low[:cluster_0_size, cluster_0_size:], 
            random_mask.astype(float)
        )
    
    #make sure rows of A_low have identical sums (since T_gt is binary)
    row_sum = A_low[:cluster_0_size,:].sum(axis=1)
    target_sum = row_sum.max()
    A_low[:cluster_0_size,:] = (target_sum[None,None] / row_sum[:,None] ) * A_low[:cluster_0_size,:]

    if shuffle:
        perm = rng.permutation(n_low)
        A_low = A_low[perm][:,perm]
        T_gt = T_gt[perm][:,:]

        T_gt = T_gt[:][:,rng.permutation(2)]
    
    S_gt = get_S(T=T_gt, A_low=A_low)
    A_high_gt = get_A_high(T=T_gt, A_low=A_low)
    return A_low, T_gt, S_gt, A_high_gt
    


def random_dag(num_nodes,edge_density,weight_mean=0,weight_std=1,seed=None):
    """
    Generates a random DAG with the specified number of nodes and edge density.
    Parameters:
        num_nodes: int, number of nodes in the DAG
        edge_density: float, probability of an edge between any two nodes
    Returns:
        adj_matrix: np.ndarray, upper-triangular adjacency matrix of the generated DAG
    """
    #input validation
    assert 0 <= edge_density <= 1, "Edge density must be between 0 and 1."
    assert (num_nodes > 0 and type(num_nodes) == int), "Number of nodes must be a positive integer."

    rng = np.random.default_rng(seed)

    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if rng.random() < edge_density:
                adj_matrix[i,j] = rng.normal(weight_mean, weight_std)
    return adj_matrix

def sample_disjoint_uniform(lower_bounds, upper_bounds, size, seed=None):
    """
    Samples from a union of intervals.
    Example: low_bounds=[-2.0, 0.5], high_bounds=[-0.5, 2.0]
    results in [-2.0, -0.5] U [0.5, 2.0]
    """

    assert len(lower_bounds) == len(upper_bounds), "lower_bounds must have the same length as upper_bounds"
    rng = np.random.default_rng(seed)
    num_intervals = len(lower_bounds)
    interval_choices = np.random.randint(0, num_intervals, size=size)

    unit_samples = np.random.uniform(0, 1, size=size)
    

    lows = np.array(lower_bounds)[interval_choices]
    highs = np.array(upper_bounds)[interval_choices]
    
    return lows + (highs - lows) * unit_samples

def generate_random_T(n_low,n_high,lower_bounds=(1.,),upper_bounds=(1.,),marg_prob=0,seed=None):
    """
    Generates a random abstraction matrix T  of shape (n_low x n_high) where each low-level variable
    maps to exactly one high-level variable. T is block diagonal.
    Parameters:
        n_low: int, number of low-level variables
        n_high: int, number of high-level variables
    Returns:
        T: np.ndarray, (n_low x n_high) abstraction matrix
        block_sizes (list)
    """
    assert n_low >= n_high, "Number of low-level variables must be greater than or equal to number of high-level variables."

    rng = np.random.default_rng(seed)
    T = np.zeros((n_low,n_high))
    partitions = rng.choice(np.arange(1,n_low),size=n_high-1,replace=False)
    partitions.sort()
    block_sizes = [partitions[0]]
    #T[0:partitions[0],0] = rng.uniform(T_range[0],T_range[1],size=partitions[0]) if rng.uniform() > marg_prob else 0
    T[0:partitions[0],0] = sample_disjoint_uniform(lower_bounds,upper_bounds,size=partitions[0]) * (rng.uniform(size=partitions[0]) > marg_prob)
    for i in range(1,n_high-1):
        block_sizes.append(partitions[i]-partitions[i-1])
        T[partitions[i-1]:partitions[i],i] = sample_disjoint_uniform(lower_bounds,upper_bounds,size=partitions[i]-partitions[i-1]) * (rng.uniform(size=partitions[i]-partitions[i-1]) > marg_prob)
        #T[partitions[i-1]:partitions[i],i] = rng.uniform(T_range[0],T_range[1],size=partitions[i]-partitions[i-1]) if rng.uniform() > marg_prob else 0
    
    block_sizes.append(n_low-partitions[-1])
    T[partitions[-1]:n_low,n_high-1] = sample_disjoint_uniform(lower_bounds,upper_bounds,size=n_low-partitions[-1]) * (rng.uniform(size=n_low-partitions[-1]) > marg_prob)
    #T[partitions[-1]:n_low,n_high-1] = rng.uniform(T_range[0],T_range[1],size=n_low-partitions[-1])
    block_sizes.append(0)

    return T,block_sizes

def sample_c_vector_old(s_j, heterogeneity, rng):
    N = len(s_j)
    s_j = s_j.reshape(-1, 1)
    c_base = s_j / (s_j.T @ s_j)

    if heterogeneity == 0 or N == 1:
        return c_base

    eta = rng.standard_normal((N, 1))
    eta = eta - s_j * ((s_j.T @ eta) / (s_j.T @ s_j))

    eta_norm = np.linalg.norm(eta)

    if eta_norm > 1e-8:
        eta = eta / eta_norm

    c = c_base + heterogeneity * eta

    return c

def sample_c_vector(s_j, heterogeneity, rng, blockiness=1.0):
    """
    Samples a vector c such that c.T @ s_j = 1.
    
    Parameters:
        s_j: np.ndarray, the local s vector for the cluster
        heterogeneity: float, scales the orthogonal noise variance
        rng: np.random.Generator
        connection_probability: float between 0 and 1. Lower means more zeros in c.
    """
    N = len(s_j)
    s_j = s_j.reshape(-1, 1)
    
    # Identify coordinates where s_j is non-zero to safely satisfy the constraint
    valid_indices = np.where(s_j.flatten() != 0)[0]
    if len(valid_indices) == 0:
        return np.zeros((N, 1))

    # 1. Determine which nodes will be active (have edges)
    # Force at least one valid node to be active so we can satisfy c.T @ s_j = 1
    active_indices = [rng.choice(valid_indices)]
    for idx in range(N):
        if idx not in active_indices and rng.random() < blockiness:
            active_indices.append(idx)
            
    active_indices = np.array(sorted(active_indices))
    
    # 2. Slice s_j down to just the active sub-vector
    s_active = s_j[active_indices]
    N_active = len(active_indices)
    
    # 3. Apply your exact original logic to the active subset
    c_active_base = s_active / (s_active.T @ s_active)

    if heterogeneity == 0 or N_active == 1:
        c_active = c_active_base
    else:
        # Generate noise orthogonal ONLY to the active sub-vector
        eta_active = rng.standard_normal((N_active, 1))
        eta_active = eta_active - s_active * ((s_active.T @ eta_active) / (s_active.T @ s_active))

        eta_norm = np.linalg.norm(eta_active)
        if eta_norm > 1e-8:
            eta_active = eta_active / eta_norm

        c_active = c_active_base + heterogeneity * eta_active

    # 4. Construct the final full-size c vector
    c = np.zeros((N, 1))
    c[active_indices] = c_active
    
    return c

def T_concretisation_sampling(A_high,T,internal_density=0,seed=None,heterogeneity=1.,blockiness=1.):
    """
    Samples a low-level causal model (A_low) from a given high-level model (A_high)
    such that A_low is a T-abstraction of A_high.
    Parameters:
        A_high: np.ndarray, (n_high x n_high) adjacency matrix of the high-level causal model
        T: np.ndarray, the (n_low x n_high) abstraction matrix mapping low-level to high-level variables. Should be block diagonal.
    Returns:
        A_low: np.ndarray, (n_low x n_low) adjacency matrix of a low-level causal model
    """
    n_low,n_high = T.shape
    assert (A_high.shape == (n_high,n_high)), f"Expected A_high shape {(n_high,n_high)}, got {A_high.shape}."

    rng = np.random.default_rng(seed)
    A_low = np.zeros((n_low,n_low))
    concrete_blocks = [list(np.where(T[:,j] != 0)[0]) for j in range(n_high)]
    
    S_gt = np.zeros((n_low,n_high))
    for j in range(n_high):
        N_j = len(concrete_blocks[j])
        A_jj = random_dag(N_j,internal_density)
        A_low[np.ix_(concrete_blocks[j], concrete_blocks[j])] = A_jj
        t_j = T[concrete_blocks[j], j].reshape(N_j,1)
        s_j = np.linalg.solve(np.eye(N_j) - A_jj, t_j)
        
        S_gt[concrete_blocks[j],j] = s_j.flatten()
        for i in range(n_high):
            if i == j:
                continue
            t_i = T[concrete_blocks[i], i].reshape(len(concrete_blocks[i]),1)
            for l, row_idx in enumerate(concrete_blocks[i]):
                #v = rng.standard_normal(size=N_j-1)
                #v = np.concatenate([v,[1-np.sum(v)]]).reshape(-1,1)  #ensure sum to 1
                c = sample_c_vector(s_j,heterogeneity=heterogeneity,blockiness=blockiness,rng=rng) #c = v/s_j
                A_low[row_idx,concrete_blocks[j]] = A_high[i,j]*t_i[l,0]*c.flatten()
    
    return A_low, S_gt

def sample_low_level_graph(n_high,n_low,density,T_lower=(-2,1),T_upper=(-1,2),marg_prob=0,seed=None,max_attempts=10,shuffle=False, blockiness=1, internal_density=0., heterogeneity=1., **kwds):
    rng = np.random.default_rng(seed)
    internal_density = 0. if internal_density is None else internal_density
    for _ in range(max_attempts):
        attempt_seed = int(rng.integers(0, 2**32 - 1)) if seed is not None else None

        # Samples weights strictly between 0.3 and 0.8 (and randomly flips the sign)
        adj_structure = (random_dag(n_high,density, seed=attempt_seed) != 0).astype(float)
        signs = rng.choice([-1, 1], size=adj_structure.shape)
        absolute_weights = rng.uniform(0.5, 2.0, size=adj_structure.shape)
        A_high_gt = adj_structure * absolute_weights * signs
        
        try:
            T_gt,block_size = generate_random_T(n_low, n_high, T_lower, T_upper, marg_prob=marg_prob, seed=attempt_seed)
            A_low, S_gt = T_concretisation_sampling(A_high=A_high_gt,T=T_gt,internal_density=internal_density,blockiness=blockiness,heterogeneity=heterogeneity,seed=seed)#sample_linear_concretization(A_high_gt,T_gt,block_size,internal=internal,alpha=1.,style='uniform')
            assert np.any(A_low != 0) #ensure we don't return a trivial graph with no edges
            if shuffle:
                perm = rng.permutation(n_low)
                A_low = A_low[perm][:,perm]
                T_gt = T_gt[perm][:,:]
                S_gt = S_gt[perm][:,:]
        
            return A_low,T_gt,S_gt,A_high_gt
        except:
            pass
    
    raise RuntimeError(f"Could not find a stable model after {max_attempts} attempts.")

def snr_adjustment(
    unadjusted_A,
    n_samples=int(1e5),
    gennorm_param=1.8,
    SNR=1,
    target_var=1,
    seed=None
):

    p = unadjusted_A.shape[0]
    adjusted_A_gt_mat = np.zeros_like(unadjusted_A)

    rng = np.random.default_rng(seed)

    adjusted_eps = np.zeros((n_samples, p))

    eps_unscaled = gennorm.rvs(
        gennorm_param, loc=0.0, scale=1.0,
        size=(n_samples, p),
        random_state=rng
    )

    for i in range(p):
        parents = np.where(np.abs(unadjusted_A[:, i]) > 1e-9)[0]

        if len(parents) == 0:
            # Root node
            adjusted_eps[:,i] = np.sqrt(target_var) * eps_unscaled[:,i]

        col = unadjusted_A[parents,i]
        col_norm = np.linalg.norm(col)

        if col_norm > 0:
            adjusted_A_gt_mat[parents, i] = (
                np.sqrt(SNR / (1 + SNR)) * col / col_norm
            )

        adjusted_eps[:,i] = np.sqrt(target_var / (1 + SNR)) * eps_unscaled[:,i]


    return adjusted_A_gt_mat, adjusted_eps

def sample_gennorm_lingam_population_standardized(A_low, n_samples, gennorm_param, snr=1, seed=None):
    p = A_low.shape[0]
    A, U = snr_adjustment(A_low, gennorm_param=gennorm_param, SNR=snr, n_samples=n_samples)
    
    E = np.linalg.inv(np.eye(p) - A)
    #exogenous_data = gennorm.rvs(gennorm_param, loc=0, scale=1, size=(n_samples, p))
    X = U @ E
    return X


def sample_by_noise_adjustment_any_order(A_low, n_samples, SNR=1, gennorm_param=1.8, target_var=1.0, seed=None):
    """
    Generates data from A_low without altering its edge weights, 
    regardless of the initial node ordering.
    """
    p = A_low.shape[0]
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, p))
    
    # 1. Compute in-degrees (number of parents for each node)
    # entry A_low[u, v] implies u -> v (u is a parent of v)
    in_degree = np.sum(np.abs(A_low) > 1e-9, axis=0)
    
    # 2. Initialize queue with root nodes (in-degree == 0)
    queue = deque([i for i in range(p) if in_degree[i] == 0])
    topo_order = []
    
    # 3. Perform Kahn's algorithm to find a valid topological sequence
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        
        # Find all children of the current node
        children = np.where(np.abs(A_low[node, :]) > 1e-9)[0]
        for child in children:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
                
    if len(topo_order) != p:
        raise ValueError("The provided A_low contains cycles. Topological sorting is impossible.")
        
    # 4. Generate the unscaled noise array
    eps_unscaled = gennorm.rvs(
        gennorm_param, loc=0.0, scale=1.0, 
        size=(n_samples, p), random_state=rng
    )
    
    # 5. Process nodes sequentially using the verified topological order
    for i in topo_order:
        parents = np.where(np.abs(A_low[:, i]) > 1e-9)[0]
        
        if len(parents) == 0:
            # Root nodes use the default target variance
            X[:, i] = np.sqrt(target_var) * eps_unscaled[:, i]
        else:
            # Parents are guaranteed to be fully computed already due to topo_order
            signal = X[:, parents] @ A_low[parents, i]
            var_signal = np.var(signal)
            
            if var_signal > 1e-9:
                var_noise = var_signal / SNR
            else:
                var_noise = target_var
                
            X[:, i] = signal + np.sqrt(var_noise) * eps_unscaled[:, i]
            
    return X

if __name__ == "__main__":
    seed = None
    n_high = 2  #number of high-level variables
    n_low = 6 #number of low-level variables
    edge_density = 0.9
    A_high = random_dag(n_high, edge_density, seed=seed)
    print("High-level adjacency matrix A_high:")
    print(A_high)

    #Define a block-diagonal T matrix
    T,_ = generate_random_T(n_low, n_high, seed=seed)
    print("\nAbstraction matrix T:")
    print(T)

    #block_sizes = np.sum((T!=0),axis=0).astype(int).tolist() + [0]
    A_low,_ = T_concretisation_sampling(A_high, T, internal_density=edge_density) #sample_linear_concretization(A_high,T,block_size=block_sizes) #
    print("\nSampled low-level adjacency matrix A_low:")
    print(A_low)