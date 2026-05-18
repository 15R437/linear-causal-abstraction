import numpy as np
from scipy.stats import gennorm
from causabs.sampling import sample_linear_anm,sample_linear_abstraction,sample_linear_concretization

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

def T_concretisation_sampling(A_high,T,edge_density=0.3,seed=None,max_abs_weight=1.):
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
    
    for j in range(n_high):
        N_j = len(concrete_blocks[j])
        A_jj = random_dag(N_j,edge_density,max_abs_weight)
        A_low[np.ix_(concrete_blocks[j], concrete_blocks[j])] = A_jj
        t_j = T[concrete_blocks[j], j].reshape(N_j,1)
        s_j = np.linalg.solve(np.eye(N_j) - A_jj, t_j)
        for i in range(n_high):
            if i == j:
                continue
            t_i = T[concrete_blocks[i], i].reshape(len(concrete_blocks[i]),1)
            for l,_ in enumerate(concrete_blocks[i]):
                v = rng.standard_normal(size=N_j-1)
                v = np.concatenate([v,[1-np.sum(v)]]).reshape(-1,1)  #ensure sum to 1
                c = v/s_j
                try:
                    A_low[np.ix_(concrete_blocks[i],concrete_blocks[j])][l,:] = A_high[i,j]*t_i[l,0]*c.T
                except IndexError:
                    import pdb;pdb.set_trace()
    
    return A_low

def sample_low_level_graph(n_high,n_low,density,T_lower=(-2,1),T_upper=(-1,2),marg_prob=0,seed=None,max_attempts=10,shuffle=False,internal=False):
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        adj_structure = (random_dag(n_high,density, seed=seed) != 0).astype(float)
        weights = rng.normal(0, 1, size=adj_structure.shape)
        A_high_gt = adj_structure * weights
        try:
            T_gt,block_size = generate_random_T(n_low, n_high, T_lower, T_upper, marg_prob=marg_prob, seed=seed)
            A_low, S_gt = sample_linear_concretization(A_high_gt,T_gt,block_size,internal=internal,alpha=1.,style='uniform')
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

def sample_gennorm_lingam_population_standardized(A_gt_mat, n_samples, gennorm_param, seed=None):
    n = A_gt_mat.shape[0]
    I = np.eye(n)
    E = np.linalg.inv(I - A_gt_mat)
    exogenous_data = gennorm.rvs(gennorm_param, loc=0, scale=1, size=(n_samples, n))
    X = exogenous_data @ E
    return X

if __name__ == "__main__":
    seed = None
    n_high = 2  #number of high-level variables
    n_low = 6 #number of low-level variables
    edge_density = 0.9
    A_high = random_dag(n_high, edge_density, max_abs_weight=3., seed=seed)
    print("High-level adjacency matrix A_high:")
    print(A_high)

    #Define a block-diagonal T matrix
    T = generate_random_T(n_low, n_high, seed=seed)
    print("\nAbstraction matrix T:")
    print(T)

    block_sizes = np.sum((T!=0),axis=0).astype(int).tolist() + [0]
    A_low,_ = sample_linear_concretization(A_high,T,block_size=block_sizes) #T_concretisation_sampling(A_high, T, edge_density=edge_density, max_abs_weight=1.)
    print("\nSampled low-level adjacency matrix A_low:")
    print(A_low)