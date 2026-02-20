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

def generate_random_T(p,k,seed=None,T_range=(1.,1.)):
    """
    Generates a random abstraction matrix T  of shape (p x k) where each low-level variable
    maps to exactly one high-level variable. T is block diagonal.
    Parameters:
        p: int, number of low-level variables
        k: int, number of high-level variables
    Returns:
        T: np.ndarray, (p x k) abstraction matrix
    """
    assert p >= k, "Number of low-level variables must be greater than or equal to number of high-level variables."

    rng = np.random.default_rng(seed)
    T = np.zeros((p,k))
    partitions = rng.choice(np.arange(1,p),size=k-1,replace=False)
    partitions.sort()
    T[0:partitions[0],0] = rng.uniform(T_range[0],T_range[1],size=partitions[0])
    for i in range(1,k-1):
        T[partitions[i-1]:partitions[i],i] = rng.uniform(T_range[0],T_range[1],size=partitions[i]-partitions[i-1])
    T[partitions[-1]:p,k-1] = rng.uniform(T_range[0],T_range[1],size=p-partitions[-1])

    return T

def T_concretisation_sampling(A_high,T,edge_density=0.3,seed=None,max_abs_weight=1.):
    """
    Samples a low-level causal model (A_low) from a given high-level model (A_high)
    such that A_low is a T-abstraction of A_high.
    Parameters:
        A_high: np.ndarray, (k x k) adjacency matrix of the high-level causal model
        T: np.ndarray, the (p x k) abstraction matrix mapping low-level to high-level variables. Should be block diagonal.
    Returns:
        A_low: np.ndarray, (p x p) adjacency matrix of a low-level causal model
    """
    p,k = T.shape
    assert (A_high.shape == (k,k)), f"Expected A_high shape {(k,k)}, got {A_high.shape}."

    rng = np.random.default_rng(seed)
    A_low = np.zeros((p,p))
    concrete_blocks = [list(np.where(T[:,j] != 0)[0]) for j in range(k)]
    
    for j in range(k):
        N_j = len(concrete_blocks[j])
        A_jj = random_dag(N_j,edge_density,max_abs_weight)
        A_low[np.ix_(concrete_blocks[j], concrete_blocks[j])] = A_jj
        t_j = T[concrete_blocks[j], j].reshape(N_j,1)
        s_j = np.linalg.solve(np.eye(N_j) - A_jj, t_j)
        for i in range(k):
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

def sample_gennorm_lingam_population_standardized(A_gt_mat, n_samples, gennorm_param, seed=None):
    n = A_gt_mat.shape[0]
    I = np.eye(n)
    E = np.linalg.inv(I - A_gt_mat)
    exogenous_data = gennorm.rvs(gennorm_param, loc=0, scale=1, size=(n_samples, n))
    X = exogenous_data @ E.T
    return X

if __name__ == "__main__":
    seed = None
    k = 2  #number of high-level variables
    p = 6 #number of low-level variables
    edge_density = 0.9
    A_high = random_dag(k, edge_density, max_abs_weight=3., seed=seed)
    print("High-level adjacency matrix A_high:")
    print(A_high)

    #Define a block-diagonal T matrix
    T = generate_random_T(p, k, seed=seed)
    print("\nAbstraction matrix T:")
    print(T)

    block_sizes = np.sum((T!=0),axis=0).astype(int).tolist() + [0]
    A_low,_ = sample_linear_concretization(A_high,T,block_size=block_sizes) #T_concretisation_sampling(A_high, T, edge_density=edge_density, max_abs_weight=1.)
    print("\nSampled low-level adjacency matrix A_low:")
    print(A_low)