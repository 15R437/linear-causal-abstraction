import numpy as np
import copy
from sklearn.linear_model import LinearRegression
from causal_experiments.losses import reconstruction_rmse

def merge(E,i,j):
    """
    Takes the total treatment effect matrix E (p x p), of a causal graph with p nodes
    and returns a matrix E_merged (p-1) x (p-1) corresponding to the graph with nodes
    i and j merged into one node. If i < j then the i'th row of E_merged corresponds 
    to the effects of the other nodes on the merged node.

    Parameters:
        E (numpy.ndarray):  The ground truth total treatment effect matrix (lower triangular).
                            E[i,j] is the effect of j on i so that under causal ordering, X=EU
                            where X and U are column vectors.
        i (int): A node to be merged.
        j (int): A node to be merged.

    Returns:
        numpy.ndarray: The total treatment effect matrix for the graph with nodes i and j merged.
    """
    if i == j:
        raise ValueError(f"Cannot merge node {i} with itself.")
    
    p = E.shape[0]
    E_merged = np.zeros((p-1,p-1))
    if j < i:
        i,j = j,i
    for k in range(p):
        if k == i:
            pass
        elif k < j:
            E_merged[i,k] = E[i,k] + E[j,k]
            E_merged[k,i] = (E[k,i] + E[k,j])/2
        elif k > j:
            E_merged[i,k-1] = E[i,k] + E[j,k]
            E_merged[k-1,i] = (E[k,i] + E[k,j])/2
    
    #set the remaining entries for E_merged to the corresponding entry in E
    merged_zeros_idx = [r for r in range(p-1) if r != i]
    unmerged_idx = [r for r in range(p) if r != i and r != j]
    E_merged[np.ix_(merged_zeros_idx,merged_zeros_idx)] = E[np.ix_(unmerged_idx,unmerged_idx)]

    np.fill_diagonal(E_merged,1)
    
    return E_merged

def unmerge(E_merged,i):
    """
    Takes a total treatment effect matrix E_merged and splits node i into nodes i and j.
    Assumes that j comes immediately after i in the causal ordering and that they have
    no edges between them. Incoming weights are divided equally between i and j whilst
    outgoing weights are the same as in E_merged.

    Parameters:
        E_merged (numpy.ndarray): The total treatment effect matrix for an abstracted graph.
        i (int): A node to be split in two.

    Returns:
        numpy.ndarray: The total treatment effect matrix for the graph with node i split.
    """
    p = E_merged.shape[0]+1
    E_unmerged = np.zeros((p,p))
    j = i+1
    for k in range(p-1):
        if k < j:
            E_unmerged[i,k] = E_merged[i,k]/2
            E_unmerged[j,k] = E_merged[i,k]/2

            E_unmerged[k,i] = E_merged[k,i]
            E_unmerged[k,j] = E_merged[k,i]
        else:
            E_unmerged[i,k+1] = E_merged[i,k]/2
            E_unmerged[j,k+1] = E_merged[i,k]/2

            E_unmerged[k+1,i] = E_merged[k,i]
            E_unmerged[k+1,j] = E_merged[k,i]

    #set the remaining entries for E_unmerged to the corresponding entry in E_merged
    merged_idx = [r for r in range(p-1) if r != i]
    unmerged_idx = [r for r in range(p) if r != i and r != j]
    E_unmerged[np.ix_(unmerged_idx,unmerged_idx)] = E_merged[np.ix_(merged_idx,merged_idx)]

    return E_unmerged

def merge_clusters(clusters,i,j):
    k = len(clusters)
    new_clusters = []
    for t in range(k):
        if t == i:
            new_clusters.append(clusters[i]+clusters[j])
        elif t == j:
            continue
        else:
            new_clusters.append(clusters[t])

    return new_clusters

def cluster_to_matrix(clusters):
    k = len(clusters)
    p = sum(len(clst) for clst in clusters)
    T = np.zeros(shape=(p,k))
    for idx,row in enumerate(clusters):
        T[np.array(row),idx] = 1
    
    return T

def matrix_to_clusters(T):
    p,k = T.shape
    clusters = [tuple(np.where(T[:,j]!=0)[0].tolist()) for j in range(k)]
    return clusters
    
def new_merge(clusters,i,j,E,X):
    """
    A new implementation of the merge function that uses data X to perform Linear Regression
    between the new high-level cluster obtained by merging clusters i and j and every other
    cluster related to i and j (given by the total treatment effect matrix E). The high-level 
    dependent variable (x_bar) in the regression is the sum of the relevant low-level variables
    in the cluster whilst the high-level independent variable (y_bar) is the inverse variance 
    weighted-sum of the relevant low-level variables.

    Parameters:
        clusters (list): list of length k consisting of tuples of low-level variable indices {0,...,p-1}.
        i (int): The index of the cluster to be merged.
        j (int): The index of the cluster to be merged.
        E (numpy.ndarray): (k x k) total treatment effect matrix over clusters.
        X (numpy.ndarray): (N x p) dataset matrix over low-level variables.
    
    Returns:
        E_merged (numpy.ndarray): (k-1) x (k-1) total treatment effect matrix over the new clusters after merging.
    """
    if i == j:
        raise ValueError(f"Cannot merge node {i} with itself.")
    if E.shape[0] != len(clusters):
        raise ValueError(f"E must be of shape len(clusters) x len(clusters) ({len(clusters)} x {len(clusters)})")
    
    k = E.shape[0]
    E_merged = np.eye(k-1,k-1)
    if j < i:
        i,j = j,i
    
    new_clusters = merge_clusters(clusters,i,j)

    #populating the lower-triangular entries of E_merged
    # Note:                 new_cluster[i] = cluster[i]+cluster[j];
    # for l >= j,           new_cluster[l] = cluster[l+1] and 
    # for l != i and l < j, new_cluster[l] = cluster[l].
    for r in range(k-1):
        for c in range(r):
            if r == i:
                NEW_EDGE = bool(E[i,c]!=0) or bool(E[j,c]!=0)
            elif c == i:
                NEW_EDGE = bool(E[r+int(r>=j),i]) or bool(E[r+int(r>=j),j])
            else:
                E_merged[r,c] = E[r+int(r>=j),c+int(c>=j)]
                NEW_EDGE = False

            if NEW_EDGE:                
                x_bar = np.sum(X[:,np.array(new_clusters[c])],axis=1)
                variances = np.var(X[:,np.array(new_clusters[r])],axis=0)
                weights = 1/variances
                y_bar = np.sum(X[:,np.array(new_clusters[r])]*weights,axis=1)/np.sum(weights)
                #y_bar = np.sum(X[:,np.array(new_clusters[r])],axis=1)
                linear_model = LinearRegression(fit_intercept=False)
                linear_model.fit(x_bar.reshape(-1,1),y_bar)
                E_merged[r,c] = linear_model.coef_.item()

    return new_clusters,E_merged  

def new_unmerge(clusters,E_merged):
    """
    A new implementation of the unmerge function that requires specifying the clusters.
    The clusters are split into their constituent low-level variables with the assumption
    that there are no intra-cluster causal relations. For a given cluster, its incoming
    and outgoing weights (given by E_merged) are, respectively, divided evenly and copied
    across the low-level variables.

    Parameters:
        clusters (list): list of length k consisting of tuples of low-level variable indices {0,...,p-1}.
        E_merged (numpy.ndarray): k x k total treatment effect matrix over clusters.
    
    Return:
        E_hat (numpy.ndarray): p x p predicted total treatment effect matrix over low-level variables.
    """
    if E_merged.shape[0] != len(clusters):
        raise ValueError(f"E_merged must be of shape len(clusters) x len(clusters) ({len(clusters)} x {len(clusters)}). Got E_merged of shape {E_merged.shape}.")
    
    p = sum(len(clst) for clst in clusters)
    k = len(clusters)
    E_hat = np.eye(p,p)

    for r in range(k):
        for c in range(r):
            #split clusters[c] (source node) and clusters[r] (sink node)
            E_hat[np.ix_(np.array(clusters[r]),np.array(clusters[c]))] = E_merged[r,c] #/len(clusters[r])
    
    return E_hat

def is_acyclic(E,eps=1e-10):
    k = E.shape[0]
    I = np.eye(k)
    A = I-np.linalg.pinv(E)
    A_pow_k = np.linalg.matrix_power(A,k)

    return np.linalg.norm(A_pow_k) < eps
              

def find_optimal_merge(clusters,E,E_low,X,KL):
    """
    Considers merging all pairs of clusters and returns a new set of clusters
    and total treatment effect matrix E_merged that corresponds to an acyclic
    graph over the new clusters and minimises a reconstruction loss.

    Parameters:
        clusters (list): list of length k consisting of tuples of low-level variable indices {0,...,p-1}.
        E (numpy.ndarray): (r x r) total treatment effect matrix over clusters (1= < r <= p).
        E_low (numpy.ndarray): (p x p) total treatment effect matrix over low-level variables.
        X (numpy.ndarray): (N x p) dataset matrix over low-level variables.
        KL (bool): flag that determines whether a KL divergence consistency loss is considered.

    Returns:
        new_clusters (list): list of length k-1 obtained by merging two clusters into one.
        E_merged (numpy.ndarray): (k-1)x(k-1) total treatment effect matrix over the new clusters.  
    """
    assert E.shape[0] == len(clusters), f"Expected E with shape ({len(clusters)} x {len(clusters)}). Got {E.shape}."
    
    k = len(clusters)
    best_loss = np.inf
    best_merge = None
    for i in range(k):
        for j in range(i):
            new_clusters,E_merged = new_merge(clusters,i,j,E,X)
            E_hat = new_unmerge(new_clusters,E_merged)
            if is_acyclic(E_merged):
                loss = reconstruction_rmse(E_hat,X=X)
                if KL:
                    T = cluster_to_matrix(new_clusters)
                    #loss += kl_loss(E_high=E_merged,E_low=E_low,T=T.T)
                if loss < best_loss:
                    best_loss = loss
                    best_merge = copy.deepcopy((new_clusters,E_merged))
    
    return best_merge

def sequential_merge(k,E_low,X,kl_loss=False):
    """
    Params:
        k (int): the number of clusters we want in the high-level model.
        E_low (numpy.ndarray): (p x p) total treatment effect matrix over low-level variables.
        X (numpy.ndarray): (N x p) dataset matrix over low-level variables.

    Returns:
        clusters_dict (dict): A dictionary of clusters for each step of the algorithm from p to k clusters.
        E_merged_dict (dict): A dictionary of optimal total effect matrices over the clusters at each step.
        E_hat_dict (dict): A dictionary of reconstructed total effect matrix for each clustering step from p to k clusters.
    """
    p = E_low.shape[0]
    E = copy.deepcopy(E_low)
    clusters = [(i,) for i in range(p)]
    num_merges_requried = min(p,max(0,p-k)+1)-1 #ensures that 1 <= k <= p

    clusters_dict = {p:clusters}
    T_dict = {p:np.eye(p)}
    E_merged_dict = {p:E_low}
    E_hat_dict = {p:E_low}
    
    for s in range(1,num_merges_requried+1):
        clusters,E = find_optimal_merge(clusters,E,E_low,X,kl_loss)
        E_hat = new_unmerge(clusters,E)

        clusters_dict[p-s] = clusters
        T_dict[p-s] = cluster_to_matrix(clusters)
        E_merged_dict[p-s] = E
        E_hat_dict[p-s] = E_hat

    return T_dict,E_merged_dict,E_hat_dict