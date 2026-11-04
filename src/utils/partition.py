import numpy as np


def partition_from_matrix(T_hard, validate=True):
    """Cluster index arrays from a 0/1 assignment matrix, one per column.

    Empty clusters are KEPT, so the result has one entry per column of T_hard and A_high
    stays k x k: callers that want only live clusters filter on `len`. `validate` checks the
    columns really are a partition of 0..p-1, which is the only way this goes wrong quietly --
    `fit_abstraction_eigen` infers p from the cluster sizes, so an unassigned or
    doubly-assigned variable surfaces later as a shape error far from its cause.
    """
    T_hard = np.asarray(T_hard)
    if validate:
        counts = (T_hard != 0).sum(axis=1)
        if not np.all(counts == 1):
            raise ValueError(
                f"T_hard is not a partition: {int((counts == 0).sum())} variable(s) in no "
                f"cluster, {int((counts > 1).sum())} in more than one. "
                f"Expected exactly one cluster per variable.")
    return [np.where(T_hard[:, i] != 0)[0] for i in range(T_hard.shape[1])]


def partition_to_matrix(partition):
    p = sum(len(cluster) for cluster in partition)
    k = len(partition)
    T = np.zeros((p, k))
    for j, cluster in enumerate(partition):
        T[cluster, j] = 1
        
    return T

def intra_cluster_edges(A_low, partition):
    A_binary = (A_low != 0)
    count = 0
    for clst in partition:
        if len(clst) > 1:
            count += A_binary[np.ix_(clst,clst)].sum()

    return count

def cluster_graph(A_low, partition):
    A_binary = (A_low != 0)
    k = len(partition)
    A_partition = np.zeros((k,k),dtype=bool)

    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            A_partition[i, j] = np.any(A_binary[np.ix_(partition[i], partition[j])])

    return A_partition

def cluster_connectivity(A_low, partition):
    """
    Returns H, the (p, k) matrix whose (r, j) entry is 1 if node r has an edge into cluster j,
    and C, the (k, k) matrix whose (i, j) entry counts the nodes in cluster i with an edge into cluster j.
    """
    A_binary = (A_low != 0).astype(float)
    p = A_low.shape[0]
    k = len(partition)

    H = np.zeros((p,k))
    for j, clst in enumerate(partition):
        H[:,j] = np.any(A_binary[: , clst],axis=1) # node-to-cluster connectivity matrix

    C = np.zeros((k,k))
    for i, clst in enumerate(partition):
        C[i] = H[clst].sum(axis=0) # how many nodes in cluster i have a connection to each j

    return H, C

def all_in_violation_matrix(A_low, partition):
    """(k, k) matrix whose (i, j) entry counts the nodes in cluster i with no edge into its child cluster j."""
    _, C = cluster_connectivity(A_low, partition)
    cluster_sizes = np.array([len(clst) for clst in partition])[:, None]
    V = (C > 0) * (cluster_sizes - C)
    np.fill_diagonal(V,0)

    return V

def all_in_violation(A_low, partition):
    return np.sum(all_in_violation_matrix(A_low, partition))

def cluster_edge_counts(A_low, partition):
    """(k, k) matrix whose (i, j) entry counts the edges from cluster i into cluster j, with a zero diagonal."""
    labels = np.empty(A_low.shape[0], dtype=int)
    for c, clst in enumerate(partition):
        labels[clst] = c

    k = len(partition)
    sources, targets = np.nonzero(A_low)
    counts = np.bincount(labels[sources] * k + labels[targets], minlength=k * k).reshape(k, k).astype(float)
    np.fill_diagonal(counts, 0)
    return counts

def merge_clusters(partition, i, j):
    """Returns the partition with clusters i and j replaced by their union (appended at the end)."""
    merged = list(partition[i]) + list(partition[j])
    return [clst for c, clst in enumerate(partition) if c != i and c != j] + [merged]
