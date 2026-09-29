import numpy as np
import itertools

def partition_to_matrix(partition):
    p = sum(len(cluster) for cluster in partition)
    k = len(partition)
    T = np.zeros((p, k))
    for j, cluster in enumerate(partition):
        T[cluster, j] = 1
        
    return T

def intra_cluster_edges(A_low, partition):
    A_binary = (A_low != 0)
    T = partition_to_matrix(partition)
    intra_mask = T @ T.T
    return np.sum(A_binary * intra_mask) // 2

def cluster_graph(A_low, partition):
    A_binary = (A_low != 0)
    k = len(partition)
    A_partition = np.zeros((k,k),dtype=bool)

    for i in range(k):
        for j in range(k):
            A_partition[i, j] = np.any(A_binary[np.ix_(partition[i], partition[j])])

    return A_partition

def is_connected_bfs(A):
    n = A.shape[0]
    visited = set()
    queue = [0]
    
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            neighbors = np.where(A[node] != 0)[0]
            queue.extend([nbr for nbr in neighbors if nbr not in visited])
            
    return len(visited) == n

def oc_closure(A_low, k, n_restarts=10, seed=None):
    """
    Takes the adjacency matrix of a connected directed graph and returns a k-cluster partition of the nodes that satisfies the One-Child condition.
    Begin with singelton clusters and iteratively merge the children of each cluster until all clusters have at most one child.
    If the number of clusters at the end of this process is greater than k, we first merge spouses (minimising intra-cluster edges) 
    then we merge sinks with their parents.

    Parameters:
    -----------
    A_low : np.ndarray
        The adjacency matrix of a connected directed graph.
    k : int
        The maximum number of clusters to return.
    n_restarts : int
            The number of times to restart the algorithm.
    seed : int, optional
        The random seed for reproducibility.

    Returns:
    --------
    partition : list of lists
        A k-cluster partition of the nodes that satisfies the One-Child condition.
    """

    assert k > 1, "k must be greater than 1"

    p = np.shape(A_low)[0]
    A_binary = (A_low != 0)
    rng = np.random.default_rng(seed)
    candidates = []

    def find(x):
        while pointer[x] != x:
            # since x is not the root (it points to some other node z),
            # we update the pointer by letting node x point directly 
            # to whatever node z points to (its grandparent).
            pointer[x] = pointer[pointer[x]] 
            x = pointer[x]
        return x
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            pointer[ry] = rx

    def get_n_clusters():
        return len(set(find(r) for r in range(p)))

    def get_partition():
        clusters = {}
        for r in range(p):
            clusters.setdefault(find(r),[]).append(r)
        return list(clusters.values())

    for _ in range(n_restarts):
        pointer = [r for r in range(p)]
        changed = True
        while changed:
            changed = False
            clusters = {}
            for r in range(p):
                clusters.setdefault(find(r),[]).append(r)
            roots = list(clusters.keys())
            rng.shuffle(roots)
            for i in roots:
                nodes = clusters[i]
                children = {find(s) for r in nodes for s in np.where(A_binary[r])[0] if find(s) != i}
                if len(children) > 1:
                    changed = True
                    children_sorted = sorted(children)
                    for clst in children_sorted[1:]:
                        union(children_sorted[0], clst)
        
        partition = get_partition()
        A_partition = cluster_graph(A_low, partition)
            
        if get_n_clusters() > k: 
            # merge spouses
            n_clusters = len(partition)
            spouses = []
            for i in range(n_clusters):
                spouses.extend(list(itertools.combinations(np.where(A_partition[:,i])[0], 2)))

            def cross_edges(i, j):
                return np.sum(A_binary[np.ix_(partition[i], partition[j])])

            sorted_spouses = sorted(spouses, key=lambda x: cross_edges(x[0], x[1]), reverse=False)

            while get_n_clusters() > k and sorted_spouses:
                i, j = sorted_spouses.pop(0)
                union(partition[i][0], partition[j][0])

            # merge sinks with parents
            partition = get_partition()
            A_partition = cluster_graph(A_low, partition)
            sinks = list(np.where(np.sum(A_partition, axis=1) == 0)[0])

            while get_n_clusters() > k and sinks:
                sink = sinks.pop(0)
                parents = np.where(A_partition[:,sink] == 1)[0]
                if len(parents) > 0:
                    parent = parents[0]
                    union(partition[parent][0],partition[sink][0])

            # merge parents with children
            partition = get_partition()
            A_partition = cluster_graph(A_low, partition)
            clusters = [(i,np.where(A_partition[i])[0][0]) for i in np.where(A_partition.sum(axis=1)>0)[0]]
            sorted_clusters = sorted(clusters, key = lambda x: cross_edges(x[0], x[1]), reverse=False)
            while get_n_clusters() > k and sorted_clusters:
                i, j = sorted_clusters.pop(0)
                union(partition[i][0], partition[j][0])

        partition = get_partition()
        candidates.append((partition, intra_cluster_edges(A_low, partition)))

    return min(candidates, key=lambda x: x[1])[0]
        


    pass

def aoc_closure(A_low, k):
    pass

def greedy_allin(A_low, k):
    pass