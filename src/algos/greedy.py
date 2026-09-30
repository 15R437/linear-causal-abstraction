import numpy as np

def log(msg, verbose=True):
    if verbose:
        print(msg)

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

def jaccard_out(A_low):
    A_binary = (A_low != 0).astype(float)
    child_intersection = A_binary @ A_binary.T
    out_degree = A_binary.sum(axis=1)
    child_union = out_degree[None, :] + out_degree[:, None] - child_intersection
    
    with np.errstate(divide='ignore', invalid='ignore'):
        jac = np.where(child_union > 0, child_intersection / child_union, 1.0)
    np.fill_diagonal(jac, 1.0)
    return jac

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

def merge_edge_counts(edge_counts, i, j):
    """Updates cluster_edge_counts to match merge_clusters(partition, i, j): rows i and j are summed into a new last row, likewise the columns."""
    keep = [c for c in range(len(edge_counts)) if c != i and c != j]
    counts = np.zeros((len(keep) + 1, len(keep) + 1))
    counts[:-1, :-1] = edge_counts[np.ix_(keep, keep)]
    counts[-1, :-1] = (edge_counts[i] + edge_counts[j])[keep]
    counts[:-1, -1] = (edge_counts[:, i] + edge_counts[:, j])[keep]
    return counts

def one_child_closure(A_low, rng):
    """
    Starting from singleton clusters, visits the clusters in a random order and merges the children of any
    cluster with more than one child, until every cluster has at most one child.
    Returns the partition as a list of lists.
    """
    p = np.shape(A_low)[0]
    A_binary = (A_low != 0)
    node_children = [np.where(A_binary[r])[0] for r in range(p)]
    pointer = [r for r in range(p)]

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

    def get_partition():
        clusters = {}
        for r in range(p):
            clusters.setdefault(find(r),[]).append(r)
        return list(clusters.values())

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
            children = {find(s) for r in nodes for s in node_children[r] if find(s) != find(i)}
            if len(children) > 1:
                changed = True
                children_sorted = sorted(children)
                for clst in children_sorted[1:]:
                    union(children_sorted[0], clst)

    return get_partition()


def oc_closure(A_low, k, n_restarts=1, seed=None, return_candidates=False):
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

    rng = np.random.default_rng(seed)
    candidates = []

    for _ in range(n_restarts):
        partition = one_child_closure(A_low, rng)
        partition = merge_spouses(A_low, partition, k)
        partition = merge_sinks(A_low, partition, k)
        partition = merge_fewest_edges(A_low, partition, k)
        candidates.append((partition, intra_cluster_edges(A_low, partition)))

    if return_candidates:
        return candidates
    return min(candidates, key=lambda x: x[1])[0]

def all_in_repair(A_low, partition):
    """
    Merges a cluster containing an All-In violating node with its child cluster, until All-In holds.
    Each merge removes a cluster, so this terminates. One-Child is preserved, since the merged cluster
    inherits the child's single child.
    """
    while True:
        violations = all_in_violation_matrix(A_low, partition)
        if not violations.any():
            return partition
        i, j = np.argwhere(violations)[0]
        partition = merge_clusters(partition, i, j)

def merge_spouses(A_low, partition, k):
    """
    Merges two clusters that share a child, until k clusters remain or no cluster has two parents.
    Under One-Child a spouse's only child is the shared one, so no edges run between spouses:
    each merge adds no intra-cluster edges and preserves All-In and One-Child.
    """
    edge_counts = cluster_edge_counts(A_low, partition)
    while len(partition) > k:
        shared_children = np.where((edge_counts > 0).sum(axis=0) > 1)[0]
        if len(shared_children) == 0:
            break
        i, j = np.where(edge_counts[:, shared_children[0]])[0][:2]
        partition = merge_clusters(partition, i, j)
        edge_counts = merge_edge_counts(edge_counts, i, j)
    return partition

def merge_sinks(A_low, partition, k):
    """
    Merges a sink cluster with one of its parents, choosing the pair joined by the fewest edges,
    until k clusters remain or no sink has a parent. Under One-Child the parent's only child is the sink,
    so the merged cluster is a sink and All-In and One-Child are preserved.
    """
    edge_counts = cluster_edge_counts(A_low, partition)
    while len(partition) > k:
        sinks = np.where(edge_counts.sum(axis=1) == 0)[0]
        pairs = [(edge_counts[parent, sink], parent, sink) for sink in sinks for parent in np.where(edge_counts[:, sink])[0]]
        if not pairs:
            break
        _, parent, sink = min(pairs)
        partition = merge_clusters(partition, parent, sink)
        edge_counts = merge_edge_counts(edge_counts, parent, sink)
    return partition

def merge_fewest_edges(A_low, partition, k):
    """
    Merges the pair of clusters joined by the fewest edges (in either direction), until k clusters remain.
    A last resort: unlike merge_spouses and merge_sinks, these merges can violate One-Child and All-In.
    """
    edge_counts = cluster_edge_counts(A_low, partition)
    while len(partition) > k:
        between = edge_counts + edge_counts.T
        np.fill_diagonal(between, np.inf)
        i, j = np.unravel_index(np.argmin(between), between.shape)
        partition = merge_clusters(partition, i, j)
        edge_counts = merge_edge_counts(edge_counts, i, j)
    return partition

def aoc_closure(A_low, k, n_restarts=1, seed=None, return_candidates=False):
    """
    Returns a partition of the nodes that satisfies both the One-Child and All-In conditions.
    Computes the one-child closure, then merges All-In violating clusters with their children until All-In holds.
    While more than k clusters remain, merges spouses and then merges sinks with their parents; both preserve
    the two conditions. The result has k clusters unless the first two steps already leave k or fewer,
    or no sink has a parent (only possible for a disconnected graph).

    Parameters:
    -----------
    A_low : np.ndarray
        The adjacency matrix of a directed graph.
    k : int
        The target number of clusters.
    n_restarts : int
        The number of random closure orders to try.
    seed : int, optional
        The random seed for reproducibility.
    return_candidates : bool
        If True, return every restart's (partition, intra-cluster edge count) instead of the best partition.

    Returns:
    --------
    partition : list of lists
        The candidate closest to k clusters, ties broken by the fewest intra-cluster edges.
    """
    rng = np.random.default_rng(seed)
    candidates = []

    for _ in range(n_restarts):
        partition = one_child_closure(A_low, rng)
        partition = all_in_repair(A_low, partition)
        partition = merge_spouses(A_low, partition, k)
        partition = merge_sinks(A_low, partition, k)
        candidates.append((partition, intra_cluster_edges(A_low, partition)))

    if return_candidates:
        return candidates
    return min(candidates, key=lambda x: (abs(len(x[0]) - k), x[1]))[0]

def greedy_allin(A_low, k, n_candidates=100):
    """
    Greedy agglomerative clustering towards All-In. Starting from singleton clusters, repeatedly merges the pair
    of clusters that leaves the fewest All-In violations, breaking ties by the fewest edges added inside a cluster.
    Only the n_candidates pairs with the highest mean Jaccard similarity of out-neighbourhoods are scored at
    each step. One-Child is not enforced.

    Returns:
    --------
    partition : list of lists
        A k-cluster partition of the nodes.
    """
    p = A_low.shape[0]
    jac = jaccard_out(A_low)
    A_binary = (A_low != 0)
    pointer = np.arange(p) # pointer[r] = label of the cluster containing node r

    while len(np.unique(pointer)) > k:
        clusters, pos = np.unique(pointer, return_inverse=True) # pos[r] = index of r's cluster in clusters
        n_clusters = len(clusters)
        partition = [np.where(pos == i)[0] for i in range(n_clusters)]

        # mean pairwise Jaccard between clusters: average rows within each cluster, then columns
        mean_jac = np.array([jac[clst].mean(axis=0) for clst in partition])
        cluster_jac = np.column_stack([mean_jac[:, clst].mean(axis=1) for clst in partition])
        iu = np.triu_indices(n_clusters, 1)
        vals = cluster_jac[iu]
        top = np.argsort(-vals, kind='stable')[:n_candidates] # most similar first, ties in (i, j) order

        H, C = cluster_connectivity(A_low, partition)
        cluster_sizes = np.bincount(pos)
        V = (C > 0) * (cluster_sizes[:, None] - C)
        np.fill_diagonal(V, 0)

        best_pair, best_score = None, None
        for t in top:
            i, j = int(iu[0][t]), int(iu[1][t])
            others = np.ones(n_clusters, dtype=bool)
            others[[i, j]] = False

            # merging i and j only changes the violations in rows and columns i and j
            removed = V[i].sum() + V[j].sum() + V[:, i].sum() + V[:, j].sum() - V[i, j] - V[j, i]
            # the merged cluster as a parent: its members reaching each cluster add up, as i and j are disjoint
            C_row = (C[i] + C[j])[others]
            added_row = ((C_row > 0) * (cluster_sizes[i] + cluster_sizes[j] - C_row)).sum()
            # the merged cluster as a child: a node reaches it if it reaches i or j
            C_col = np.bincount(pos, weights=np.maximum(H[:, i], H[:, j]), minlength=n_clusters)[others]
            added_col = ((C_col > 0) * (cluster_sizes[others] - C_col)).sum()
            # change in the total violation count; ranks the same as the total
            violation_change = added_row + added_col - removed

            # edges the merge adds inside a cluster; ranks the same as the total intra-cluster count
            added_edges = A_binary[np.ix_(partition[i], partition[j])].sum() + A_binary[np.ix_(partition[j], partition[i])].sum()
            score = (violation_change, added_edges)

            if best_score is None or score < best_score:
                best_pair, best_score = (i,j), score

        i, j = best_pair
        pointer[pos == j] = clusters[i]

    return [list(np.where(pointer == c)[0]) for c in np.unique(pointer)]

        


if __name__ == "__main__":
    import os
    import sys
    import time
 
    # This script uses the rest of the repo; put the repo root on the path so the
    # `src.*` imports below resolve regardless of the working directory.
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    from src.utils.generate_data import sample_low_level_graph
    from src.utils.plots import visualize_causal_graph, plot_matrix

    p,k,d,seed = 10,5,0.8,None
    A_low,T_gt,S_gt,A_high_gt = sample_low_level_graph(n_high=k, n_low=p, density=d, seed=seed, graph_type='one_child')
    T_gt_binary = (T_gt !=0).astype(float)

    start = time.time()
    partition = greedy_allin(A_low, k)
    runtime = time.time() - start
    print(f"Finished in {runtime:.4f}s")
    T_hard = partition_to_matrix(partition)
    A_high_binary = cluster_graph(A_low, partition)

    visualize_causal_graph(A_low, A_high_binary, T_hard, T_hard)
    plot_matrix(matrix_list=[T_gt_binary,T_hard], title_list=['GT', 'OC Closure'])
