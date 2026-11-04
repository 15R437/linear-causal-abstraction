"""The algebra relating a low-level SCM to its abstraction, plus the DAG surgery applied
to the result.

Given A_low and a fitted T, three quantities follow: S (`get_S`), A_high (`get_A_high`) and
the reconstruction T A_high pinv(T)^T (`reconstruct_adj`). `enforce_dag` and `is_dag` are the
acyclicity tools the experiments apply to a fitted A_high -- `synthetic/benchmark.py` nests them,
`enforce_dag(get_A_high(...))`.

Finding the partition lives in `algos/`; operating on one lives in `utils/partition.py`.
"""

import numpy as np
import torch

from src.utils.partition import partition_from_matrix


def get_device(cpu=True):
    """Best available torch device: CUDA, then Apple Silicon MPS, then CPU."""
    if cpu:
        return torch.device('cpu')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def reconstruct_adj(A_high, T):
    """Push A_high back down through T: the low-level graph the abstraction implies."""
    R = np.linalg.pinv(T.T)
    return T @ A_high @ R.T


def enforce_dag(A):
    """Greedily drop the smallest-magnitude edges that close a cycle.

    Edges are re-added in descending |weight| order and an edge is skipped whenever the
    reverse path already exists, so the result is acyclic and keeps as much weight as a
    single greedy pass can. Accepts a numpy array or a torch tensor and returns that type.
    """
    is_torch = hasattr(A, "detach")
    if is_torch:
        device, dtype = A.device, A.dtype
        A_np = A.detach().cpu().numpy()
    else:
        A_np = np.asarray(A)

    n = A_np.shape[0]
    A_dag = np.zeros_like(A_np)

    edges = [(abs(A_np[i, j]), A_np[i, j], i, j)
             for i in range(n) for j in range(n) if A_np[i, j] != 0]
    edges.sort(key=lambda e: e[0], reverse=True)

    def has_path(adj, u, v):
        """DFS: does a path u -> ... -> v already exist in adj?"""
        stack, visited = [u], set()
        while stack:
            curr = stack.pop()
            if curr == v:
                return True
            if curr not in visited:
                visited.add(curr)
                stack.extend(nxt for nxt in range(n) if adj[curr, nxt] != 0)
        return False

    for _, w, i, j in edges:
        if not has_path(A_dag, j, i):
            A_dag[i, j] = w

    if is_torch:
        return torch.tensor(A_dag, device=device, dtype=dtype)
    return A_dag


def is_dag(A):
    """True if the square adjacency matrix A is acyclic (Kahn's algorithm)."""
    n = A.shape[0]
    in_degree = np.sum(A != 0, axis=0)

    queue = [i for i in range(n) if in_degree[i] == 0]
    visited_count = 0

    while queue:
        u = queue.pop(0)
        visited_count += 1
        for v in range(n):
            if A[u, v] != 0:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

    return visited_count == n


def get_S(T, E_low=None, T_hard=None, A_low=None, partition=None):
    """S_i = (I - A_low[pi_i, pi_i])^-1 t_i, supported on cluster i.

    FIXED 2026-09-23. The previous version returned E_low[pi_i, pi_i] @ t_i. That is only
    correct when the QUOTIENT is acyclic. The abstraction equation (I - A_low) S = T (I-A_high)
    has diagonal block (I - A_low[pi_i, pi_i]) S_ii = t_i -- every cross term dies because T
    and S are both block-supported, and the only A_high entry that could appear is the
    diagonal, which is zero -- so S_ii is the LOCAL inverse. E_low's diagonal block instead
    also counts paths that leave cluster i and come back, and with a cyclic quotient the two
    differ by O(1): the generating abstraction scored RAE = 0.02-0.21 against ITSELF (squared
    scale, i.e. 0.14-0.46 under the ratio-of-norms RAE in use since 2026-09-24). Random
    partitions induce a cyclic quotient 87-100% of the time even on acyclic graphs, so control
    arms were affected too. See experiments/abstraction_study/REPORT.md.

    Equivalently S_i = e_ii^-1 E_low[pi_i, pi_i] t_i, from the (pi_i, i) block of
    E_low T = S E_high. That form is a VECTOR identity in the direction t_i holding only under
    exactness -- as matrices e_ii^-1 E_ii and (I - A_ii)^-1 differ by up to 4.0 on cyclic
    instances -- so this uses the local inverse, which needs no such assumption and is correct
    on approximate fits too.

    The support is taken from `partition` when given (what the fitters and the searches carry
    natively), otherwise from `T_hard`, otherwise from the sparsity pattern of T itself.

    Pass A_low. E_low is accepted only so older call sites keep working; A_low is then
    recovered from it at the cost of one extra O(p^3) inverse.
    """
    if A_low is None:
        if E_low is None:
            raise ValueError("get_S requires A_low (preferred) or E_low")
        A_low = np.eye(E_low.shape[0]) - np.linalg.inv(E_low)
    if partition is None:
        if T_hard is None:
            T_hard = (T != 0).astype(int)
        partition = partition_from_matrix(T_hard)

    S = np.zeros_like(T)
    for i in range(T.shape[1]):
        if len(partition[i]) == 0:
            continue
        A_ii = A_low[np.ix_(partition[i], partition[i])]
        S[partition[i], i] = np.linalg.solve(np.eye(len(partition[i])) - A_ii, T[partition[i], i])
    return S


def get_A_high(T, A_low, priority='abstraction', T_hard=None, partition=None):
    """Least-squares A_high induced by T.

    `priority='abstraction'` solves for the weights that best satisfy the abstraction
    equation; `priority='recon'` instead minimises reconstruction error. The diagonal is
    zeroed by convention in the 'abstraction' branch -- a cluster has no self-edge.

    `T_hard` and `partition` are passed through to `get_S`, and are what make this usable on
    a SOFT T: without one, the support is read from T's own sparsity pattern, and a dense T
    puts every variable in every cluster, which is not a partition. Pass the argmax support
    when evaluating an unconverged or interpolated T.
    """
    n_low, n_high = T.shape
    if priority == 'recon':
        with np.errstate(divide='ignore', invalid='ignore'):
            T_norm = T / np.sum(T**2, axis=0)

        T_norm = np.nan_to_num(T_norm)
        A_high = T_norm.T @ A_low @ T

    elif priority == 'abstraction':
        E_low = np.linalg.inv(np.eye(n_low) - A_low)
        S = get_S(T, E_low, T_hard=T_hard, A_low=A_low, partition=partition)
        AS = A_low @ S

        A_high = T.T @ AS
        t_norms = np.sum(T**2, axis=0)
        t_norms = np.clip(t_norms, 1e-8, None)

        A_high = A_high / t_norms[:, None]
        A_high = A_high * (1 - np.eye(n_high))

    return A_high
