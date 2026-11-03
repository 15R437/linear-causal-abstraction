"""Sample a low-level model whose HIGH-LEVEL model is cyclic, keeping the abstraction valid.

The wrong way (scripts/cyclic.py, and my first attempt at this) is to sample an acyclic pair
and then reverse edges in A_low. That mutation is not in the image of T_concretisation_sampling
-- each low-level block is a rank-one outer product tied to t_i and s_j -- so afterwards
(A_low, T_gt, S_gt, A_high) is no longer an abstraction pair at all, and no exact abstraction
need exist at the generating partition. It also used weights ~U(0.5, 2) against a matrix whose
median entry is ~0.1, so the added edges dominated the Frobenius mass.

The right way is to put the cycles in A_high and CONCRETISE from there:

    A_high  <- random DAG + n_back reversed edges, rescaled to rho < RHO_HIGH
    T_gt    <- generate_random_T
    A_low   <- T_concretisation_sampling(A_high, T_gt)      (S_gt comes back exact)

T_concretisation_sampling never assumes A_high is acyclic -- it only inverts (I - A_jj) for
the INTERNAL block DAG, which stays acyclic -- so the pair it returns is a valid abstraction
with a genuinely cyclic high-level model, and the quotient of the generating partition is
cyclic by construction rather than by accident.
"""
import numpy as np
from src.utils.generate_data import (random_dag, generate_random_T, T_concretisation_sampling,
                                     _is_weakly_connected)

RHO_HIGH, RHO_LOW = 0.9, 0.95


def sample_cyclic_pair(n_high, n_low, density, n_back=2, blockiness=1., heterogeneity=0.,
                       internal_density=0., seed=None, shuffle=True, connected=True,
                       max_attempts=200):
    """(A_low, T_gt, S_gt, A_high) with a cyclic A_high. Mirrors sample_low_level_graph."""
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        att = int(rng.integers(0, 2 ** 32 - 1))
        r2 = np.random.default_rng(att)
        adj = (random_dag(n_high, density, seed=att) != 0).astype(float)
        A_high = adj * r2.uniform(0.5, 2.0, size=adj.shape) * r2.choice([-1, 1], size=adj.shape)

        # reverse n_back existing high-level edges; weights drawn on the SAME scale as the
        # entries already present, so a back-edge is a peer of the edges it joins
        src, dst = np.where(A_high != 0)
        if len(src) == 0:
            continue
        scale = float(np.median(np.abs(A_high[A_high != 0])))
        n_added = 0
        for e in r2.choice(len(src), size=min(n_back, len(src)), replace=False):
            i, j = int(src[e]), int(dst[e])
            if A_high[j, i] == 0:
                A_high[j, i] = r2.uniform(0.5, 1.5) * scale * r2.choice([-1, 1])
                n_added += 1
        if n_added == 0:
            continue
        r = np.max(np.abs(np.linalg.eigvals(A_high)))
        if r >= RHO_HIGH:
            A_high = A_high * (RHO_HIGH / r)
        if not _is_cyclic(A_high):
            continue

        try:
            T_gt, _ = generate_random_T(n_low, n_high, (1.,), (1.,), marg_prob=0, seed=att)
            A_low, S_gt = T_concretisation_sampling(
                A_high=A_high, T=T_gt, internal_density=internal_density,
                blockiness=blockiness, heterogeneity=heterogeneity, seed=seed)
        except Exception:
            continue
        if not np.any(A_low != 0):
            continue
        if np.max(np.abs(np.linalg.eigvals(A_low))) >= RHO_LOW:
            continue                      # (I - A_low)^-1 must exist and be well conditioned
        if connected and not _is_weakly_connected(A_low):
            continue
        if shuffle:
            perm = rng.permutation(n_low)
            A_low, T_gt, S_gt = A_low[perm][:, perm], T_gt[perm], S_gt[perm]
        return A_low, T_gt, S_gt, A_high
    raise RuntimeError('no cyclic pair after %d attempts' % max_attempts)


def _is_cyclic(A):
    n = A.shape[0]
    R = (A != 0).copy()
    for _ in range(int(np.ceil(np.log2(max(n, 2)))) + 1):
        R = R | (R @ R)
    return bool(np.any(np.diag(R)))
