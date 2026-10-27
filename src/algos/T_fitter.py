"""Closed-form T-fitter: sinks are free, cycle members are PINNED by an eigenvector equation.

THE IDENTITY. Every quotient edge i->j imposes the Massidda collinearity condition

    M_ij t_j  =  m_ij t_i,        M_ij := A[pi_i, pi_j] (I - A[pi_j, pi_j])^{-1},

with m_ij the high-level weight. Composing it around a cycle i_1 -> i_2 -> ... -> i_1 and
cancelling the scalars gives

    Pi t_{i_1} = (prod_l m_l) t_{i_1},     Pi := M_{i1 i2} M_{i2 i3} ... M_{ik i1},

so a cluster on a cycle is an EIGENVECTOR of the cycle product, with eigenvalue the product
of the high-level weights around it. Verified to 1e-16 on every cycle tested, with the
eigenvalue matching prod A_high exactly.

WHY THIS IS A SOLVER AND NOT JUST AN IDENTITY. It removes the cyclic case from the
optimiser entirely. The degrees of freedom split three ways:

    sink cluster        t free                (nothing downstream constrains it)
    cluster on a cycle  t PINNED by the eigenproblem -- no free parameters at all
    anything else       t propagated from its children, as in `fit_abstraction`

At blockiness 1 each block A[pi_i, pi_j] is RANK ONE, so Pi is rank one and its single
nonzero eigenvector is the answer -- no combinatorial choice. Below blockiness 1 the blocks
have rank 5-6 and Pi carries ~8 nonzero eigenvalues, but the true t came back as the
LARGEST-MAGNITUDE eigenvector in every instance tested (n=4-5 per level, blockiness 0.5-0.9),
so `argmax |lambda|` is the rule here. That last point is measured, not proved -- treat it as
provisional below blockiness 1.

WHAT IT FIXES. `fit_abstraction_K` annihilates outright on cyclic quotients at small k
(edge recall 0.000 at k=4 and k=6, p=100). This returns machine precision there, 40-140x
faster, because there is nothing to search.

SINKS ARE OPTIMISED, NOT SAMPLED (fixed 2026-09-26). The abstraction condition does not pin
a sink's direction, but the choice is not free of consequence: every cluster upstream is
propagated through M_ij t_j, so a wrong sink corrupts the whole chain above it. How much
depends on the block rank. At blockiness 1 the blocks are RANK ONE, so M_ij t_j is parallel
to the block's left factor whatever t_j is -- measured min |cos| = 1.0000 over 50 random
draws -- and the sink is genuinely irrelevant. At blockiness 0.8 the rank is 3 and that
min |cos| falls to 0.1025, so it matters a great deal.

Drawing sinks at random therefore fails outside the rank-one regime, and fails as a SEARCH
rather than structurally: at blockiness 0.8 the RAE runs 0.210 -> 0.105 -> 0.063 -> 0.058 for
1, 5, 50, 200 random draws, against 2.2e-05 for `fit_abstraction`, which optimises the same
parameters with SLSQP. This now does the same -- the sink directions are the optimisation
variables and the collinearity disagreement is the objective -- so on an acyclic quotient it
reduces to `fit_abstraction`'s propagation, and the eigen step is a strict addition for the
cyclic case. With NO sinks (the usual cyclic case) there is nothing to optimise and the
result is closed form.

OVERDETERMINATION IS REAL AND IS NOT RECONCILED HERE. A cluster on several cycles, or one
that both sits in an SCC and points downstream out of it, must satisfy all of those
constraints at once. This solver takes ONE cycle per node (the shortest within its SCC) and
lets any disagreement show up as residual rather than least-squaring it away, so the residual
is diagnostic. `penalty` in the returned info is the multi-child/one-cycle disagreement.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

__all__ = ['fit_abstraction_eigen']


def _M_block(A, partition, a, b):
    """M_ab = A[pi_a, pi_b] (I - A[pi_b, pi_b])^{-1}."""
    Wab = A[np.ix_(partition[a], partition[b])]
    Wbb = A[np.ix_(partition[b], partition[b])]
    return Wab @ np.linalg.inv(np.eye(len(partition[b])) - Wbb)


def _cycle_within(M, comp, ci, start, maxlen=12):
    """Shortest simple cycle through `start` using only edges inside SCC `ci`."""
    from collections import deque
    q = deque([[start]])
    while q:
        path = q.popleft()
        if len(path) > maxlen:
            continue
        for v in np.where(M[path[-1]])[0]:
            if comp[v] != ci:
                continue
            if v == start and len(path) >= 1:
                return path
            if v not in path:
                q.append(path + [v])
    return None


def _reconcile(votes):
    """Combine several collinearity votes for the same t, and report their disagreement.

    Each vote is only defined up to sign and scale, so they are unit-normalised and
    sign-aligned to the first before averaging. The residual after averaging is the
    disagreement the abstraction condition cannot satisfy -- the same quantity
    `fit_abstraction` minimises as its multi-child penalty.
    """
    u = []
    for v in votes:
        n = np.linalg.norm(v)
        if n > 1e-12:
            u.append(v / n)
    if not u:
        return None, 0.0
    ref = u[0]
    u = [w if w @ ref >= 0 else -w for w in u]
    m = np.mean(u, axis=0)
    n = np.linalg.norm(m)
    if n <= 1e-12:
        return u[0], float(sum(np.linalg.norm(w - u[0]) ** 2 for w in u))
    m = m / n
    return m, float(sum(np.linalg.norm(w - m) ** 2 for w in u))

def get_S(A_low, T, partition=None):
    """
    s_i = (I - A_low[pi_i, pi_i])^-1 t_i, supported on cluster i.
    """

    if T_hard is None:
        T_hard = (T != 0).astype(int)
    
    S = np.zeros_like(T)
    for i in range(T.shape[1]):
        if len(partition[i]) == 0:
            continue
        A_ii = A_low[np.ix_(partition[i], partition[i])]
        S[partition[i], i] = np.linalg.solve(np.eye(len(partition[i])) - A_ii, T[partition[i], i])
    return S


def fit_abstraction_eigen(A_low, partition, seed=0, n_restarts=5, maxiter=200,
                          return_info=False):
    """Fit (A_high, T): eigen-solve cyclic SCCs, propagate the rest, OPTIMISE the sinks.

    Returns (A_high, T, S), or (A_high, T, info) with return_info=True. `info` carries
    `penalty` (collinearity disagreement at the optimum), `n_sinks`, `n_cyclic` (clusters
    pinned by an eigenproblem) and `n_scc`.

    Cyclic SCC members do not depend on the sinks, so they are solved ONCE, outside the
    optimisation. Only the sink directions are searched, over sum(|pi_s|) parameters rather
    than p, with `n_restarts` random starts. When there are no sinks the whole fit is closed
    form and `n_restarts` is ignored.
    """
    from scipy.optimize import minimize

    A = np.asarray(A_low, dtype=float)
    p = sum(len(cluster) for cluster in partition)
    k = len(partition)

    if A_low.shape != (p , p):
        raise ValueError(f"A_low shape {A_low.shape} does not match partition size {p}.")
    
    A_binary = (A != 0)
    M = np.zeros((k, k), bool)
    for i in range(k):
        pi_i = partition[i]
        if not len(pi_i):
            continue
        reach = A_binary[pi_i].any(0)
        for j in range(k):
            if i != j and len(partition[j]) and reach[partition[j]].any():
                M[i, j] = True

    # every M_ij once; the optimiser reuses them on each forward pass
    Mb = {}
    for i in range(k):
        for j in np.where(M[i])[0]:
            if len(partition[i]) and len(partition[j]):
                Mb[(i, int(j))] = _M_block(A, partition, i, int(j))

    ncomp, comp = connected_components(csr_matrix(M), directed=True, connection='strong')
    cond = {c: set() for c in range(ncomp)}
    for i in range(k):
        for j in np.where(M[i])[0]:
            if comp[i] != comp[j]:
                cond[comp[i]].add(comp[j])
    order, seen = [], np.zeros(ncomp, np.int8)

    def visit(c):
        if seen[c]:
            return
        seen[c] = 1
        for d in cond[c]:
            visit(d)
        order.append(c)
    for c in range(ncomp):
        visit(c)

    members = [np.where(comp == c)[0] for c in range(ncomp)]
    cyclic = [len(members[c]) > 1 or (len(members[c]) == 1 and M[members[c][0], members[c][0]])
              for c in range(ncomp)]

    # ---- cyclic SCCs are pinned and independent of the sinks: solve them once
    t_fixed = [None] * k
    n_cyc = 0
    for c in order:
        if not cyclic[c]:
            continue
        for i in members[c]:
            cyc = _cycle_within(M, comp, c, int(i))
            if cyc is None:
                continue
            Pi = None
            for a_, b_ in zip(cyc, cyc[1:] + [cyc[0]]):
                Mm = Mb.get((a_, b_))
                if Mm is None:
                    Pi = None
                    break
                Pi = Mm if Pi is None else Pi @ Mm
            if Pi is None:
                continue
            w, V = np.linalg.eig(Pi)
            v = np.real(V[:, int(np.argmax(np.abs(w)))])
            n = np.linalg.norm(v)
            if n > 1e-12:
                t_fixed[i] = v / n
                n_cyc += 1

    # ---- sinks: the only free parameters
    sinks = [int(members[c][0]) for c in order
             if not cyclic[c] and not M[members[c][0]].any()]
    sizes = [len(partition[i]) for i in sinks]
    offs = np.cumsum([0] + sizes)

    def forward(x):
        """Fill every t from the sink directions in x; return (t, collinearity penalty)."""
        t = list(t_fixed)
        for n_, i in enumerate(sinks):
            v = x[offs[n_]:offs[n_ + 1]]
            nv = np.linalg.norm(v)
            t[i] = v / nv if nv > 1e-12 else np.ones(len(partition[i])) / np.sqrt(len(partition[i]))
        pen = 0.0
        for c in order:
            if cyclic[c]:
                continue
            i = int(members[c][0])
            if i in sinks:
                continue
            votes = [Mb[(i, j)] @ t[j] for j in (int(j) for j in np.where(M[i])[0])
                     if t[j] is not None and (i, j) in Mb]
            ti, dis = _reconcile(votes)
            pen += dis
            if ti is None:
                ti = np.ones(len(partition[i])) / np.sqrt(len(partition[i]))
            t[i] = ti
        return t, pen

    n_par = int(offs[-1])
    if n_par == 0:
        best_x, best_pen = np.zeros(0), forward(np.zeros(0))[1]
    else:
        rng = np.random.default_rng(seed)
        best_x, best_pen = None, np.inf
        for _ in range(max(1, n_restarts)):
            x0 = rng.standard_normal(n_par)
            try:
                res = minimize(lambda z: forward(z)[1], x0, method='SLSQP', tol=1e-12,
                               options={'maxiter': maxiter})
                xc, pc = res.x, forward(res.x)[1]
            except Exception:
                xc, pc = x0, forward(x0)[1]
            if pc < best_pen:
                best_x, best_pen = xc, pc
        if best_x is None:
            best_x, best_pen = np.zeros(n_par), np.inf

    t, penalty = forward(best_x)
    T = np.zeros((p, k))
    for i in range(k):
        if t[i] is not None:
            T[partition[i], i] = t[i]
    A_high = np.zeros((k, k))
    for (i, j), Mm in Mb.items():
        if t[i] is None or t[j] is None:
            continue
        ti = t[i]
        A_high[i, j] = float((Mm @ t[j]) @ ti / (ti @ ti))
    if return_info:
        return A_high, T, dict(penalty=penalty, n_sinks=len(sinks), n_cyclic=n_cyc,
                               n_free=len(sinks), n_scc=ncomp)
    
    return A_high, T, get_S(A_low, T, partition)
