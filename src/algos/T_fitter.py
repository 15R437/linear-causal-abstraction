"""T-fitter: propagate every cluster from its children, free one only where propagation has
nothing to go on, and make EVERY quotient edge a residual.

THE RULE (2026-11-04). Every quotient edge i->j imposes the Massidda collinearity condition

    M_ij t_j  =  m_ij t_i,        M_ij := A[pi_i, pi_j] (I - A[pi_j, pi_j])^{-1},

with m_ij the high-level weight. Each edge therefore ties t_i to t_j, so the constraint graph
is the UNDIRECTED quotient graph and the system decomposes into weakly connected components
and no finer. Within a component the fit is one rule, with no special case for cycles:

    a cluster is a FREE PARAMETER iff it has no children (a sink) or it lies in a cyclic SCC
    (one seed per SCC); every other cluster is PROPAGATED from its children's votes; and
    every quotient edge contributes a residual

        r_ij = unit(M_ij t_j) -+ t_i,          objective = sum_ij ||r_ij||^2

    minimised over the seed directions.

An edge consumed by propagation as a cluster's only vote has r_ij = 0 by construction; an
edge that is one of several votes contributes the deviation from the reconciled mean, which
is the multi-child penalty `fit_abstraction` already minimised; an edge propagation never
used -- a cycle-closing edge, or an edge into a cluster already determined -- contributes
the whole constraint. On a k=20 random partition that last class was 182 of 371 edges, so it
is most of the constraint set, and the previous design dropped all of it.

The schedule is purely structural -- which cluster is a seed, which votes each cluster takes,
in what order -- so it is built once as a PLAN and the forward pass only executes it. That
also keeps the residual vector a fixed length, which `least_squares` requires.

THE IDENTITY IS NOW IMPLIED, NOT IMPOSED. Composing the collinearity conditions around a
cycle i_1 -> ... -> i_1 and cancelling the scalars gives

    Pi t_{i_1} = (prod_l m_l) t_{i_1},     Pi := M_{i1 i2} M_{i2 i3} ... M_{ik i1},

so a cluster on a cycle is an eigenvector of the cycle product with eigenvalue the product of
the high-level weights. That is exactly what the per-edge residuals compose to: where a cycle
member takes a single vote, t_{i_2} = unit(M_{i2 i1} t_{i1}) and the residual on the closing
edge reads unit(Pi t_{i1}) -+ t_{i1}, which vanishes iff the eigenproblem is solved. Measured
on a 2-cycle: the closing-edge residual is 5e-16 at each eigenvector of Pi with nonzero
eigenvalue, against 1.2-1.4 at the null directions. On a partition admitting an exact
abstraction the rule drives every edge to zero and the eigenvalue comes back exact to six
figures (0.300000 = A_high[0,1] A_high[1,0]); where no exact abstraction exists the cycle
condition is deliberately NOT enforced -- it is one equation weighed against the rest, which
is the whole point of fitting all of them.

WHY THE PIN WAS REPLACED (it was wrong, not merely approximate). Until 2026-11-04 a cyclic
SCC was solved outside the optimisation by `argmax |lambda|` on Pi, and cyclic SCCs then
contributed NO residual at all. So a cluster inside an SCC that also pointed downstream had
its downstream equations silently dropped: on a quotient with a 2-cycle above a sink, on the
GENERATING partition, where an exact abstraction exists, that scored RAE 0.13-0.24 on 5 of 6
seeds (the sink was left unconstrained, its edge in nobody's objective) where this rule
returns 1e-15 on 6 of 6. It also mis-selected the eigenvector outright on 2 of 30 cyclic GT
partitions at blockiness 0.8, and on most quotient-cyclic partitions that were valid but not
the generating one. Over 315 instances (p x k in {40x4, 100x10, 100x20, 200x20}, blockiness
{1.0, 0.8}, generating / 10%-perturbed / random partitions) the rule recovered exactness on
15 instances the pin missed and broke none of the 130 it had.

WHAT IT COSTS. The pin was closed form on a cyclic quotient; this searches. On fits that come
out exact the warm start plus `exact_tol` ends it after one restart, so the cost is ~2x
(median 0.0039 -> 0.0071 s). On an inexact fit all restarts run: median 0.0021 -> 0.471 s,
~184x, p90 2.0 s, worst 6 s at p=200. Callers that fit inside a search loop should weigh
that; `free_mode='min'` is cheaper and `n_restarts` is the dial.

WHAT IT BUYS ON INEXACT PARTITIONS. Median RAE 0.3776 -> 0.3440 (ratio 0.870, better on
72% of 185 fits, Wilcoxon on the log-ratio p=2e-08), median reconstruction 0.9732 -> 0.9601.
The MEAN is unchanged (0.4451 -> 0.4460): the quarter of fits it loses on lose by enough to
cancel it, p90 of the ratio 1.47 and one 8.6x outlier. The objective is collinearity
residual, not RAE, so lower residual is not guaranteed to be lower RAE.

PENALTY IS NOW THE TRUE DEFECT. The old `penalty` read exactly 0.000 on 98.8% of cyclic
quotients regardless of the disagreement actually present, since no SCC contributed residual.
It now equals sum_ij ||r_ij||^2 over every quotient edge, so it is a usable diagnostic.

THE PURE-CYCLE OBJECTIVE IS DEGENERATE, AND THE WARM START IS THE TIE-BREAK. On a terminal
SCC reached by a single chain of votes, EVERY eigenvector of Pi with nonzero eigenvalue is an
exact global minimum -- the residual cannot choose among them. `warm_start=True` starts the
seed at the `argmax |lambda|` eigenvector, so that empirical rule still decides the case the
objective is indifferent to, and it matched the old fitter to |cos| = 1.000000000000 there.
That rule remains measured, not proved, below blockiness 1.

STILL DIRECTIONAL. A propagated cluster takes the Frechet mean of its children's votes, which
is the exact least-squares fit of its DOWNWARD equations and ignores how it enters its
parents'. Freeing every cluster instead (`free_mode='all'`, or `polish=True`, which refines
the rule's answer with one warm-started restart) lowered the objective on 154 of 185 inexact
fits (median ratio 0.815) and was the only arm to improve the MEAN RAE as well
(0.4451 -> 0.4176, median 0.2980), for 1.3x the time of the rule alone. It is not the default
only because the default should be the cheapest thing that is correct.

SINKS ARE OPTIMISED, NOT SAMPLED (2026-09-26). The abstraction condition does not pin a
sink's direction, but the choice is not free of consequence: everything upstream is propagated
through M_ij t_j. At blockiness 1 the blocks are rank one, so M_ij t_j is parallel to the
block's left factor whatever t_j is (measured min |cos| = 1.0000 over 50 draws) and the sink
is irrelevant; at blockiness 0.8 the rank is 3 and that min |cos| falls to 0.1025. Sampling
sinks therefore fails outside the rank-one regime -- RAE 0.210 / 0.105 / 0.063 / 0.058 for
1 / 5 / 50 / 200 random draws against 2.2e-05 for an optimised fit -- so they are optimisation
variables here.
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


def _cycle_within(M, comp, ci, start, maxlen=None):
    """Shortest simple cycle through `start`, using only edges inside SCC `ci`.

    BFS over NODES with parent pointers, O(V + E). Every simple cycle through `start` is a
    path start -> ... -> v followed by the edge v -> start, so the shortest one runs along a
    shortest path to some in-neighbour of `start`. BFS pops in non-decreasing distance, so
    the first such v it reaches gives the shortest cycle; the BFS-tree path is simple and
    `start` appears only at its ends, so closing it keeps the cycle simple.

    Returns the cycle as [start, ..., v] (the closing edge v -> start is implied), or None
    when `start` is on no cycle inside `ci`.
    """
    from collections import deque
    par = {start: None}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in np.where(M[u])[0]:
            v = int(v)
            if comp[v] != ci:
                continue
            if v == start:
                path, w = [], u
                while w is not None:
                    path.append(w)
                    w = par[w]
                return path[::-1]
            if v not in par:
                par[v] = u
                q.append(v)
    return None


def _reconcile(votes, residuals=False):
    """Combine several collinearity votes for the same t, and report their disagreement.

    Each vote is only defined up to sign and scale, so they are unit-normalised and
    sign-aligned to the first before averaging. The residual after averaging is the
    disagreement the abstraction condition cannot satisfy -- the same quantity
    `fit_abstraction` minimises as its multi-child penalty.

    With residuals=True also returns the per-vote residual VECTORS (w - m), stacked. The
    penalty is their sum of squares, and a least-squares solver wants the vectors rather
    than that sum: squaring costs half the available precision. The stack always has one
    row per element of `votes`, degenerate votes contributing zeros, so its length depends
    only on the graph and stays fixed across calls -- which `least_squares` requires.
    """
    d = len(votes[0]) if votes else 0
    keep, u = [], []
    for n_, v in enumerate(votes):
        n = np.linalg.norm(v)
        if n > 1e-12:
            keep.append(n_)
            u.append(v / n)

    def _out(m, R):
        dis = float(np.sum(R ** 2))
        return (m, dis, R.ravel()) if residuals else (m, dis)

    if not u:
        return _out(None, np.zeros((len(votes), d)))
    ref = u[0]
    u = [w if w @ ref >= 0 else -w for w in u]
    m = np.mean(u, axis=0)
    n = np.linalg.norm(m)
    m = u[0] if n <= 1e-12 else m / n
    R = np.zeros((len(votes), d))
    for n_, w in zip(keep, u):
        R[n_] = w - m
    return _out(m, R)

def _reconcile_d(votes, dvotes):
    """`_reconcile` plus its derivative, for the analytic Jacobian.

    Mirrors `_reconcile` branch for branch -- same degeneracy thresholds, same sign
    reference, same fallbacks -- so the residual it returns is identical and only the
    derivative is new. Returns (m, dis, R_flat, dm, dR_flat), where dR_flat has one row per
    entry of R_flat and one column per parameter.

    Forward mode: each t depends on x only through its children, so one pass down the
    condensation order carries dt/dx alongside t. The sign alignment is piecewise constant,
    so it contributes nothing to the derivative away from a measure-zero set; the same goes
    for the two norm thresholds.
    """
    d = len(votes[0]) if votes else 0
    n_par = dvotes[0].shape[1] if dvotes else 0
    keep, u, du = [], [], []
    for n_, v in enumerate(votes):
        n = np.linalg.norm(v)
        if n > 1e-12:
            uj = v / n
            keep.append(n_)
            u.append(uj)
            du.append(((np.eye(d) - np.outer(uj, uj)) / n) @ dvotes[n_])
    R = np.zeros((len(votes), d))
    dR = np.zeros((len(votes), d, n_par))
    if not u:
        return None, 0.0, R.ravel(), np.zeros((d, n_par)), dR.reshape(-1, n_par)
    ref = u[0]
    sg = [1.0 if (w @ ref) >= 0 else -1.0 for w in u]
    u = [a * w for a, w in zip(sg, u)]
    du = [a * D for a, D in zip(sg, du)]
    m = np.mean(u, axis=0)
    dm = np.mean(du, axis=0)
    n = np.linalg.norm(m)
    if n <= 1e-12:
        m, dm = u[0], du[0]
    else:
        m, dm = m / n, ((np.eye(d) - np.outer(m / n, m / n)) / n) @ dm
    for idx, n_ in enumerate(keep):
        R[n_] = u[idx] - m
        dR[n_] = du[idx] - dm
    return m, float(np.sum(R ** 2)), R.ravel(), dm, dR.reshape(-1, n_par)


# s_i = (I - A_low[pi_i, pi_i])^-1 t_i lives in utils/abstraction_algebra.py, which carries the derivation
# and the local-inverse correction. Imported inside the fit so this numpy-only module does not
# pull torch in through that module, matching how scipy is imported below.


def fit_abstraction_eigen(A_low, partition, seed=0, n_restarts=5, maxiter=2000,
                          tol=1e-15, exact_tol=1e-28, analytic_jac=True,
                          free_mode='cycle', warm_start=True, polish=False, x0_T=None,
                          return_info=False):
    """Fit (A_high, T) by the rule in the module docstring.

    Returns (A_high, T, S), or (A_high, T, info) with return_info=True. `info` carries
    `penalty` (the true total collinearity defect at the optimum), `n_free` (seeds),
    `n_sinks`, `n_cyclic` (cyclic SCCs), `n_scc`, `n_edges` and `free_mode`.

    free_mode
        'cycle'  (default) a seed per sink and one per cyclic SCC -- the rule.
        'min'    seeds only where propagation has nothing below: one per TERMINAL component
                 of the condensation. Fewest parameters, and the minimum any propagation
                 scheme can have. A cycle reached from below is then determined entirely by
                 its downstream votes, and while its cycle-closing residual does still depend
                 on the seed -- through the whole chain of votes -- what the seed can REACH is
                 limited by the RANK of the blocks along that chain: t_i is confined to
                 col(M_ij), of dimension rank(A[pi_i, pi_j]), which is ONE at blockiness 1 and
                 3-6 below it, inside an ambient R^|pi_i|. Where the block is rank one,
                 M_ij t_j is parallel to its left factor whatever t_j is, so the seed moves
                 that cluster not at all. Measured on a 2-cycle above a sink: the cycle edge
                 sat at sqrt(2) over 4000 random sink directions with a spread of 2e-16
                 (col(M_02) one-dimensional and orthogonal to col(M_01), the eigenvector the
                 cycle wants at distance 1.0 from the reachable set), while the two multi-vote
                 edges swung over 0.006-0.765. Freeing a cluster in the cycle makes the whole
                 sphere reachable instead, which on that topology is worth 0.310 -> 0.263 in
                 median RAE and 5.36 -> 4.42 in objective over 36 inexact instances. This mode
                 still reaches machine precision on partitions that admit an exact abstraction
                 (36/36 there), since nothing then needs trading.
        'all'    every cluster free: the weakly-connected-component-symmetric fit, p
                 parameters, no privileged direction. Best paired with x0_T or `polish`.
    warm_start
        start the first restart with each cyclic seed at the `argmax |lambda|` eigenvector of
        its cycle product. Strictly an initialisation -- but see the degeneracy note in the
        module docstring, where it is also the tie-break.
    polish
        after the main fit, refine it with free_mode='all' and one warm-started restart,
        keeping whichever has the lower penalty. Measured better on both median and mean RAE.
    x0_T
        start from the columns of an existing T instead of the eigen warm start.

    `tol` and `maxiter` govern the search and set the accuracy floor of the whole fit. On a
    partition that admits an exact abstraction this reaches RAE ~1e-15, which is where the
    generating abstraction itself scores.
    """
    from collections import deque
    from scipy.optimize import least_squares

    A = np.asarray(A_low, dtype=float)
    p = sum(len(cluster) for cluster in partition)
    k = len(partition)

    if A.shape != (p, p):
        raise ValueError(f"A_low shape {A.shape} does not match partition size {p}.")
    if free_mode not in ('cycle', 'min', 'all'):
        raise ValueError(f"free_mode must be 'cycle', 'min' or 'all', got {free_mode!r}.")

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

    # every M_ij once; the forward pass reuses them on each call
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
    kids_of = {i: [int(j) for j in np.where(M[i])[0] if (i, int(j)) in Mb] for i in range(k)}
    edges = [(i, j) for i in range(k) for j in kids_of[i]]        # every quotient edge

    # ('seed', i) -> t_i is free; ('vote', i, kids) -> t_i determined from t_j of children.
    # Structural, so built once: the condensation is walked children-first, and inside an SCC
    # a member becomes votable as soon as one of its children is determined (the reverse of a
    # strongly connected graph is strongly connected, so one seed reaches every member).
    plan, done, seeds = [], [False] * k, []
    for c in order:
        mem = [int(i) for i in members[c]]
        if free_mode == 'all':
            for i in mem:
                plan.append(('seed', i))
                seeds.append(i)
                done[i] = True
            continue
        pending = set(mem)
        if free_mode == 'cycle' and cyclic[c]:
            i = min(pending)                     # one free seed inside every cycle
            plan.append(('seed', i))
            seeds.append(i)
            done[i] = True
            pending.discard(i)
        while pending:
            ready = [i for i in sorted(pending) if any(done[j] for j in kids_of[i])]
            if not ready:                        # nothing below: this component needs a seed
                i = min(pending)
                plan.append(('seed', i))
                seeds.append(i)
                done[i] = True
                pending.discard(i)
                continue
            for i in ready:
                plan.append(('vote', i, [j for j in kids_of[i] if done[j]]))
                done[i] = True
                pending.discard(i)

    sizes = [len(partition[i]) for i in seeds]
    offs = np.cumsum([0] + sizes)
    n_par = int(offs[-1])
    slot = {i: n_ for n_, i in enumerate(seeds)}

    def forward(x, residuals=False, want_jac=False):
        """Execute the plan, then score every quotient edge.

        Returns (t, penalty), or with the residual vector, or with it and the Jacobian.
        """
        t = [None] * k
        dt = [None] * k if want_jac else None
        for op in plan:
            if op[0] == 'seed':
                i = op[1]
                ci = len(partition[i])
                v = x[offs[slot[i]]:offs[slot[i] + 1]]
                nv = np.linalg.norm(v)
                if nv > 1e-12:
                    t[i] = v / nv
                    if want_jac:
                        D = np.zeros((ci, n_par))
                        D[:, offs[slot[i]]:offs[slot[i] + 1]] = \
                            (np.eye(ci) - np.outer(t[i], t[i])) / nv
                        dt[i] = D
                else:
                    t[i] = np.ones(ci) / np.sqrt(ci)
                    if want_jac:
                        dt[i] = np.zeros((ci, n_par))
                continue
            _, i, kids = op
            ci = len(partition[i])
            use = [j for j in kids if t[j] is not None]
            votes = [Mb[(i, j)] @ t[j] for j in use]
            if not votes:
                t[i] = np.ones(ci) / np.sqrt(ci)
                if want_jac:
                    dt[i] = np.zeros((ci, n_par))
                continue
            if want_jac:
                ti, _, _, dti, _ = _reconcile_d(votes, [Mb[(i, j)] @ dt[j] for j in use])
                dt[i] = dti if ti is not None else np.zeros((ci, n_par))
            else:
                ti = _reconcile(votes)[0]
            t[i] = ti if ti is not None else np.ones(ci) / np.sqrt(ci)

        pen, res, jac = 0.0, [], []
        for (i, j) in edges:
            ci = len(partition[i])
            r_ij = np.zeros(ci)
            d_ij = np.zeros((ci, n_par)) if want_jac else None
            if t[i] is not None and t[j] is not None:
                w = Mb[(i, j)] @ t[j]
                nw = np.linalg.norm(w)
                if nw > 1e-12:
                    u = w / nw
                    s = 1.0 if u @ t[i] >= 0 else -1.0      # the condition is sign-free
                    r_ij = s * u - t[i]
                    if want_jac:
                        d_ij = (s * ((np.eye(ci) - np.outer(u, u)) / nw)
                                @ (Mb[(i, j)] @ dt[j])) - dt[i]
            pen += float(r_ij @ r_ij)
            if residuals or want_jac:
                res.append(r_ij)
            if want_jac:
                jac.append(d_ij)
        if want_jac:
            return (t, pen, np.concatenate(res) if res else np.zeros(1),
                    np.concatenate(jac, axis=0) if jac else np.zeros((1, n_par)))
        if residuals:
            return t, pen, (np.concatenate(res) if res else np.zeros(1))
        return t, pen

    n_failed, last_error = 0, None
    if n_par == 0:
        best_x, best_pen = np.zeros(0), forward(np.zeros(0))[1]
    else:
        rng = np.random.default_rng(seed)
        x_warm = None
        if x0_T is not None:
            x_warm = rng.standard_normal(n_par)
            for i in seeds:
                col = np.asarray(x0_T)[partition[i], i]
                if np.linalg.norm(col) > 1e-12:
                    x_warm[offs[slot[i]]:offs[slot[i] + 1]] = col
        elif warm_start:
            # the cycle-product eigenvector, as a starting point and as the tie-break on a
            # pure cycle, where every eigenvector with nonzero eigenvalue is a global minimum
            x_warm = rng.standard_normal(n_par)
            for i in seeds:
                c = comp[i]
                if not cyclic[c]:
                    continue
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
                nv = np.linalg.norm(v)
                if nv > 1e-12:
                    x_warm[offs[slot[i]]:offs[slot[i] + 1]] = v / nv

        best_x, best_pen = None, np.inf
        for n_r in range(max(1, n_restarts)):
            x0 = x_warm if (n_r == 0 and x_warm is not None) else rng.standard_normal(n_par)
            try:
                out = least_squares(lambda z: forward(z, residuals=True)[2], x0,
                                    jac=(lambda z: forward(z, want_jac=True)[3])
                                        if analytic_jac else '2-point',
                                    method='trf', xtol=tol, ftol=tol, gtol=tol,
                                    max_nfev=maxiter)
                xc, pc = out.x, forward(out.x)[1]
            except Exception as e:
                n_failed += 1
                last_error = e
                xc, pc = x0, forward(x0)[1]
            if pc < best_pen:
                best_x, best_pen = xc, pc
            if best_pen <= exact_tol:
                break
        if best_x is None:
            best_x, best_pen = np.zeros(n_par), np.inf

    if n_par and n_failed == max(1, n_restarts):
        raise RuntimeError(
            f'the seed search failed on all {n_failed} restarts, so the fit would be a '
            f'random direction. Last error: {last_error!r}') from last_error

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

    if polish and free_mode != 'all' and penalty > exact_tol:
        # release every direction, starting from what the rule found
        Ah2, T2, info2 = fit_abstraction_eigen(
            A_low=A, partition=partition, seed=seed, n_restarts=1, maxiter=maxiter,
            tol=tol, exact_tol=exact_tol, analytic_jac=analytic_jac, free_mode='all',
            x0_T=T, return_info=True)
        if info2['penalty'] < penalty:
            A_high, T, penalty = Ah2, T2, info2['penalty']

    if return_info:
        return A_high, T, dict(penalty=penalty, n_failed=n_failed,
                               n_sinks=sum(1 for i in seeds if not kids_of[i]),
                               n_cyclic=sum(1 for c in range(ncomp) if cyclic[c]),
                               n_free=len(seeds), n_scc=ncomp, n_edges=len(edges),
                               free_mode=free_mode)

    from src.utils.abstraction_algebra import get_S
    return A_high, T, get_S(T=T, A_low=A, partition=partition)
