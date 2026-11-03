"""One-Child Closure (OCC) -- the complete method, in one place.

Given a low-level adjacency matrix, return a partition of its nodes that satisfies

  * All-In     -- for every high-level edge X->Y, EVERY member of X has an edge into Y
  * one-child  -- every cluster has at most one child cluster

which together are sufficient for a linear causal abstraction to exist.

ACYCLICITY IS NOT A CONDITION HERE, and the code to enforce it was REMOVED on 2026-09-26.
It is irrelevant to All-In and to one-child: neither says anything about whether the
quotient has a cycle. It was only ever imposed to suit the FITTER -- `fit_abstraction`
calls `enforce_dag` and deletes cycle edges before it starts, and `reverse_topo_order` has
no cycle detection -- and `eigen_solver.fit_abstraction_eigen` now solves cyclic quotients
in closed form, cycle members being pinned by an eigenvector equation.

Enforcing it was expensive: on cyclic low-level graphs it drove AOC Closure to a median
k_used of 1, with 58% of instances left holding NO quotient edges at all -- the empty
abstraction. Every csv in data/ predates the removal.

`conditions()` still REPORTS acyclicity as a third return value, and `info['acyclic']`
still carries it, because it remains a useful diagnostic. It is no longer a gate. Note
that several older scripts gate on `all(conditions(...))`, which does still require it;
those reproduce their own historical csvs and were left alone. On a weakly
connected low-level DAG the resulting high-level graph is necessarily a single IN-TREE
(out-degree <= 1, acyclic, one sink).

Two stages:

  1. CLOSURE. Start from singletons; whenever a cluster has more than one child, merge
     those children; repeat to a fixed point. Enforces one-child but NOT All-In, and is
     order-dependent -- so we draw several random orders and keep the best valid one.

  2. REPAIR (fallback). If no draw is valid, every violation has the form "node i in X has
     no edge into X's child Y", and we merge X with Y.

     NOTE this does NOT repair the violation locally: the merged cluster Z = X u Y inherits
     Y's child W, and members of X need not have edges into W, so a new violation can
     appear. The guarantee is weaker and comes from the loop, not the step:

       * termination -- each merge strictly decreases the cluster count, so at most k-1
         merges; in the worst case we reach the single-cluster partition, which satisfies
         All-In vacuously. The loop re-scans after every merge.
       * one-child is preserved -- Z's children are (ch(X) u ch(Y)) \ {X,Y} = {W}, and
         merging can only REDUCE the number of distinct children any other cluster sees.
       * acyclicity is preserved -- contracting X->Y in a DAG can create a cycle when a
         second path X->a->Y exists, but under one-child X's only child is Y, so no such
         path can exist.

     Measured over 38 instances that needed repair: at most 2 merges, k 39.6 -> 38.1, never
     collapsing to the trivial partition, all three conditions holding every time.

  3. BUDGET (optional, `k=`). The closure fixes its own width; pass k to coarsen further to
     EXACTLY k clusters using only merges that keep all three conditions (budget_coarsening).
     Any 1 <= k <= |closure| is reachable; a k above the closure width is not, since merges
     cannot refine.

Measured: stage 1 alone succeeds on ~83% of random instances (35-39% at density 0.1);
stage 2 lifts coverage to 1.000, needing 1-2 merges and ~30ms. Cost is ~p^1.6; 0.14s at
p=200, 2s at p=400. Stage 3 costs ~0.7s to halve k at p=200.

    from occ import occ, oc_closure
    groups, T, info = occ(A_low, n_draws=12, seed=0)          # both conditions
    groups, T, info = occ(A_low, n_draws=12, seed=0, k=20)    # ... at exactly k=20
    groups, T, info = oc_closure(A_low, n_draws=12, seed=0)  # one-child only

SELECTION CRITERIA (measured over 47 instances, p=100):

  * oc_closure picks the draw with LEAST INFORMATION LOSS. All draws are one-child, so all
    are eligible. This beats selecting on cluster count (0.542 -> 0.465 intra-cluster mass
    pooled, never worse, k unchanged), with the gain concentrated at mid-density.

  * occ keeps selecting the WIDEST ALL-IN-VALID draw, then repairs only if none is valid.
    Selecting those valid draws on information loss instead changes nothing (+0.000 median,
    2/47 instances), and repairing ALL draws before selecting changes nothing at all
    (identical on 47/47) for 28x the repair calls -- repair is confluent here, collapsing
    every draw onto the same partition. The draws differ only in where singletons land,
    and repair does not touch singletons.

  Redrawing therefore buys VALIDITY, not partition quality: at d=0.5 as few as 2 of 12
  draws satisfy All-In, and avoiding the repair stage is what n_draws is for.
"""
import numpy as np

__all__ = ['occ', 'oc_closure', 'one_child_closure', 'information_loss', 'conditions',
           'repair', 'budget_coarsening', 'high_level_graph']


def information_loss(A, groups):
    """Fraction of |A| edge mass falling INSIDE clusters, i.e. lost on aggregation.

    Costs sum_x |X_x|^2 <= p^2 -- microseconds at p=100, ~9% of the cost of generating one
    closure draw, so it is cheap enough to score every draw.
    """
    absA = np.abs(A)
    tot = max(absA.sum(), 1e-12)
    s = 0.0
    for g in groups:
        if len(g) > 1:
            sub = absA[np.ix_(g, g)]
            s += sub.sum() - np.trace(sub)
    return float(s / tot)


def oc_closure(A_low, n_draws=12, seed=0, return_T=True):
    """OC Closure: the one-child fixed point, selecting the draw with LEAST information loss.

    one_child_closure is order-dependent, so n_draws random orders give different
    partitions. All of them satisfy one-child by construction, so ALL draws are eligible
    here -- unlike occ(), which may only select among the All-In-valid ones.

    SELECTION CRITERION. Measured over 47 instances, selecting on information loss rather
    than on cluster count (the obvious proxy) lowers intra-cluster edge mass 0.542 -> 0.465
    pooled, is never worse, and leaves k unchanged. The gain is concentrated at mid-density
    (0.106 at d=0.5, 0.050 at d=0.7, 0.034 at d=0.3) and is zero at d=0.1 and d=0.9, where
    the draws are near-identical -- at d=0.9 typically 1-2 distinct partitions from 12,
    against up to 11 at d=0.5. Width is only a proxy: draws with the same k can differ
    substantially in how much edge mass they bury inside clusters.

    Note this enforces one-child ONLY -- never All-In. Use occ() for both conditions.
    Returns (groups, T, info) like occ().
    """
    A = np.asarray(A_low)
    p = A.shape[0]
    rng = np.random.default_rng(seed)
    best_g, best_loss = None, np.inf
    for _ in range(n_draws):
        g = one_child_closure(A, rng)
        loss = information_loss(A, g)
        if loss < best_loss:
            best_g, best_loss = g, loss
    groups = best_g
    ai, oc, dag = conditions(A, groups)
    T = np.zeros((p, len(groups)))
    for j, m in enumerate(groups):
        T[m, j] = 1.0
    info = dict(k=len(groups), allin=bool(ai), one_child=bool(oc), acyclic=bool(dag),
                information_loss=best_loss,
                cluster_sizes=sorted((len(g) for g in groups), reverse=True))
    return groups, (T if return_T else None), info


def high_level_graph(Ab, groups):
    """Boolean K x K quotient adjacency: edge x->y iff any member of x points into y."""
    K = len(groups)
    M = np.zeros((K, K), bool)
    for x in range(K):
        for y in range(K):
            if x != y and Ab[np.ix_(groups[x], groups[y])].any():
                M[x, y] = True
    return M


def conditions(A, groups):
    """(All-In holds, one-child holds, quotient is acyclic)."""
    Ab = (A != 0)
    M = high_level_graph(Ab, groups)
    K = len(groups)
    allin = all(all(Ab[i, groups[y]].any() for i in groups[x])
                for x in range(K) for y in np.where(M[x])[0])
    onechild = bool(M.sum(1).max() <= 1) if K else True
    seen = [0] * K
    ok = [True]

    def visit(u):
        if seen[u] == 1:
            ok[0] = False
            return
        if seen[u] == 2:
            return
        seen[u] = 1
        for v in np.where(M[u])[0]:
            visit(v)
        seen[u] = 2
    for u in range(K):
        visit(u)
    return allin, onechild, ok[0]


def one_child_closure(A, rng=None):
    """Stage 1: merge the children of any multi-child cluster, to a fixed point.

    Enforces one-child ONLY -- never All-In. Order-dependent, so `rng` gives a different
    feasible partition per draw. Returns a list of arrays of node indices.
    """
    rng = np.random.default_rng() if rng is None else rng
    p = A.shape[0]
    Ab = (A != 0)
    lab = list(range(p))

    def find(x):
        while lab[x] != x:
            lab[x] = lab[lab[x]]
            x = lab[x]
        return x

    changed = True
    while changed:
        changed = False
        blocks = {}
        for i in range(p):
            blocks.setdefault(find(i), []).append(i)
        roots = list(blocks)
        rng.shuffle(roots)
        for r in roots:
            mem = blocks.get(r)
            if mem is None:
                continue
            kids = {find(j) for i in mem for j in np.where(Ab[i])[0] if find(j) != find(r)}
            if len(kids) > 1:
                kl = sorted(kids)
                rng.shuffle(kl)
                for o in kl[1:]:
                    a, b = find(kl[0]), find(o)
                    if a != b:
                        lab[b] = a
                changed = True
    out = {}
    for i in range(p):
        out.setdefault(find(i), []).append(i)
    return [np.array(v) for v in out.values()]


def _first_violation(Ab, groups):
    M = high_level_graph(Ab, groups)
    for x in range(len(groups)):
        for y in np.where(M[x])[0]:
            for i in groups[x]:
                if not Ab[i, groups[y]].any():
                    return x, y
    return None


def repair(A, groups, max_iter=10_000):
    """Stage 2: merge a violating cluster with its child, repeatedly, until All-In holds.

    A single merge does NOT guarantee All-In -- the merged cluster inherits the child's
    child, which its members may not reach. Correctness comes from the loop: each merge
    strictly decreases the cluster count, so this terminates within k-1 merges, at worst at
    the single-cluster partition. One-child and acyclicity are preserved by each merge (see
    the module docstring). Returns (groups, n_merges).
    """
    Ab = (A != 0)
    groups = [np.asarray(g) for g in groups]
    for it in range(max_iter):
        v = _first_violation(Ab, groups)
        if v is None:
            return groups, it
        x, y = v
        merged = np.concatenate([groups[x], groups[y]])
        groups = [g for j, g in enumerate(groups) if j not in (x, y)] + [merged]
    raise RuntimeError('repair did not converge')


def budget_coarsening(A_low, groups, k, rng=None):
    """Coarsen a valid partition down to EXACTLY k clusters, preserving all three conditions.

    Unlike `repair`, which merges to repair a violation and may overshoot, every
    merge here is required to leave All-In, one-child AND acyclicity intact, so the loop can
    be stopped at any width. A merge of X and Y is admissible when, writing
    W = (ch(X) u ch(Y)) \\ {X, Y} for the merged cluster's children:

      * |W| <= 1                                               (one-child survives)
      * if W = {W}, every member of X u Y has an edge into W    (All-In survives)

    Nothing else can break: a cluster U that pointed into X now points into Z = X u Y, and
    All-In for U is a statement about edges into a SUPERSET, so it still holds; and merging
    can only reduce the number of distinct children any other cluster sees.

    EXACT k IS ALWAYS REACHABLE for 1 <= k <= len(groups). While K >= 2 an admissible merge
    always exists: if some sink S has a parent X, then merging X with S gives W = {} and both
    conditions are vacuous; and if no sink has a parent then every cluster is an isolated
    sink (a non-sink's child chain ends at a sink with a parent), so any pair merges with
    W = {}. Each merge drops K by one, so the loop lands on k rather than passing it.

    Among the admissible merges we take the one burying the least edge mass inside a cluster,
    i.e. the smallest increase in information_loss, `rng` breaking ties.

    The three K x K tables (reachability counts, edge mass, membership) are carried across
    merges rather than rebuilt, since a merge only sums two rows and two columns; that keeps
    the per-merge cost at O(pK + K^2). Measured at p=200, 73 -> 36 clusters: 0.7 s, against
    6.5 s when the tables are rebuilt each iteration, same partition.

    Returns (groups, n_merges). Raises if `groups` is not one-child to begin with, or if
    All-In fails so badly on the input that no admissible merge exists.
    """
    A = np.asarray(A_low)
    absA = np.abs(A)
    if k < 1:
        raise ValueError(f'budget k must be at least 1, got {k}')
    groups = [np.asarray(g, dtype=int) for g in groups]
    if len(groups) <= k:
        return groups, 0
    p = A.shape[0]

    T = np.zeros((p, len(groups)))
    for j, g in enumerate(groups):
        T[g, j] = 1.0
    reach = ((A != 0).astype(float) @ T) > 0   # (p, K): node i has an edge into cluster y
    C = T.T @ reach                            # (x, y): members of x with an edge into y
    cross = T.T @ absA @ T                     # edge mass between clusters
    sizes = T.sum(0)

    merges = 0
    while len(groups) > k:
        K = len(groups)
        off = C.copy()
        np.fill_diagonal(off, 0)
        M = off > 0
        full = off == sizes[:, None]           # EVERY member of x has an edge into y
        nkids = M.sum(1)
        if nkids.max() > 1:
            raise ValueError('budget_coarsening expects a one-child partition; cluster '
                             f'{int(nkids.argmax())} has {int(nkids.max())} children')
        child = np.where(nkids == 1, M.argmax(1), -1)
        ys = np.arange(K)
        best_key, best_pair = None, None
        for x in range(K):
            y = ys[ys != x]
            cx, cy = child[x], child[y]
            x_out = (cx != -1) & (cx != y)      # X keeps its child after the merge
            y_out = (cy != -1) & (cy != x)      # Y keeps its child after the merge
            n_kids = x_out.astype(int) + y_out.astype(int) - (x_out & y_out & (cx == cy))
            ok = n_kids <= 1
            w = np.where(x_out, cx, np.where(y_out, cy, -1))
            has_w = w >= 0
            wi = np.where(has_w, w, 0)
            ok &= ~has_w | (full[x, wi] & full[y, wi])
            if not ok.any():
                continue
            score = cross[x, y] + cross[y, x]
            if rng is not None:
                score = score + rng.random(len(y)) * 1e-12
            j = int(np.argmin(np.where(ok, score, np.inf)))
            if best_key is None or score[j] < best_key:
                best_key, best_pair = score[j], (x, int(y[j]))
        if best_pair is None:
            raise RuntimeError(f'no condition-preserving merge available at k={K}; the input '
                               'partition is not All-In valid')

        # merge x and y into a cluster appended at the end, and fold the tables the same way
        x, y = best_pair
        keep = np.array([j for j in range(K) if j not in (x, y)], dtype=int)
        groups = [groups[j] for j in keep] + [np.concatenate([groups[x], groups[y]])]
        T = np.column_stack([T[:, keep], T[:, x] + T[:, y]])
        reach = np.column_stack([reach[:, keep], reach[:, x] | reach[:, y]])
        # members of Z reaching a is additive (X and Y are disjoint); members of a reaching Z
        # is NOT (a node may reach both), so that column is recounted from `reach`.
        C_col = T.T @ reach[:, -1].astype(float)
        C_row = np.append((C[x] + C[y])[keep], C_col[-1])
        C = np.vstack([np.column_stack([C[np.ix_(keep, keep)], C_col[:-1]]), C_row])
        cross = np.vstack([
            np.column_stack([cross[np.ix_(keep, keep)], (cross[:, x] + cross[:, y])[keep]]),
            np.append((cross[x] + cross[y])[keep],
                       cross[x, x] + cross[y, y] + cross[x, y] + cross[y, x])])
        sizes = np.append(sizes[keep], sizes[x] + sizes[y])
        merges += 1
    return groups, merges


def occ(A_low, n_draws=12, seed=0, return_T=True, k=None):
    """The complete method (AOC Closure). Returns (groups, T, info).

    `k`, if given, is a CLUSTER BUDGET: after the closure (and repair, if needed) the
    partition is coarsened by condition-preserving merges until it has exactly k clusters
    (see budget_coarsening). Any 1 <= k <= |closure| is reachable, and all three conditions
    still hold at k. A budget ABOVE the closure width cannot be met -- merges cannot refine a
    partition -- and is reported as budget_met=False rather than raising.

    groups : list of arrays of low-level node indices, one per cluster
    T      : (p, k) hard assignment matrix, or None if return_T is False
    info   : dict -- k, valid_draws, repaired, n_merges, allin, one_child, acyclic,
             information_loss (fraction of |A| edge mass falling inside clusters), plus
             budget, budget_met, k_closure and budget_merges when a budget is given
    """
    A = np.asarray(A_low)
    Ab = (A != 0)
    p = A.shape[0]
    rng = np.random.default_rng(seed)

    best, best_g, widest, widest_g, n_ok = 0, None, 0, None, 0
    for _ in range(n_draws):
        g = one_child_closure(A, rng)
        if len(g) > widest:
            widest, widest_g = len(g), g
        ai, oc, _ = conditions(A, g)
        if ai and oc:
            n_ok += 1
            if len(g) > best:
                best, best_g = len(g), g

    if best_g is not None:
        groups, merges, repaired = best_g, 0, False
    else:
        groups, merges = repair(A, widest_g)
        repaired = True

    extra = {}
    if k is not None:
        k_closure = len(groups)
        b_merges = 0
        if k_closure > k:
            groups, b_merges = budget_coarsening(A, groups, k, rng=rng)
        extra = dict(budget=k, budget_met=len(groups) == k, k_closure=k_closure,
                     budget_merges=b_merges)

    ai, oc, dag = conditions(A, groups)
    T = np.zeros((p, len(groups)))
    for j, m in enumerate(groups):
        T[m, j] = 1.0
    co = T @ T.T
    np.fill_diagonal(co, 0)
    info = dict(k=len(groups), valid_draws=n_ok, repaired=repaired, n_merges=merges,
                allin=bool(ai), one_child=bool(oc), acyclic=bool(dag),
                information_loss=float((np.abs(A) * co).sum() / max(np.abs(A).sum(), 1e-12)),
                cluster_sizes=sorted((len(g) for g in groups), reverse=True), **extra)
    return groups, (T if return_T else None), info
