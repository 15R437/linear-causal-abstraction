"""Estimate the largest k admitting a partition satisfying one-child AND All-In.

Brute force is Bell(p), so only p<=11 is enumerable. This estimates a LOWER BOUND by
(a) running the one-child merge closure from many random orders and (b) greedily splitting
clusters wherever a split keeps both conditions. Validated against brute force at p=10.
"""
import numpy as np


def conditions(A, groups):
    """(All-In holds, one-child holds, high-level graph acyclic)."""
    Ab = (A != 0)
    K = len(groups)
    M = np.zeros((K, K), bool)
    for a in range(K):
        for b in range(K):
            if a != b and Ab[np.ix_(groups[a], groups[b])].any():
                M[a, b] = True
    allin = all(all(Ab[i, groups[b]].any() for i in groups[a])
                for a in range(K) for b in np.where(M[a])[0])
    onechild = bool(M.sum(1).max() <= 1) if K else True
    seen = [0] * K
    ok = [True]

    def vis(u):
        if seen[u] == 1:
            ok[0] = False
            return
        if seen[u] == 2:
            return
        seen[u] = 1
        for v in np.where(M[u])[0]:
            vis(v)
        seen[u] = 2
    for u in range(K):
        vis(u)
    return allin, onechild, ok[0]


def closure(A, rng):
    """Finest one-child partition reachable from singletons by forced merges."""
    p = A.shape[0]
    Ab = (A != 0)
    lab = list(range(p))

    def find(x):
        while lab[x] != x:
            lab[x] = lab[lab[x]]
            x = lab[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            lab[ry] = rx

    changed = True
    while changed:
        changed = False
        groups = {}
        for i in range(p):
            groups.setdefault(find(i), []).append(i)
        roots = list(groups)
        rng.shuffle(roots)
        for r in roots:
            mem = groups.get(r)
            if mem is None:
                continue
            kids = {find(j) for i in mem for j in np.where(Ab[i])[0] if find(j) != find(r)}
            if len(kids) > 1:
                kl = sorted(kids)
                rng.shuffle(kl)
                for o in kl[1:]:
                    union(kl[0], o)
                changed = True
    out = {}
    for i in range(p):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def refine_by_splitting(A, groups, rng, tries=40):
    """Greedily split clusters while both conditions survive, raising k."""
    groups = [list(g) for g in groups]
    improved = True
    while improved:
        improved = False
        order = sorted(range(len(groups)), key=lambda j: -len(groups[j]))
        for j in order:
            mem = groups[j]
            if len(mem) < 2:
                continue
            for _ in range(tries):
                cut = rng.integers(1, len(mem))
                perm = rng.permutation(mem)
                a, b = list(perm[:cut]), list(perm[cut:])
                cand = groups[:j] + [a, b] + groups[j + 1:]
                ai, oc, dag = conditions(A, cand)
                if ai and oc and dag:
                    groups = cand
                    improved = True
                    break
            if improved:
                break
    return groups


def estimate_max_k(A, n_orders=30, seed=0, refine=True):
    rng = np.random.default_rng(seed)
    best = 0
    best_groups = None
    for _ in range(n_orders):
        g = closure(A, rng)
        ai, oc, dag = conditions(A, g)
        if not (ai and oc and dag):
            # the closure enforces one-child only; drop draws that break All-In
            continue
        if refine:
            g = refine_by_splitting(A, g, rng)
        if len(g) > best:
            best, best_groups = len(g), g
    return best, best_groups
