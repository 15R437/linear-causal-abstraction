"""Graph-partition baselines for the All-In / one-child conditions.

WHY THESE ALGORITHMS. All-In holds for cluster X iff every member of X has the SAME
child-cluster signature (if any member points into Y, all must). That is a regular/
structural-equivalence condition -- bisimulation -- NOT a community-detection one.
Community detection (modularity, normalised cut, METIS) MAXIMISES intra-cluster edge
mass, which is precisely the information loss being plotted, so those methods optimise
the wrong sign of the y-axis. `spectral-ncut` is kept as a negative control to show that.

THE k BUDGET IS LOAD-BEARING. Singletons satisfy All-In vacuously AND have zero
intra-cluster mass, so the finest partition is a trivial optimum of both axes. Every
method here is therefore given the SAME budget K (= AOC Closure's own cluster count on
that instance), which is what makes the comparison non-trivial.

Methods
  bisim        partition refinement toward All-In, stopped at the budget (principled)
  sig-agglom   average-linkage on Jaccard distance between out-neighbourhoods
  sig-kmeans   k-means on the binary out-neighbourhood rows
  spectral-ncut  normalised cut on the symmetrised |A|  -- NEGATIVE CONTROL
  OCC          AOC Closure (one-child fixed point + All-In repair)
  UCA, random  existing references
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans, SpectralClustering


def _to_T(lab, p):
    u = {c: j for j, c in enumerate(sorted(set(lab)))}
    T = np.zeros((p, len(u)))
    for i, c in enumerate(lab):
        T[i, u[c]] = 1.0
    return T


def high_level_mask(Ab, lab):
    K = lab.max() + 1
    M = np.zeros((K, K), bool)
    for x in range(K):
        gx = np.where(lab == x)[0]
        if not len(gx):
            continue
        reach = Ab[gx].any(0)
        for y in range(K):
            if x != y and reach[lab == y].any():
                M[x, y] = True
    return M


def bisim(A_low, k):
    """Refine toward All-In: split each block by its members' child-block signatures.

    At the fixed point every block is signature-uniform, i.e. All-In holds. Stopped at the
    budget k, so what is measured is how close refinement gets under that ceiling. Splits
    the block with the most distinct signatures first (greedy: largest violation removed
    per split), and re-derives signatures after every split because splitting one block
    changes the signature of every block pointing into it.
    """
    Ab = (A_low != 0)
    p = Ab.shape[0]
    lab = np.zeros(p, int)
    while lab.max() + 1 < k:
        nb = lab.max() + 1
        best, best_sig, best_n = None, None, 1
        for x in range(nb):
            idx = np.where(lab == x)[0]
            if len(idx) < 2:
                continue
            sig = {}
            for i in idx:
                # MULTISET of out-neighbour blocks, own block INCLUDED. The literal
                # All-In signature (distinct OTHER blocks) cannot bootstrap: from a
                # single block every node's signature is empty and refinement stops
                # at k=1. The multiset splits by out-degree on the first pass, and
                # its fixed point still implies All-In (equal multisets => equal
                # child sets). This is colour/equitable-partition refinement.
                s = tuple(sorted(int(lab[j]) for j in np.where(Ab[i])[0]))
                sig.setdefault(s, []).append(i)
            if len(sig) > best_n:
                best, best_sig, best_n = x, sig, len(sig)
        if best is None:
            break                                   # signature-uniform: All-In reached
        groups = list(best_sig.values())
        room = k - (lab.max() + 1)
        # keep the largest group in place; peel off the rest while the budget allows
        groups.sort(key=len, reverse=True)
        for g in groups[1:1 + room]:
            lab[g] = lab.max() + 1
    return _to_T(lab, p)


def _outsim(Ab):
    """Jaccard similarity between out-neighbourhoods (structural equivalence)."""
    X = Ab.astype(float)
    inter = X @ X.T
    deg = X.sum(1)
    union = deg[:, None] + deg[None, :] - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        S = np.where(union > 0, inter / np.maximum(union, 1e-12), 1.0)
    np.fill_diagonal(S, 1.0)
    return S


def sig_agglom(A_low, k):
    Ab = (A_low != 0)
    D = 1.0 - _outsim(Ab)
    np.fill_diagonal(D, 0.0)
    D = np.clip((D + D.T) / 2, 0, None)
    Z = linkage(squareform(D, checks=False), method='average')
    return _to_T(fcluster(Z, t=k, criterion='maxclust') - 1, Ab.shape[0])


def sig_kmeans(A_low, k, seed=0):
    Ab = (A_low != 0).astype(float)
    lab = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(Ab)
    return _to_T(lab, Ab.shape[0])


def spectral_ncut(A_low, k, seed=0):
    """NEGATIVE CONTROL: minimises the cut, i.e. MAXIMISES intra-cluster edge mass."""
    W = np.abs(A_low); W = (W + W.T) / 2
    W = W + 1e-9
    lab = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=seed,
                             assign_labels='kmeans', n_init=4).fit_predict(W)
    return _to_T(lab, A_low.shape[0])


def violations(A_low, T_hard):
    """All-In and one-child violation counts for a hard assignment."""
    Ab = (A_low != 0)
    lab = T_hard.argmax(1)
    keep = sorted(set(lab.tolist()))
    remap = {c: j for j, c in enumerate(keep)}
    lab = np.array([remap[c] for c in lab])
    K = len(keep)
    M = high_level_mask(Ab, lab)
    bad_nodes, bad_pairs = set(), 0
    for x in range(K):
        gx = np.where(lab == x)[0]
        for y in np.where(M[x])[0]:
            gy = np.where(lab == y)[0]
            for i in gx:
                if not Ab[i, gy].any():
                    bad_nodes.add(int(i)); bad_pairs += 1
    nch = M.sum(1)
    return dict(k_used=K,
                allin_nodes=len(bad_nodes), allin_pairs=int(bad_pairs),
                oc_clusters=int((nch > 1).sum()), oc_excess=int(np.maximum(nch - 1, 0).sum()),
                allin_ok=int(len(bad_nodes) == 0), oc_ok=int(nch.max() <= 1 if K else 1))


def _violation_count(Ab, lab, k):
    """All-In violation PAIRS: sum over quotient edges x->y of #members of x missing it."""
    p = Ab.shape[0]
    B = np.empty((p, k), bool)
    for y in range(k):
        B[:, y] = Ab[:, lab == y].any(1)
    N = np.empty((k, k), int)
    for x in range(k):
        N[x] = B[lab == x].sum(0)
    s = np.bincount(lab, minlength=k)
    Tm = (N > 0) * (s[:, None] - N)
    np.fill_diagonal(Tm, 0)
    return int(Tm.sum())


def allin_agglom(A_low, k, n_cand=120):
    """Greedy agglomerative clustering whose merge criterion IS the All-In violation count.

    From singletons (violation count 0, trivially All-In) merge down to k, at each step
    choosing the merge that leaves the fewest All-In violations, breaking ties by the
    smaller increase in intra-cluster edge mass.

    Evaluation of a candidate is EXACT -- the violation count is recomputed in full -- but
    the candidate SET is shortlisted to the `n_cand` most out-neighbourhood-similar pairs,
    because scoring all O(k^2) pairs exactly at every one of the p-k steps is O(p^2 k^3).
    Similarity shortlisting is aligned with the objective rather than neutral: All-In is a
    structural-equivalence condition, so similar out-neighbourhoods are exactly where the
    violation-free merges live.
    """
    Ab = (A_low != 0)
    p = Ab.shape[0]
    absA = np.abs(A_low)
    lab = np.arange(p)
    sim = _outsim(Ab)
    nxt = p
    while len(set(lab.tolist())) > k:
        ids = np.array(sorted(set(lab.tolist())))
        kk = len(ids)
        pos = {c: j for j, c in enumerate(ids)}
        cl = np.array([pos[c] for c in lab])
        # cluster-level similarity = mean pairwise node similarity between the blocks
        M = np.empty((kk, p))
        for x in range(kk):
            M[x] = sim[cl == x].mean(0)
        Csim = np.empty((kk, kk))
        for y in range(kk):
            Csim[:, y] = M[:, cl == y].mean(1)
        np.fill_diagonal(Csim, -np.inf)
        iu = np.triu_indices(kk, 1)
        vals = Csim[iu]
        take = min(n_cand, len(vals))
        top = np.argpartition(-vals, take - 1)[:take]
        best, best_key = None, None
        for t in top:
            a, b = int(iu[0][t]), int(iu[1][t])
            trial = cl.copy(); trial[trial == b] = a
            u = np.unique(trial); trial = np.searchsorted(u, trial)
            v = _violation_count(Ab, trial, len(u))
            ma, mb = cl == a, cl == b
            add = float(absA[np.ix_(np.where(ma)[0], np.where(mb)[0])].sum()
                        + absA[np.ix_(np.where(mb)[0], np.where(ma)[0])].sum())
            key = (v, add)
            if best_key is None or key < best_key:
                best, best_key = (a, b), key
        if best is None:
            break
        a, b = best
        lab[cl == b] = ids[a]
    return _to_T(lab, p)


def _cond_scores(Ab, absA, tot, lab, k):
    """(All-In violation pairs, one-child excess child-edges, intra-cluster mass fraction)."""
    p = Ab.shape[0]
    B = np.empty((p, k), bool)
    for y in range(k):
        B[:, y] = Ab[:, lab == y].any(1)
    N = np.empty((k, k), int)
    for x in range(k):
        N[x] = B[lab == x].sum(0)
    s = np.bincount(lab, minlength=k)
    Tm = (N > 0) * (s[:, None] - N)
    np.fill_diagonal(Tm, 0)
    M = N > 0
    np.fill_diagonal(M, False)
    nch = M.sum(1)
    intra = 0.0
    for x in range(k):
        idx = np.where(lab == x)[0]
        if len(idx) > 1:
            intra += absA[np.ix_(idx, idx)].sum() - absA[idx, idx].sum()
    return int(Tm.sum()), int(np.maximum(nch - 1, 0).sum()), intra / tot


def greedy_merge(A_low, k, w_allin=1.0, w_oc=0.0, w_intra=0.0, n_cand=120):
    """Greedy agglomerative clustering on a scalarised condition objective.

        J = w_allin * (All-In violation pairs)
          + w_oc    * (one-child excess child-edges)
          + w_intra * p * (intra-cluster edge mass fraction)

    intra is multiplied by p so that all three terms are on a comparable count scale
    (violations and excess edges run to a few hundred; the mass fraction is in [0,1]).

    Same structure as allin_agglom: exact evaluation of every candidate, over a shortlist
    of the n_cand most out-neighbourhood-similar cluster pairs.
    """
    Ab = (A_low != 0)
    p = Ab.shape[0]
    absA = np.abs(A_low)
    tot = max(absA.sum(), 1e-12)
    lab = np.arange(p)
    sim = _outsim(Ab)
    while len(set(lab.tolist())) > k:
        ids = np.array(sorted(set(lab.tolist())))
        kk = len(ids)
        pos = {c: j for j, c in enumerate(ids)}
        cl = np.array([pos[c] for c in lab])
        M = np.empty((kk, p))
        for x in range(kk):
            M[x] = sim[cl == x].mean(0)
        Csim = np.empty((kk, kk))
        for y in range(kk):
            Csim[:, y] = M[:, cl == y].mean(1)
        np.fill_diagonal(Csim, -np.inf)
        iu = np.triu_indices(kk, 1)
        vals = Csim[iu]
        take = min(n_cand, len(vals))
        top = np.argpartition(-vals, take - 1)[:take]
        best, best_key = None, None
        for t in top:
            a, b = int(iu[0][t]), int(iu[1][t])
            trial = cl.copy(); trial[trial == b] = a
            u = np.unique(trial); trial = np.searchsorted(u, trial)
            ai, oc, intra = _cond_scores(Ab, absA, tot, trial, len(u))
            J = w_allin * ai + w_oc * oc + w_intra * p * intra
            if best_key is None or J < best_key:
                best, best_key = (a, b), J
        if best is None:
            break
        a, b = best
        lab[cl == b] = ids[a]
    return _to_T(lab, p)

def _allin_cost(Ab, lab, k, mode='pairs'):
    """All-In violation cost under three weightings of the same violations.

    For each quotient edge X_x -> X_y let m_xy = #{i in X_x with no edge into X_y}.
      'pairs'    sum_xy m_xy              -- (node, child-cluster) incidences. Scales with
                                             |X_x|, so large-cluster violations cost MORE.
      'nodes'    #{i violating for ANY y} -- distinct nodes; a node failing 3 children
                                             counts once. Size-neutral.
      'dilution' sum_xy m_xy / |X_x|      -- the share of cluster weight that fails to
                                             participate, i.e. ~ the aggregation error a
                                             violation induces. Large-cluster violations
                                             cost LESS.
    """
    p = Ab.shape[0]
    B = np.empty((p, k), bool)
    for y in range(k):
        B[:, y] = Ab[:, lab == y].any(1)
    N = np.empty((k, k), int)
    for x in range(k):
        N[x] = B[lab == x].sum(0)
    s = np.bincount(lab, minlength=k)
    miss = (N > 0) * (s[:, None] - N)
    np.fill_diagonal(miss, 0)
    if mode == 'pairs':
        return float(miss.sum())
    if mode == 'dilution':
        return float((miss / np.maximum(s, 1)[:, None]).sum())
    if mode == 'nodes':
        bad = np.zeros(p, bool)
        for x in range(k):
            gx = np.where(lab == x)[0]
            for y in np.where((N[x] > 0))[0]:
                if y == x:
                    continue
                bad[gx[~B[gx, y]]] = True
        return float(bad.sum())
    raise ValueError(mode)


def allin_agglom_mode(A_low, k, mode='pairs', n_cand=120):
    """allin_agglom with a selectable violation weighting (see _allin_cost)."""
    Ab = (A_low != 0)
    p = Ab.shape[0]
    absA = np.abs(A_low)
    lab = np.arange(p)
    sim = _outsim(Ab)
    while len(set(lab.tolist())) > k:
        ids = np.array(sorted(set(lab.tolist())))
        kk = len(ids)
        pos = {c: j for j, c in enumerate(ids)}
        cl = np.array([pos[c] for c in lab])
        M = np.empty((kk, p))
        for x in range(kk):
            M[x] = sim[cl == x].mean(0)
        Csim = np.empty((kk, kk))
        for y in range(kk):
            Csim[:, y] = M[:, cl == y].mean(1)
        np.fill_diagonal(Csim, -np.inf)
        iu = np.triu_indices(kk, 1)
        vals = Csim[iu]
        take = min(n_cand, len(vals))
        top = np.argpartition(-vals, take - 1)[:take]
        best, best_key = None, None
        for t in top:
            a, b = int(iu[0][t]), int(iu[1][t])
            trial = cl.copy(); trial[trial == b] = a
            u = np.unique(trial); trial = np.searchsorted(u, trial)
            v = _allin_cost(Ab, trial, len(u), mode)
            ma, mb = np.where(cl == a)[0], np.where(cl == b)[0]
            addm = float(absA[np.ix_(ma, mb)].sum() + absA[np.ix_(mb, ma)].sum())
            key = (v, addm)
            if best_key is None or key < best_key:
                best, best_key = (a, b), key
        if best is None:
            break
        a, b = best
        lab[cl == b] = ids[a]
    return _to_T(lab, p)
