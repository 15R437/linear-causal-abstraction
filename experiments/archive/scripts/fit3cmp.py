"""fit_abstraction vs fit_abstraction_alternative vs fit_abstraction_K.

Ground-truth partitions (acyclic and cyclic generating graphs) + random partitions as
a control. Measures abstraction error (RAE), T-faithfulness, and wall-clock time.
"""
import os, sys, time, warnings
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch, multiprocessing as mp
from src.utils.generate_data import (sample_low_level_graph, generate_random_T,
                                     T_concretisation_sampling)
from src.utils.helper import (fit_abstraction, fit_abstraction_alternative,
                              fit_abstraction_K, get_S, cluster_indices, high_level_mask)
from src.utils.losses import RAE

OUT = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/fit3cmp.csv'
NRESTARTS = 10


# ---------------------------------------------------------------- generation
def cyclic_A_high(k, density, rho, rng):
    """Random high-level graph that is guaranteed to contain a cycle, rescaled to rho."""
    for _ in range(500):
        M = (rng.random((k, k)) < density).astype(float)
        np.fill_diagonal(M, 0.)
        A = M * rng.uniform(0.5, 2.0, (k, k)) * rng.choice([-1, 1], (k, k))
        r = np.max(np.abs(np.linalg.eigvals(A)))
        if r < 1e-9:
            continue
        R = (A != 0); Rc = R.copy()
        for _ in range(k):
            Rc = Rc | (Rc @ R)
        if not np.any(np.diag(Rc)):
            continue
        return A * (rho / r)
    return None


def gen_instance(kind, p, k, density, seed):
    """Returns A_low, T_gt, A_high_gt. Both kinds are EXACT abstractions by construction."""
    if kind == 'acyclic':
        A_low, T_gt, _, A_high = sample_low_level_graph(
            n_high=k, n_low=p, density=density, shuffle=True, blockiness=0.5,
            heterogeneity=1., internal_density=0., graph_type='random', seed=seed)
        return A_low, T_gt, A_high
    rng = np.random.default_rng(seed)
    A_high = cyclic_A_high(k, density, 0.7, rng)
    if A_high is None:
        return None
    T_gt, _ = generate_random_T(p, k, (1., -2.), (2., -1.), seed=seed)
    A_low, _ = T_concretisation_sampling(A_high=A_high, T=T_gt, internal_density=0.,
                                         blockiness=0.5, heterogeneity=1., seed=seed)
    perm = rng.permutation(p)
    return A_low[perm][:, perm], T_gt[perm], A_high


def random_partition(p, k, rng):
    """Balanced-ish random assignment with every cluster non-empty."""
    lab = np.concatenate([np.arange(k), rng.integers(0, k, p - k)])
    rng.shuffle(lab)
    T = np.zeros((p, k))
    T[np.arange(p), lab] = 1.
    return T


# ---------------------------------------------------------------- metrics
def get_S_local(T, T_hard, A_low):
    """S_ii = (I - A_low[pi_i, pi_i])^-1 t_i.

    This is what the abstraction equation (I-A_low)S = T(I-A_high) forces. `get_S` uses
    E_low[pi_i, pi_i] instead, which equals it only when the QUOTIENT is acyclic -- a
    path that leaves cluster i and returns contributes to the full inverse but not to
    the local one.
    """
    idx = cluster_indices(T_hard)
    S = np.zeros_like(T)
    for i in range(T.shape[1]):
        if len(idx[i]) == 0:
            continue
        Aii = A_low[np.ix_(idx[i], idx[i])]
        S[idx[i], i] = np.linalg.solve(np.eye(len(idx[i])) - Aii, T[idx[i], i])
    return S


def block_structure(A_low, T_hard):
    """(k,k) boolean: block[i,j] iff any low-level edge from cluster j into cluster i."""
    idx = cluster_indices(T_hard)
    k = T_hard.shape[1]
    M = np.zeros((k, k), bool)
    for i in range(k):
        for j in range(k):
            if i != j and len(idx[i]) and len(idx[j]):
                M[i, j] = bool(np.any(A_low[np.ix_(idx[i], idx[j])]))
    return M


def is_cyclic(M):
    k = M.shape[0]
    R = M.copy()
    for _ in range(k):
        R = R | (R @ M)
    return bool(np.any(np.diag(R)))


def score(name, A_high, T, A_low, E_low, T_hard, block, t):
    k = T.shape[1]
    r = {f'{name}_time': t}
    if A_high is None or not np.all(np.isfinite(A_high)):
        r[f'{name}_failed'] = 1
        return r
    r[f'{name}_failed'] = 0
    for tag, S in (('loc', get_S_local(T, T_hard, A_low)),
                   ('repo', get_S(T=T, E_low=E_low))):
        try:
            r[f'{name}_rae_{tag}'] = float(RAE(A_low=A_low, T=T, A_high=A_high, S=S,
                                               E_low=E_low, T_hard=T_hard))
        except Exception:
            r[f'{name}_rae_{tag}'] = np.nan
    # T-faithfulness: every low-level cross-cluster block must show up as a high-level edge
    absA = np.abs(A_high)
    np.fill_diagonal(absA, 0.)
    mx = absA.max() if absA.size else 0.
    for tag, tol in (('abs', 1e-8), ('rel', 1e-6 * mx)):
        pred = absA > tol
        nb = int(block.sum())
        r[f'{name}_recall_{tag}'] = float((pred & block).sum() / nb) if nb else np.nan
        r[f'{name}_spurious_{tag}'] = int((pred & ~block).sum())
        r[f'{name}_faithful_{tag}'] = int(nb > 0 and (pred & block).sum() == nb)
        r[f'{name}_edges_{tag}'] = int(pred.sum())
    return r


# ---------------------------------------------------------------- job
def job(a):
    torch.set_num_threads(1)
    kind, p, k, density, part, trial = a
    inst = gen_instance(kind, p, k, density, trial)
    if inst is None:
        return None
    A_low, T_gt, A_high_gt = inst
    E_low = np.linalg.inv(np.eye(p) - A_low)

    if part == 'gt':
        T_hard = (T_gt != 0).astype(float)
    else:
        T_hard = random_partition(p, k, np.random.default_rng(10_000 + trial))

    block = block_structure(A_low, T_hard)
    row = dict(kind=kind, p=p, k=k, density=density, part=part, trial=trial,
               rho_low=float(np.max(np.abs(np.linalg.eigvals(A_low)))),
               block_edges=int(block.sum()),
               quotient_cyclic=int(is_cyclic(block)),
               mask_edges=int(high_level_mask(A_low=A_low, T_hard=T_hard).sum()),
               gt_edges=int((A_high_gt != 0).sum()))
    # ceiling: the generating abstraction itself, scored the same way (gt partition only)
    if part == 'gt':
        row.update(score('oracle', A_high_gt, T_gt, A_low, E_low, T_hard, block, 0.))

    fits = (('reg', lambda: fit_abstraction(A_low=A_low, T_hard=T_hard, E_low=E_low,
                                            n_restarts=NRESTARTS, seed=trial)),
            ('alt', lambda: fit_abstraction_alternative(A_low=A_low, E_low=E_low, T_hard=T_hard,
                                                        n_restarts=NRESTARTS, seed=trial)),
            ('K',   lambda: fit_abstraction_K(A_low=A_low, E_low=E_low, T_hard=T_hard,
                                              n_restarts=NRESTARTS, seed=trial)))
    for name, fn in fits:
        t0 = time.perf_counter()
        try:
            A_h, T_f = fn()
            dt = time.perf_counter() - t0
            row.update(score(name, A_h, T_f, A_low, E_low, T_hard, block, dt))
        except Exception as e:
            row[f'{name}_time'] = time.perf_counter() - t0
            row[f'{name}_failed'] = 1
            row[f'{name}_err'] = type(e).__name__
    return row


if __name__ == '__main__':
    N_TRIALS = 15
    jobs = [(kind, p, k, 0.6, part, t)
            for kind in ('acyclic', 'cyclic')
            for (p, k) in ((24, 4), (40, 6))
            for part in ('gt', 'random')
            for t in range(N_TRIALS)]
    t0 = time.time()
    rows = []
    with mp.Pool(6) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            if r is not None:
                rows.append(r)
            if i % 20 == 0:
                print(f'{i}/{len(jobs)}  {(time.time()-t0)/60:.1f} min', flush=True)
    df = pd.DataFrame(rows).sort_values(['kind', 'p', 'part', 'trial'])
    df.to_csv(OUT, index=False)
    print(f'\ndone in {(time.time()-t0)/60:.1f} min   n={len(df)}   -> {OUT}')
