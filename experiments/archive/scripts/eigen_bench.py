"""Does the closed-form eigen solver replace `_K` on cyclic quotients, and does it survive
partitions that are not the ground truth?

THREE FITTERS on the same (instance, partition):
    auto    fit_abstraction_auto        -- propagation on an acyclic quotient, _K on a cyclic
    _K      fit_abstraction_K(faithful=False) -- the arm tk_sweep reports
    eigen   fit_abstraction_eigen       -- sinks free, cyclic SCCs pinned by an eigenproblem

FIVE PARTITIONS, because the eigen argument is about the QUOTIENT's cycle structure and says
nothing about the partition being correct -- so a partition that is merely valid, or not even
that, is the real test:
    GT, AOC Closure, OC Closure, Greedy All-In, random (balanced, same k)

TWO REGIMES: acyclic (sample_low_level_graph) and cyclic (a high-level graph built to contain
a cycle, spectral radius 0.7, concretised through a random T). The eigen solver should help
ONLY in the cyclic one: on an acyclic quotient every cluster is either a sink or propagated,
which is what `fit_abstraction` already does.

METRICS. `edge_recall` is the T-faithfulness measure, and it needs no ground truth, so it
is defined for a learned partition exactly as for the generating one: the QUOTIENT MASK says
where a high-level edge ought to be -- (i,j) is in it iff some low-level edge runs from
cluster i into cluster j -- and recall is the fraction of those that A_high actually has,

    edge_recall = mean over {(i,j) in quotient mask} of [ |A_high[i,j]| > 1e-6 max|A_high| ]

thresholded relative to the fit's own largest entry, which makes it gauge-invariant
(D-conjugation scales entry (i,j) by d_j/d_i != 0 and leaves the support alone). It is a
SUPPORT test, so read it beside `norm_Ahigh`: a uniformly annihilated fit can score
edge_recall = 1.000 at ||A_high|| = 1e-8, and the norm is what catches that.

`terr` (1 - |cos| against the true t, averaged over clusters) is only meaningful for the GT
partition and is NaN elsewhere, since a different partition's columns have no true t to be
compared against.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, argparse, warnings
B = '/Users/issy/Projects/linear-causal-abstraction'
sys.path.insert(0, B)
sys.path.insert(0, B + '/experiments/abstraction_study')
sys.path.insert(0, B + '/experiments/abstraction_study/scripts')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.utils.generate_data import (sample_low_level_graph, generate_random_T,
                                     T_concretisation_sampling)
from src.utils.helper import (fit_abstraction_auto, fit_abstraction_K, get_S,
                              cluster_indices, quotient_mask, is_dag)
from src.utils.losses import RAE, reconstruction_loss
from src.algos.T_fitter import fit_abstraction_eigen
from occ import occ, oc_closure, budget_coarsening
from partition_baselines import allin_agglom

NR, MAXITER, DENS, BLK, HET = 5, 100, 0.5, 1.0, 0.0
# Two axes sharing (100, 10), as in tk_sweep: p at fixed p/k = 10, and k at fixed p = 100.
# Cost is dominated by the largest cells (480 CPU-s per seed at (200,20), 230 at (100,20)
# against 4-30 for the small ones), so the intermediate points are close to free and the
# axes are worth filling in rather than leaving at three points each.
CELLS = [(40, 4), (60, 6), (80, 8), (140, 14), (200, 20),        # p axis, k = p/10
         (100, 4), (100, 8), (100, 10), (100, 16), (100, 20)]    # k axis, p = 100


def cyclic_A_high(k, density, rho, rng):
    for _ in range(500):
        Mk = (rng.random((k, k)) < density).astype(float)
        np.fill_diagonal(Mk, 0.)
        Ah = Mk * rng.uniform(0.5, 2.0, (k, k)) * rng.choice([-1, 1], (k, k))
        r = np.max(np.abs(np.linalg.eigvals(Ah)))
        if r < 1e-9:
            continue
        R = (Ah != 0); Rc = R.copy()
        for _ in range(k):
            Rc = Rc | (Rc @ R)
        if np.any(np.diag(Rc)):
            return Ah * (rho / r)
    return None


def random_partition(p, k, rng):
    lab = np.concatenate([np.arange(k), rng.integers(0, k, p - k)])
    rng.shuffle(lab)
    T = np.zeros((p, k)); T[np.arange(p), lab] = 1.
    return T


def score(A, T_gt, Ah_gt, regime, com, P, K, seed):
    E = np.linalg.inv(np.eye(P) - A); su = np.ones(P)
    Dg = np.linalg.norm(T_gt, axis=0); Dg = np.where(Dg > 1e-30, Dg, 1.0)
    Ah_u = (Dg[:, None] * Ah_gt) / Dg[None, :]
    Th_gt = (T_gt != 0).astype(float)
    rng = np.random.default_rng(seed)

    parts = {'GT': Th_gt, 'random': random_partition(P, K, rng)}
    try:
        parts['AOC Closure'] = occ(A, seed=seed, k=K)[1]
    except Exception:
        pass
    try:
        g = oc_closure(A, seed=seed, return_T=False)[0]
        try:
            g, _ = budget_coarsening(A, g, K, rng=np.random.default_rng(seed))
        except Exception:
            pass
        T2 = np.zeros((P, max(len(g), K)))
        for j, mem in enumerate(g):
            T2[np.asarray(mem), j] = 1.
        parts['OC Closure'] = T2
    except Exception:
        pass
    try:
        parts['Greedy All-In'] = allin_agglom(A, K)
    except Exception:
        pass

    out = []
    for pname, Th in parts.items():
        qb = quotient_mask(A, Th).astype(bool)
        base = dict(partition=pname, regime=regime, k_used=Th.shape[1],
                    qcyc=int(not is_dag(qb.astype(float))), n_qedges=int(qb.sum()), **com)
        gt_gauge = (pname == 'GT' and Th.shape == Th_gt.shape and (Th == Th_gt).all())
        idx = cluster_indices(Th)
        def recall(Ah):
            """Fraction of quotient edges that A_high actually carries."""
            if not qb.any():
                return np.nan
            m = np.abs(Ah).max()
            return float((np.abs(Ah[qb]) > 1e-6 * m).mean()) if m > 0 else 0.0

        for fname, fn in (
                ('auto', lambda Th=Th: fit_abstraction_auto(
                    A_low=A, T_hard=Th, E_low=E, n_restarts=NR, maxiter=MAXITER,
                    seed=seed, faithful=True)[:2]),
                ('_K', lambda Th=Th: fit_abstraction_K(
                    A_low=A, E_low=E, T_hard=Th, n_restarts=NR, maxiter=MAXITER,
                    seed=seed, faithful=False)),
                ('eigen', lambda Th=Th: fit_abstraction_eigen(
                    A_low=A, T_hard=Th, seed=seed, n_restarts=NR)[:2])):
            r = dict(base, fitter=fname)
            t0 = time.time()
            try:
                Ah, Tp = fn()
                r['secs'] = time.time() - t0
                S = get_S(T=Tp, T_hard=Th, A_low=A)
                r['rae'] = RAE(A_low=A, T=Tp, A_high=Ah, S=S, E_low=E, T_hard=Th, sigma_u=su)
                r['recon'] = reconstruction_loss(A_low=A, T=Tp, A_high=Ah)
                r['norm_Ahigh'] = float(np.linalg.norm(Ah))
                r['edge_recall'] = recall(Ah)
                if gt_gauge:
                    e = []
                    for j in range(Th.shape[1]):
                        ix = idx[j]
                        if not len(ix):
                            continue
                        a, b = Tp[ix, j], T_gt[ix, j]
                        na, nb = np.linalg.norm(a), np.linalg.norm(b)
                        e.append(1.0 if min(na, nb) <= 1e-12 else 1 - abs(a @ b) / (na * nb))
                    r['terr'] = float(np.mean(e)) if e else np.nan
            except Exception as ex:
                r['secs'] = time.time() - t0
                r['rae'] = r['recon'] = np.nan
                r['err'] = type(ex).__name__
            out.append(r)
    return out


def job(a):
    torch.set_num_threads(1)
    P, K, seed = a
    com = dict(p=P, k=K, seed=seed, density=DENS)
    out = []
    try:
        A, T_gt, _, Ah_gt = sample_low_level_graph(
            n_high=K, n_low=P, density=DENS, shuffle=True, blockiness=BLK, heterogeneity=HET,
            internal_density=0., graph_type='random', seed=seed, connected=True,
            max_attempts=400)
        out += score(A, T_gt, Ah_gt, 'acyclic', com, P, K, seed)
    except Exception:
        pass
    try:
        rng = np.random.default_rng(10_000 + seed)
        Ah_c = cyclic_A_high(K, DENS, 0.7, rng)
        if Ah_c is not None:
            T_c, _ = generate_random_T(P, K, (1., -2.), (2., -1.), seed=seed)
            A_c, _ = T_concretisation_sampling(A_high=Ah_c, T=T_c, internal_density=0.,
                                               blockiness=BLK, heterogeneity=HET, seed=seed)
            pm = rng.permutation(P)
            out += score(A_c[pm][:, pm], T_c[pm], Ah_c, 'cyclic', com, P, K, seed)
    except Exception:
        pass
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--procs', type=int, default=6)
    ap.add_argument('--out', default='eigen_bench')
    args = ap.parse_args()
    jobs = [(p, k, i) for i in range(args.n) for p, k in CELLS]
    print('%d jobs = %d seeds x %d cells; 5 partitions x 3 fitters x 2 regimes'
          % (len(jobs), args.n, len(CELLS)), flush=True)
    t0 = time.time(); rows = []
    path = '%s/experiments/abstraction_study/data/%s.csv' % (B, args.out)
    with mp.Pool(args.procs) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs); pd.DataFrame(rows).to_csv(path, index=False)
            el = (time.time() - t0) / 60
            print('%d/%d %.1fmin (eta %.0fmin)' % (i, len(jobs), el, el / i * (len(jobs) - i)),
                  flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print('done %.1f min -> %s' % ((time.time() - t0) / 60, path), flush=True)
