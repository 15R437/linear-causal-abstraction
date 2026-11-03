"""`fit_abstraction_K` against the oracle T, as p grows at fixed p/k and as k grows at fixed p.

THREE ARMS, one partition. Every arm is scored on the SAME ground-truth partition of the
SAME instance, so nothing here is about partition learning -- it isolates what the T-fitter
does once the partition is already correct:

    T_gt          the generating (T, A_high) straight from the sampler. Not a fit: the
                  oracle, i.e. the value each metric takes when T is exactly right. It is
                  what makes the other two readable -- RAE ~ 1e-16 here, so anything the
                  fitter reports above that is fitter error, not instance difficulty.
    _K faithful   fit_abstraction_K(faithful=True)
    _K unfaithful fit_abstraction_K(faithful=False)

`fit_abstraction_K` ONLY -- not `fit_abstraction_auto`. The quotient of a ground-truth
partition on a random DAG is acyclic, so `auto` would route every one of these rows to the
propagation solver and never exercise `_K` at all.

TWO REGIMES, recorded in the `regime` column and generated per cell from the same seed:

    acyclic   sample_low_level_graph -- the quotient of a ground-truth partition on a random
              DAG, which is itself a DAG.
    cyclic    a high-level graph built to CONTAIN a cycle (spectral radius 0.7), concretised
              through a random T. `fit_abstraction_K` is the only solver that handles these
              at all: `fit_abstraction` runs `enforce_dag` and deletes the cycle edges before
              it starts, and `reverse_topo_order` has no cycle detection.

The regimes are worth carrying together because a strongly connected quotient HAS NO SINKS,
and the sink t-vectors are exactly the ones the abstraction condition leaves free (measured
over 24 acyclic fits, sink clusters land at median 1 - |cos(t_hat, t_gt)| = 1.000 against
0.000 for non-sinks). So the cyclic regime is the direct test of whether reconstruction error
on an acyclic quotient is sink freedom rather than fitter error.

The cyclic arm is fitted at faithful=False ONLY -- the reported setting -- because the
faithful path costs 5-7x and is not what any figure here plots.

TWO AXES, sharing the (p=100, k=10) cell:

    p at fixed p/k = 10   40, 60, 80, 100, 120, 160, 200   (k = p/10)
    k at fixed p = 100    4, 8, 10, 16, 20, 25

Holding p/k fixed while p grows is the point of the first axis: cluster COUNT and cluster
SIZE then grow together, so it reads as "the same problem, larger", and any drift is scale,
not granularity. The second axis moves granularity with the problem size held still.

BLOCKINESS = 1, HETEROGENEITY = 0 IS REQUIRED for `recon` to mean anything. There A is
exactly T A_high T^+, so recon = 0 is attainable and the oracle arm confirms it does reach
~2e-16. At blockiness 0.5 the ORACLE itself reconstructs at 0.87-0.95 (ALPHA_RECON_K.md) and
no arm can be told from any other.

METRICS. `rae` and `recon` come from src/utils/losses.py, both ratios of Frobenius NORMS
since 2026-09-24 (so the square root of values in any earlier CSV here). Two diagnostics are
recorded alongside them because RAE and recon between them do not say WHERE a fit went wrong:

    terr_mean/max   1 - |cos(t_j, t_j^gt)| over clusters. Both vectors are supported on the
                    same block pi(j) and each is determined only up to scale, so the angle is
                    the whole of the disagreement. This is T-faithfulness measured DIRECTLY,
                    against the truth, rather than inferred from a reconstruction.
    edge_recall     of the low-level CROSS-CLUSTER edges, the fraction that have a
                    corresponding high-level edge in A_high -- T-faithfulness asked directly
                    and structurally. Gauge-INVARIANT (D-conjugation scales entry (i,j) by
                    d_j/d_i != 0, so the support is untouched), unlike the norm. Each edge is
                    thresholded at 1e-3 of its own true magnitude in the shared gauge, so
                    this also catches uniform annihilation.
    norm_Ahigh      ||A_high||_F, with the ORACLE REGAUGED to unit columns so the two are
                    comparable (see score()). An annihilating fit drives this to 0 while
                    still scoring a low RAE.
    norm_ratio      norm_Ahigh / norm_Ahigh_gt, both in the fitter's gauge.

`faithful` is recorded for both settings but the REPORTED arm is faithful=False. faithful=True
is not a better fit here: its delta floor (1e-9) sits ~8 orders below where the true T lives
(measured min ratio 0.19-0.23 at p=100, k=10), so the residual optimiser slides every
rank-deficient block down to the floor and rests there, and unlike faithful=False it carries
no ||A_high||-maximising tie-break to pull it back. helper.py's own "RESTING ON THE DELTA
FLOOR" log line fires on these instances at k=4 and k=10, not merely above k~15.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, argparse, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.utils.generate_data import (sample_low_level_graph, generate_random_T,
                                     T_concretisation_sampling)
from src.utils.helper import fit_abstraction_K, get_S, cluster_indices, quotient_mask, is_dag
from src.algos.T_fitter import fit_abstraction_eigen
from src.utils.losses import RAE, reconstruction_loss

NR, MAXITER = 5, 100
DENS, BLK, HET = 0.5, 1.0, 0.0
RATIO = 10
P_SWEEP = [40, 60, 80, 100, 120, 160, 200]          # k = p // RATIO
K_SWEEP = [4, 8, 10, 16, 20, 25]                    # at p = 100
P_FIXED = 100
CELLS = sorted({(p, p // RATIO) for p in P_SWEEP} | {(P_FIXED, k) for k in K_SWEEP})


def t_angle_err(T_fit, T_gt, T_hard):
    """Per-cluster 1 - |cos| between the fitted and true t-vectors, on their shared block."""
    errs = []
    for ix in cluster_indices(T_hard):
        if len(ix) == 0:
            continue
        j = int(np.argmax(T_hard[ix[0]]))
        a, b = T_fit[ix, j], T_gt[ix, j]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        errs.append(1.0 if min(na, nb) <= 1e-12
                    else float(1.0 - abs(a @ b) / (na * nb)))
    return (float(np.mean(errs)), float(np.max(errs))) if errs else (np.nan, np.nan)


def score(A, T_gt, S_gt, Ah_gt, regime, com, faiths):
    """The three arms on one instance: the oracle, then fit_abstraction_K at each setting.

    GAUGE. A_high is defined only up to D-conjugation: rescaling T's columns, T -> T D, sends
    A_high -> D^-1 A_high D, and ||.||_F is NOT invariant under that. The fitter returns
    unit-norm t_j; `generate_random_T` does not, so the raw Ah_gt lives in a DIFFERENT gauge
    from every fitted A_high and the two cannot be put on one axis. Regauging Ah_gt to unit
    columns moves its norm by -14% to +40%, growing with k. Everything below therefore uses
    Ah_gt_u, the oracle in the fitter's gauge. The SUPPORT is gauge-invariant either way,
    which is what makes edge_recall the better-behaved of the two measures.
    """
    P, K = T_gt.shape
    Th = (T_gt != 0).astype(float)
    E = np.linalg.inv(np.eye(P) - A)
    su = np.ones(P)
    Dg = np.linalg.norm(T_gt, axis=0)
    Dg = np.where(Dg > 1e-30, Dg, 1.0)
    Ah_gt_u = (Dg[:, None] * Ah_gt) / Dg[None, :]        # diag(Dg) Ah_gt diag(1/Dg)
    nAh_gt = float(np.linalg.norm(Ah_gt_u))
    qm = quotient_mask(A, Th)
    com = dict(com, regime=regime, n_qedges=int(qm.sum()), qcyc=int(not is_dag(qm)),
               norm_Ahigh_gt=nAh_gt, n_sinks=int((qm.astype(bool).sum(1) == 0).sum()))
    qb = qm.astype(bool)

    def recall(Ahat):
        """Of the low-level cross-cluster edges, how many have a high-level edge?

        Thresholded against each edge's OWN true magnitude in the shared gauge, not against
        Ahat's largest entry: a relative-to-self cut scores a uniformly annihilated fit at
        1.000 (measured: recall 1.000 at ||A_high|| = 2.5e-07), which is exactly the failure
        the measure exists to catch.
        """
        if not qb.any():
            return np.nan
        return float((np.abs(Ahat[qb]) > 1e-3 * np.abs(Ah_gt_u[qb])).mean())

    out = [dict(arm='T_gt', faithful=-1, secs=0.0, **com,
                rae=RAE(A_low=A, T=T_gt, A_high=Ah_gt, S=S_gt, E_low=E, T_hard=Th, sigma_u=su),
                recon=reconstruction_loss(A_low=A, T=T_gt, A_high=Ah_gt),
                terr_mean=0.0, terr_max=0.0, norm_Ahigh=nAh_gt, norm_ratio=1.0, marg=0,
                edge_recall=recall(Ah_gt_u))]

    for faith in faiths:
        nm = ('eigen' if faith == 'eigen'
              else ('_K faithful' if faith else '_K unfaithful'))
        r = dict(arm=nm, faithful=(-2 if faith == 'eigen' else int(faith)), **com)
        t0 = time.time()
        try:
            if faith == 'eigen':
                Ah, Tp = fit_abstraction_eigen(A_low=A, T_hard=Th, seed=com['seed'],
                                               n_restarts=NR)[:2]
            else:
                Ah, Tp = fit_abstraction_K(A_low=A, E_low=E, T_hard=Th, n_restarts=NR,
                                           seed=com['seed'], maxiter=MAXITER, faithful=faith)
            r['secs'] = time.time() - t0
            S = get_S(T=Tp, T_hard=Th, A_low=A)
            r['rae'] = RAE(A_low=A, T=Tp, A_high=Ah, S=S, E_low=E, T_hard=Th, sigma_u=su)
            r['recon'] = reconstruction_loss(A_low=A, T=Tp, A_high=Ah)
            r['terr_mean'], r['terr_max'] = t_angle_err(Tp, T_gt, Th)
            r['norm_Ahigh'] = float(np.linalg.norm(Ah))
            r['norm_ratio'] = r['norm_Ahigh'] / max(nAh_gt, 1e-30)
            r['marg'] = int((np.abs(Tp).max(1) <= 1e-2).sum())
            r['edge_recall'] = recall(Ah)
        except Exception as e:
            r['secs'] = time.time() - t0
            r['rae'] = r['recon'] = np.nan
            r['err'] = type(e).__name__
        out.append(r)
    return out


def cyclic_A_high(k, density, rho, rng):
    """Random high-level graph GUARANTEED to contain a cycle, rescaled to spectral radius rho.

    Lifted from fitter_gt_random.py so the two studies share one cyclic generator. Rejects
    draws whose transitive closure has an empty diagonal, i.e. that came out acyclic.
    """
    for _ in range(500):
        M = (rng.random((k, k)) < density).astype(float)
        np.fill_diagonal(M, 0.)
        Ah = M * rng.uniform(0.5, 2.0, (k, k)) * rng.choice([-1, 1], (k, k))
        r = np.max(np.abs(np.linalg.eigvals(Ah)))
        if r < 1e-9:
            continue
        R = (Ah != 0); Rc = R.copy()
        for _ in range(k):
            Rc = Rc | (Rc @ R)
        if not np.any(np.diag(Rc)):
            continue
        return Ah * (rho / r)
    return None


def job(a):
    torch.set_num_threads(1)
    P, K, seed, ARMS, BLK = a
    com = dict(p=P, k=K, density=DENS, seed=seed, ratio=P / K)
    try:
        A, T_gt, S_gt, Ah_gt = sample_low_level_graph(
            n_high=K, n_low=P, density=DENS, shuffle=True, blockiness=BLK,
            heterogeneity=HET, internal_density=0., graph_type='random', seed=seed,
            connected=True, max_attempts=400)
    except Exception:
        return []
    out = score(A, T_gt, S_gt, Ah_gt, 'acyclic', com, ARMS)

    # ---- cyclic instance at the same (p, k, seed); faithful=False only
    try:
        rng = np.random.default_rng(10_000 + seed)
        Ah_c = cyclic_A_high(K, DENS, 0.7, rng)
        if Ah_c is not None:
            T_c, _ = generate_random_T(P, K, (1., -2.), (2., -1.), seed=seed)
            A_c, S_c = T_concretisation_sampling(A_high=Ah_c, T=T_c, internal_density=0.,
                                                 blockiness=BLK, heterogeneity=HET, seed=seed)
            pm = rng.permutation(P)
            out += score(A_c[pm][:, pm], T_c[pm], S_c[pm], Ah_c, 'cyclic', com,
                         [a for a in ARMS if a is not True])
    except Exception:
        pass
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20, help='seeds per cell')
    ap.add_argument('--out', default='tk_sweep')
    ap.add_argument('--procs', type=int, default=6)
    ap.add_argument('--blk', type=float, default=BLK,
                    help='blockiness (default %g). At 1.0 A is EXACTLY T A_high T^+, so the '
                         'ORACLE reconstructs at ~2e-16 and ReconErr reads as absolute error. '
                         'Below 1.0 the oracle itself has non-zero ReconErr (0.41 at 0.8 in '
                         'the alpha study), so the fitted curves must be read against that '
                         'floor, not against zero.' % BLK)
    ap.add_argument('--arms', nargs='+', default=['unfaithful', 'eigen'],
                    choices=['faithful', 'unfaithful', 'eigen'],
                    help='which fitters to score. faithful=True is ~80%% of the runtime and '
                         'is not the reported arm.')
    args = ap.parse_args()
    amap = {'faithful': True, 'unfaithful': False, 'eigen': 'eigen'}
    arms = [amap[a] for a in args.arms]
    jobs = [(p, k, i, arms, args.blk) for i in range(args.n) for p, k in CELLS]
    print('%d jobs = %d seeds x %d cells %s; 3 arms each (fit_abstraction_K only)'
          % (len(jobs), args.n, len(CELLS), CELLS), flush=True)
    t0 = time.time(); rows = []
    path = ('/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
            '/data/%s.csv' % args.out)
    with mp.Pool(args.procs) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs)
            pd.DataFrame(rows).to_csv(path, index=False)
            el = (time.time() - t0) / 60
            print('%d/%d %.1fmin (eta %.0fmin)' % (i, len(jobs), el, el / i * (len(jobs) - i)),
                  flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print('done %.1f min -> %s' % ((time.time() - t0) / 60, path), flush=True)
