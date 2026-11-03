"""fit_abstraction_auto on four partition types: does it separate good from bad, and what
does faithfulness cost?

Only the DEFAULT fitter is measured. `auto` routes on quotient acyclicity, so it is the
propagation solver on the acyclic cells and `_K` on the cyclic ones -- that routing is part
of what is being measured, and `fit_method` records which branch each row took.

    GT acyclic   generating partition of an acyclic model   -> auto picks propagation
    GT cyclic    generating partition of a CYCLIC model     -> auto picks _K
    random       balanced random assignment, same budget k  -> the control

Those three are scored at faithful=True and faithful=False on RAE, reconstruction and time.

UCA IS NOT FITTED. It is not a partition learner -- it learns a continuous T end-to-end -- so
binarising it and handing it to the fitter would measure something UCA never claims to do.
It is scored on its OWN raw continuous T and its own A_high (`get_A_high`), with its own
training time, and appears as a single reference that does not vary with `faithful` (the flag
is a property of the fitter, and UCA does not use one).

UCA is trained for 6000 gradient steps (`wrapper.uca(num_epochs=6000)`). NOTE: prior to
2026-09-24 `uca_train` applied no gradient at all -- it called `_training_step()` without
`backward()`/`step()`, so every earlier UCA result in this repo was an untrained random
initialisation. Its T is still fairly DENSE after training -- every entry nonzero, with ~0.41
of the column mass in the argmax cell (0.16 untrained). So `recon` and `A_high` use the raw T, unmasked. A hard
assignment is still needed for the two places the RAE definition structurally requires one
(which cluster a do() acts on, and the block-local S), and the column argmax is used there;
`uca_argmax_mass` records how much of T that reading represents, so the caveat is visible in
the data rather than buried here.

WHY. REPORT.md records that a random partition reaches RAE ~ 0 too, so RAE alone cannot serve
as a partition-quality signal. Profiled at p=100, d=0.5:

                                RAE        recon     time
    GT      faithful=True     8.3e-32     5.9e-12    5.4s
    GT      faithful=False    1.3e-31     5.1e-01    0.1s
    random  faithful=True     2.2e-12     9.9e-01   20.3s
    random  faithful=False    2.2e-12     9.9e-01    3.3s

RAE is ~1e-12 or below on BOTH -- no separation at all -- while reconstruction separates them
completely, but ONLY at faithful=True: faithful=False collapses the GT partition to 0.51 as
well. So the figure's claim is "reconstruction discriminates, RAE does not, and faithfulness
is what buys the discrimination".

BLOCKINESS = 1, HETEROGENEITY = 0 IS REQUIRED. There A is exactly T A_high T^+ so recon = 0
is attainable; at blockiness 0.3 even the ground truth reconstructs at 0.51-0.57
(ALPHA_RECON_K.md) and the separation cannot appear at all.

THREE CANDIDATE X-AXES are swept so the most informative can be chosen from the data rather
than assumed: p (size), k (cluster count) and density. They share the p=100, k=10, d=0.5 cell.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, argparse, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.utils.generate_data import (sample_low_level_graph, generate_random_T,
                                     T_concretisation_sampling)
from src.utils.helper import fit_abstraction_auto, get_A_high, get_S, is_dag
from src.models.wrapper import uca
# Canonical metrics from src/utils/losses.py, NOT the local copies in frontier_fig.py.
# Both were rescaled on 2026-09-24: RAE now returns sqrt(num/den) and reconstruction_loss is
# a ratio of Frobenius NORMS, not of squares. Values here are therefore the square root of
# those in any CSV produced before that date -- do not compare the two scales directly.
from src.utils.losses import RAE, reconstruction_loss

NR = 5
UCA_EPOCHS = 6000
BLK, HET = 1.0, 0.0
# p, k, density. The three sweeps share (100, 10, 0.5).
CELLS = ([(p, k, 0.5) for p, k in ((24, 4), (40, 6), (60, 8), (100, 10))]
         + [(100, k, 0.5) for k in (4, 16, 25)]
         + [(100, 10, d) for d in (0.1, 0.9)])


def cyclic_A_high(k, density, rho, rng):
    """Random high-level graph guaranteed to contain a cycle, rescaled to spectral radius rho."""
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


def random_partition(p, k, rng):
    lab = np.concatenate([np.arange(k), rng.integers(0, k, p - k)])
    rng.shuffle(lab)
    T = np.zeros((p, k))
    T[np.arange(p), lab] = 1.
    return T


def score(A, Th, pname, com):
    """Score one partition under fit_abstraction_auto at both faithfulness settings."""
    p, kq = Th.shape
    E = np.linalg.inv(np.eye(p) - A); su = np.ones(p)
    Aq = (Th.T @ np.abs(A) @ Th) * (1 - np.eye(kq))
    qcyc = int(not is_dag(Aq))
    out = []
    for faith in (True, False):
        r = dict(partition=pname, faithful=int(faith), k=kq, qcyc=qcyc, **com)
        t0 = time.time()
        try:
            Ah, Tp, fm = fit_abstraction_auto(A_low=A, T_hard=Th, E_low=E, n_restarts=NR,
                                              seed=com['seed'], maxiter=100, faithful=faith,
                                              return_method=True)
            r['secs'] = time.time() - t0
            r['fit_method'] = fm
            S = get_S(T=Tp, T_hard=Th, A_low=A)
            r['rae'] = RAE(A_low=A, T=Tp, A_high=Ah, S=S, E_low=E, T_hard=Th, sigma_u=su)
            r['recon'] = reconstruction_loss(A_low=A, T=Tp, A_high=Ah)
            r['marg'] = int((np.abs(Tp).max(1) <= 1e-2).sum())
            r['norm_Ahigh'] = float(np.linalg.norm(Ah))
        except Exception as e:
            r['secs'] = time.time() - t0
            r['rae'] = r['recon'] = np.nan; r['err'] = type(e).__name__
        out.append(r)
    return out


def job(a):
    torch.set_num_threads(1)
    P, K, dens, seed = a
    com = dict(p=P, k_budget=K, density=dens, seed=seed)
    out = []

    # ---- acyclic instance: carries GT acyclic, random and UCA
    try:
        A, T_gt, _, _ = sample_low_level_graph(n_high=K, n_low=P, density=dens, shuffle=True,
            blockiness=BLK, heterogeneity=HET, internal_density=0., graph_type='random',
            seed=seed, connected=True, max_attempts=400)
    except Exception:
        A = None
    if A is not None:
        out += score(A, (T_gt != 0).astype(float), 'GT acyclic', com)
        rng = np.random.default_rng(seed)
        out += score(A, random_partition(P, K, rng), 'random', com)
        t0 = time.time()
        try:
            Tu = uca(A_low=A, k=K, num_epochs=UCA_EPOCHS)
            secs = time.time() - t0
            # UCA's own continuous T, restricted to its argmax support; no refit.
            Thu = (Tu == Tu.max(axis=1, keepdims=True)).astype(float)
            E = np.linalg.inv(np.eye(P) - A); su = np.ones(P)
            Ahu = get_A_high(Tu, A)                      # raw dense T, unmasked
            Su = get_S(T=Tu, T_hard=Thu, A_low=A)
            Aq = (Thu.T @ np.abs(A) @ Thu) * (1 - np.eye(K))
            base = dict(partition='UCA', k=K, qcyc=int(not is_dag(Aq)), secs=secs,
                        fit_method='none (UCA end-to-end)',
                        rae=RAE(A_low=A, T=Tu, A_high=Ahu, S=Su, E_low=E, T_hard=Thu,
                                sigma_u=su),
                        recon=reconstruction_loss(A_low=A, T=Tu, A_high=Ahu),
                        marg=int((np.abs(Tu).max(1) <= 1e-2).sum()),
                        uca_argmax_mass=float((np.abs(Tu) * Thu).sum() / np.abs(Tu).sum()),
                        uca_epochs=UCA_EPOCHS,
                        norm_Ahigh=float(np.linalg.norm(Ahu)), **com)
            # same row under both faithfulness settings: the flag does not apply to UCA
            for faith in (1, 0):
                out.append(dict(base, faithful=faith))
        except Exception:
            pass

    # ---- cyclic instance: carries GT cyclic
    try:
        rng = np.random.default_rng(10_000 + seed)
        Ah_c = cyclic_A_high(K, dens, 0.7, rng)
        if Ah_c is not None:
            T_c, _ = generate_random_T(P, K, (1., -2.), (2., -1.), seed=seed)
            A_c, _ = T_concretisation_sampling(A_high=Ah_c, T=T_c, internal_density=0.,
                                               blockiness=BLK, heterogeneity=HET, seed=seed)
            perm = rng.permutation(P)
            out += score(A_c[perm][:, perm], (T_c[perm] != 0).astype(float), 'GT cyclic', com)
    except Exception:
        pass
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--out', default='fitter_gt_random')
    ap.add_argument('--procs', type=int, default=6)
    args = ap.parse_args()
    jobs = [(p, k, d, i) for i in range(args.n) for p, k, d in CELLS]
    print('%d jobs = %d seeds x %d cells; 4 partitions x 2 faithful, fitter=auto'
          % (len(jobs), args.n, len(CELLS)), flush=True)
    t0 = time.time(); rows = []
    path = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/%s.csv' % args.out
    with mp.Pool(args.procs) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs); pd.DataFrame(rows).to_csv(path, index=False)
            el = (time.time() - t0) / 60
            print('%d/%d %.1fmin (eta %.0fmin)' % (i, len(jobs), el, el / i * (len(jobs) - i)),
                  flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print('done %.1f min -> %s' % ((time.time() - t0) / 60, path), flush=True)
