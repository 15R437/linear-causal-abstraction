"""Frontier: tolerated abstraction error (alpha) vs reconstruction error.

Replaces the data behind paper_alpha_main, with ReconErr on y instead of information loss,
and with everything that has changed since: current method names, fixed budget k <= 20,
`fit_abstraction_auto` as the fitter, and the canonical metrics from src/utils/losses.py
(RAE returns sqrt(num/den); reconstruction_loss is a ratio of Frobenius NORMS, so both are
the square root of pre-2026-09-24 values).

THE COMPOSITE. At tolerance alpha a method contributes its OWN ReconErr if it both clears the
tolerance (RAE <= alpha) and beats AOC Closure on ReconErr; otherwise it falls back to AOC
Closure. So the AOC line is a ceiling by construction, every curve starts there as alpha -> 0,
and the figure reads as "how far below the safe default can each method get, and at what
tolerance". That is the sense in which the family offers a RANGE of choices.

BLOCKINESS = 1, HETEROGENEITY = 0 IS REQUIRED, not a default. There A is exactly T A_high T^+,
so ReconErr = 0 is attainable and the axis has full range. At blockiness 0.3 even the ground
truth reconstructs at 0.51-0.57 (ALPHA_RECON_K.md) and no frontier can appear at all.

FAITHFUL=TRUE IS ALSO REQUIRED. An unfaithful fit attenuates A_high toward zero and
reconstructs the ground-truth partition itself at ~0.5, which collapses the y axis
(see figures/fitter_gt_random_table.tex).

--n is deliberately small by default: this is meant to be run once for the shape and again
later for the error bars.
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
from src.utils.generate_data import sample_low_level_graph
from src.utils.helper import fit_abstraction_auto, get_S, is_dag
from src.utils.losses import RAE, reconstruction_loss
from src.models.hail_mary import HailMary
from occ import occ, oc_closure, budget_coarsening
from partition_baselines import allin_agglom

P, EPOCHS, BUDGET = 100, 3000, 20
DENS = [0.1, 0.5, 0.9]
LAMBDAS = [0., 100., 1000.]
BLK, HET = 1.0, 0.0


def job(a):
    torch.set_num_threads(1)
    dens, seed = a
    try:
        A, T_gt, _, _ = sample_low_level_graph(n_high=10, n_low=P, density=dens, shuffle=True,
            blockiness=BLK, heterogeneity=HET, internal_density=0., graph_type='random',
            seed=seed, connected=True, max_attempts=400)
    except Exception:
        return []
    E = np.linalg.inv(np.eye(P) - A); su = np.ones(P)
    K = BUDGET
    com = dict(density=dens, seed=seed, blk=BLK)
    out = []

    def score(nm, lam, Th, secs):
        r = dict(method=nm, lam_oc=lam, k_used=int((Th.sum(0) > 0).sum()), secs=secs, **com)
        try:
            Ah, Tp, fm = fit_abstraction_auto(A_low=A, T_hard=Th, E_low=E, n_restarts=5,
                                              seed=seed, maxiter=100, faithful=True,
                                              return_method=True)
            S = get_S(T=Tp, T_hard=Th, A_low=A)
            r['rae'] = RAE(A_low=A, T=Tp, A_high=Ah, S=S, E_low=E, T_hard=Th, sigma_u=su)
            r['recon'] = reconstruction_loss(A_low=A, T=Tp, A_high=Ah)
            r['fit_method'] = fm
            r['marg'] = int((np.abs(Tp).max(1) <= 1e-2).sum())
        except Exception as e:
            r['rae'] = r['recon'] = np.nan; r['err'] = type(e).__name__
        return r

    # ground truth reference (its own generating abstraction, no fit)
    out.append(score('ground truth', np.nan, (T_gt != 0).astype(float), 0.))

    def _oc():
        g, T, _ = oc_closure(A, seed=seed)
        if len(g) <= K:
            return T
        try:
            g2, _ = budget_coarsening(A, g, K, rng=np.random.default_rng(seed))
            T2 = np.zeros((P, len(g2)))
            for j, mem in enumerate(g2):
                T2[np.asarray(mem), j] = 1.0
            return T2
        except Exception:
            return T

    for nm, fn in (('AOC Closure', lambda: occ(A, seed=seed, k=K)[1]),
                   ('OC Closure', _oc),
                   ('Greedy All-In', lambda: allin_agglom(A, min(K, P)))):
        t0 = time.time(); Th = fn(); out.append(score(nm, np.nan, Th, time.time() - t0))

    for lam in LAMBDAS:
        t0 = time.time()
        torch.manual_seed(31337 + seed)
        m = HailMary(n_low=P, n_high=K, A_low=torch.from_numpy(A).float(), init_type='random')
        m.train(num_epochs=EPOCHS, lambda_recon=1., lambda_graph=100., lambda_one_child=lam,
                lambda_info=0., lambda_surj=0, verbose=False)
        out.append(score('GradCAP', lam, m.get_partition()[0].detach().numpy(),
                         time.time() - t0))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--out', default='alpha_frontier')
    ap.add_argument('--procs', type=int, default=6)
    args = ap.parse_args()
    jobs = [(d, i) for i in range(args.n) for d in DENS]
    print('%d jobs = %d seeds x %d dens; budget k<=%d, blockiness %g, faithful fits'
          % (len(jobs), args.n, len(DENS), BUDGET, BLK), flush=True)
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
