"""paper_alpha_recon re-run: fit_abstraction_K vs fit_abstraction_alternative, 2 blockiness.

Same design as scripts/alpha_recon.py -- reconstruction error vs tolerance alpha, one panel
per density, every method at the SAME budget k = |pi_0|, AOC Closure as the fallback -- with
three changes:

1. `get_S` is the FIXED version (local inverse (I - A[pi_i,pi_i])^-1 t_i, not the submatrix
   E[pi_i,pi_i] t_i). Verified here: the generating abstraction now scores RAE ~1e-32
   against itself at every density, where the old S gave 0.02-0.21.

2. EVERY partition is scored under BOTH fitters. HailMary never sees a fitter -- it reads
   only A_low and its logits -- so each arm is trained ONCE and re-scored, exactly the paired
   design of FITTER_SWITCH.md. Every K-vs-alt difference below is the fitter alone, with zero
   training noise. `_K` at n_restarts=10, `_alt` at process_output's defaults (5, maxiter=500)
   so the `_alt` arm reproduces the published figure.

3. A `GT partition` arm: the ground-truth partition handed to the fitter. This is the
   ceiling-attainability check the original figure lacked. `ground truth` (analytic) is the
   generating pair itself and needs no fit.

WHY BOTH FITTERS. On the GT partition at blockiness=1, where recon=0 is attainable:
`_alt` reaches recon 0.000 in 9/9 probes; `_K` reaches 0.30-1.00 (median 0.95) at the SAME
RAE ~1e-12, with ||A_high|| collapsing to 0.00-8.93. `_K` is exact but T-UNFAITHFUL: it
satisfies the abstraction equation by attenuating A_high, because it has no equivalent of
`_alt`'s `maximise ||A_high||^2 subject to residual <= tol` constraint -- its tie-break
(`tie_tol = best*1e-3 + 1e-8`, then max ||A_high|| among tied restarts) is far too weak when
the best residual is ~1e-13. So a `_K`-only version of this figure would pin every method AND
the ceiling near recon=1.0 and look like universal failure. Running both makes that
attributable instead of invisible. See FITTER_SWITCH.md 2026-09-23 ("port that constrained
formulation to `_K`; do not soften it into the objective").

    recon = ||A - A_hat||_F^2 / ||A||_F^2      1.0 = edgeless high-level graph, 0 = perfect

Blockiness sweep (heterogeneity=0 throughout, so blockiness is the only thing moving):
    --blk 0.3   weak blocks: A is far from T A_high T^+; GT recon ~0.67-0.71, little range
    --blk 1.0   ideal: A is exactly T A_high T^+; GT recon = 0.000, full range available
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
from src.utils.helper import fit_abstraction_K, fit_abstraction_alternative, get_S, is_dag
from src.models.hail_mary import HailMary
from occ import occ, oc_closure
from partition_baselines import allin_agglom
from frontier_fig import rae

P, EPOCHS = 100, 3000
DENS = [0.1, 0.5, 0.9]
LAMBDAS = [0., 100., 1000.]
NR_K, NR_ALT, MAXITER_ALT = 10, 5, 500


def job(a):
    torch.set_num_threads(1)
    dens, seed, BLK, HET = a
    try:
        A, T_gt, S_gt, A_high_gt = sample_low_level_graph(n_high=10, n_low=P, density=dens,
            shuffle=True, blockiness=BLK, heterogeneity=HET, internal_density=0.,
            graph_type='random', seed=seed, connected=True, max_attempts=400)
    except Exception:
        return []
    E = np.linalg.inv(np.eye(P) - A); nA = (A ** 2).sum(); su = np.ones(P)
    K = max(occ(A, seed=seed)[2]['k'], 2)
    com = dict(density=dens, seed=seed, blk=BLK, het=HET)
    out = []

    # ---- analytic reference: the generating abstraction, weighted T (not binarised).
    # RAE is MEASURED, not assumed 0 -- it is the calibration check for the get_S fix.
    Th_gt = (T_gt != 0).astype(float)
    try:
        rae_gt = rae(A, Th_gt, T_gt, A_high_gt, S_gt, E, su)[0]
    except Exception:
        rae_gt = np.nan
    out.append(dict(method='ground truth', fitter='analytic', lam_oc=np.nan,
                    k=T_gt.shape[1], recon=float(((A - T_gt @ A_high_gt @ np.linalg.pinv(T_gt)) ** 2).sum() / nA),
                    rae=rae_gt, secs=0., norm_Ahigh=float(np.linalg.norm(A_high_gt)),
                    marg=0, qcyc=np.nan, **com))

    def score(nm, lam, Th, train_secs):
        """Score ONE partition under BOTH fitters. Th is identical across the two rows."""
        rows = []
        kq = Th.shape[1]
        try:
            Aq = (Th.T @ np.abs(A) @ Th) * (1 - np.eye(kq))
            qcyc = int(not is_dag(Aq))
        except Exception:
            qcyc = -1
        for fit_nm, fn in (
            ('K',   lambda: fit_abstraction_K(A_low=A, E_low=E, T_hard=Th,
                                              n_restarts=NR_K, seed=seed)),
            ('alt', lambda: fit_abstraction_alternative(A_low=A, E_low=E, T_hard=Th,
                                                        n_restarts=NR_ALT, maxiter=MAXITER_ALT,
                                                        seed=seed, faithful=True)[:2])):
            r = dict(method=nm, fitter=fit_nm, lam_oc=lam, k=kq, train_secs=train_secs,
                     qcyc=qcyc, **com)
            t0 = time.time()
            try:
                Ah, Tp = fn()
                S = get_S(T=Tp, T_hard=Th, A_low=A)
                r['recon'] = float(((A - Tp @ Ah @ np.linalg.pinv(Tp)) ** 2).sum() / nA)
                r['rae'] = rae(A, Th, Tp, Ah, S, E, su)[0]
                r['norm_Ahigh'] = float(np.linalg.norm(Ah))
                r['marg'] = int((np.abs(Tp).max(1) <= 1e-2).sum())
            except Exception as e:
                r['recon'] = r['rae'] = np.nan; r['err'] = type(e).__name__
            r['secs'] = time.time() - t0
            rows.append(r)
        return rows

    out += score('GT partition', np.nan, Th_gt, 0.)
    for nm, fn in (('AOC Closure', lambda: occ(A, seed=seed)[1]),
                   ('OC Closure', lambda: oc_closure(A, seed=seed)[1]),
                   ('Greedy All-In', lambda: allin_agglom(A, K))):
        t0 = time.time(); Th = fn(); out += score(nm, np.nan, Th, time.time() - t0)

    for lam in LAMBDAS:
        t0 = time.time()
        torch.manual_seed(31337 + seed)
        m = HailMary(n_low=P, n_high=K, A_low=torch.from_numpy(A).float(), init_type='random')
        m.train(num_epochs=EPOCHS, lambda_recon=1., lambda_graph=100., lambda_one_child=lam,
                lambda_info=0., lambda_surj=0, verbose=False)
        tt = time.time() - t0
        out += score('GradCAP', lam, m.get_partition()[0].detach().numpy(), tt)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--out', default='alpha_recon_K')
    ap.add_argument('--blk', type=float, nargs='+', default=[0.3, 1.0])
    ap.add_argument('--het', type=float, default=0.0)
    ap.add_argument('--procs', type=int, default=6)
    args = ap.parse_args()
    # interleave blockiness so a truncated run still has both settings at every density
    jobs = [(d, i, b, args.het) for i in range(args.n) for d in DENS for b in args.blk]
    print('%d jobs = %d seeds x %d dens x %d blk; each = 7 partitions x 2 fitters, het=%g'
          % (len(jobs), args.n, len(DENS), len(args.blk), args.het), flush=True)
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
