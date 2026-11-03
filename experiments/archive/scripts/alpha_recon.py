"""Data for paper_alpha_recon: reconstruction error vs tolerance alpha, ideal regime.

blockiness=1, heterogeneity=0 -- the only regime where reconstruction is attainable at all
(A is exactly T A_high T^+, so the ground truth reconstructs to ~0). At blockiness<1 even a
perfect answer leaves 33-73% of the error, so the metric has almost no range.

    recon = ||A - A_hat||_F^2 / ||A||_F^2      1.0 = edgeless high-level graph, 0 = perfect

Every method runs at the SAME budget k = |pi_0| and falls back to AOC Closure unless it both
clears the tolerance (RAE <= alpha) AND beats AOC Closure on reconstruction.

--n controls instances per density (start small, raise later).
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
from src.utils.helper import process_output, get_S
from src.models.hail_mary import HailMary
from occ import occ, oc_closure
from partition_baselines import allin_agglom
from frontier_fig import rae

P, EPOCHS = 100, 3000
DENS = [0.1, 0.5, 0.9]
LAMBDAS = [0., 100., 1000.]


BLK, HET = 0.8, 0.0     # set from argv in __main__


def job(a):
    torch.set_num_threads(1)
    dens, seed, BLK, HET = a
    try:
        A, T_gt, _, A_high_gt = sample_low_level_graph(n_high=10, n_low=P, density=dens,
            shuffle=True, blockiness=BLK, heterogeneity=HET, internal_density=0.,
            graph_type='random', seed=seed, connected=True, max_attempts=400)
    except Exception:
        return []
    E = np.linalg.inv(np.eye(P) - A); nA = (A ** 2).sum(); su = np.ones(P)
    K = max(occ(A, seed=seed)[2]['k'], 2)
    com = dict(density=dens, seed=seed)
    out = []

    # ground-truth reference: the generating abstraction, weighted T (not binarised)
    Ah_gt = T_gt @ A_high_gt @ np.linalg.pinv(T_gt)
    out.append(dict(method='ground truth', lam_oc=np.nan, k=T_gt.shape[1],
                    recon=float(((A - Ah_gt) ** 2).sum() / nA), rae=0.0, secs=0., **com))

    def score(nm, lam, Th, secs):
        r = dict(method=nm, lam_oc=lam, k=Th.shape[1], secs=secs, **com)
        try:
            S, Tp, Ah = process_output(T_hard=Th, A_low=A, E_low=E)
            r['recon'] = float(((A - Tp @ Ah @ np.linalg.pinv(Tp)) ** 2).sum() / nA)
            r['rae'] = rae(A, Th, Tp, Ah, S, E, su)[0]
        except Exception:
            r['recon'] = r['rae'] = np.nan
        return r

    for nm, fn in (('AOC Closure', lambda: occ(A, seed=seed)[1]),
                   ('OC Closure', lambda: oc_closure(A, seed=seed)[1]),
                   ('Greedy All-In', lambda: allin_agglom(A, K))):
        t0 = time.time(); out.append(score(nm, np.nan, fn(), time.time() - t0))

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
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--out', default='alpha_recon')
    ap.add_argument('--blk', type=float, default=0.8)
    ap.add_argument('--het', type=float, default=0.0)
    args = ap.parse_args()
    BLK, HET = args.blk, args.het
    globals()['BLK'], globals()['HET'] = BLK, HET
    jobs = [(d, i, BLK, HET) for d in DENS for i in range(args.n)]
    print('%d instances x (3 refs + %d GradCAP lambdas), blk=%g het=%g'
          % (len(jobs), len(LAMBDAS), BLK, HET), flush=True)
    t0 = time.time(); rows = []
    path = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/%s.csv' % args.out
    with mp.Pool(6) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs); pd.DataFrame(rows).to_csv(path, index=False)
            print('%d/%d %.1fmin' % (i, len(jobs), (time.time() - t0) / 60), flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print('done %.1f min -> %s' % ((time.time() - t0) / 60, path), flush=True)
