"""HailMary vs graph-partition baselines on the All-In / one-child conditions.

Two axes per plot: violation count (All-In, or one-child) against information loss
(intra-cluster edge mass). No T is fitted -- intra-cluster mass is a property of the
PARTITION alone, so this is fitter-independent and cheap. That also means the y-axis here
is plain intra-cluster mass, NOT the marginalisation-aware IEM used in paper_alpha_main.

Every method gets the SAME budget K = AOC Closure's cluster count on that instance,
because singletons trivially optimise both axes.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary
from src.utils.generate_data import sample_low_level_graph
from src.models.wrapper import random as random_partition, uca, hc_fast
from occ import occ
from partition_baselines import (bisim, sig_agglom, sig_kmeans, spectral_ncut,
                                 allin_agglom, violations)

P, EPOCHS = 100, 3000
DENS = [0.1, 0.3, 0.5, 0.7, 0.9]
NINST = 10
LAMBDAS = [0., 0.3, 1., 3., 10., 30., 100., 300., 1000.]


def row(A, T, absA, tot, **extra):
    v = violations(A, T)
    co = T @ T.T; np.fill_diagonal(co, 0)
    v['intra'] = float((absA * co).sum() / tot)
    v['allin_nodes_frac'] = v['allin_nodes'] / A.shape[0]
    v.update(extra)
    return v


def job(a):
    torch.set_num_threads(1)
    dens, seed = a
    try:
        A, *_ = sample_low_level_graph(n_high=10, n_low=P, density=dens, shuffle=True,
            blockiness=0.5, heterogeneity=1., internal_density=0., graph_type='random',
            seed=seed, connected=True, max_attempts=300)
    except Exception:
        return []
    absA = np.abs(A); tot = absA.sum()
    g0, T0, ci = occ(A, seed=seed); K = max(ci['k'], 2)
    com = dict(k=K, seed=seed, density=dens)
    out = [row(A, T0, absA, tot, method='AOC Closure', lam_oc=np.nan, secs=0., **com)]

    for nm, fn in (('all-in greedy', allin_agglom),
                   ('bisim', bisim), ('sig-agglom', sig_agglom), ('sig-kmeans', sig_kmeans),
                   ('hc', lambda Al, k: hc_fast(A_low=Al, k=k)),
                   ('spectral-ncut', spectral_ncut),
                   ('UCA', lambda Al, k: uca(A_low=Al, k=k)),
                   ('random', lambda Al, k: random_partition(A_low=Al, k=k))):
        t0 = time.time()
        try:
            T = fn(A, K)
            T = (T == T.max(axis=1, keepdims=True)).astype(float) if nm == 'UCA' else T
            out.append(row(A, T, absA, tot, method=nm, lam_oc=np.nan,
                           secs=time.time() - t0, **com))
        except Exception as e:
            print('  %s failed on d=%.1f s=%d: %s' % (nm, dens, seed, e), flush=True)

    # HailMary: graph-only, then the lambda_one-child dial on the hybrid objective
    for nm, lr, lam_list in (('HM graph-only', 0., [0.]), ('HailMary', 1., LAMBDAS)):
        for lam in lam_list:
            t0 = time.time()
            torch.manual_seed(31337 + seed)
            m = HailMary(n_low=P, n_high=K, A_low=torch.from_numpy(A).float(),
                         init_type='random')
            m.train(num_epochs=EPOCHS, lambda_recon=lr, lambda_graph=100.,
                    lambda_one_child=lam, lambda_info=0., lambda_surj=0, verbose=False)
            T = m.get_partition()[0].detach().numpy()
            out.append(row(A, T, absA, tot, method=nm, lam_oc=lam,
                           secs=time.time() - t0, **com))
    return out


if __name__ == '__main__':
    jobs = [(d, i) for d in DENS for i in range(NINST)]
    print('%d instances x (8 baselines + 1 graph-only + %d lambdas), p=%d, %d epochs'
          % (len(jobs), len(LAMBDAS), P, EPOCHS), flush=True)
    t0 = time.time(); rows = []
    path = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/partcmp.csv'
    with mp.Pool(6) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs)
            pd.DataFrame(rows).to_csv(path, index=False)
            if i % 5 == 0:
                print('%d/%d %.1fmin' % (i, len(jobs), (time.time() - t0) / 60), flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print('\ndone %.1f min -> %s' % ((time.time() - t0) / 60, path), flush=True)
