"""all-in greedy through the FULL pipeline: partition -> fit_abstraction_alternative.

Reuses frontier_fig.score(), so RAE / info_eff (iem) are computed exactly as in
frontier2.csv and the numbers are directly comparable. OCC is re-scored on the same
instances so the comparison is paired rather than cross-file.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, multiprocessing as mp
from src.utils.generate_data import sample_low_level_graph
from occ import occ
from partition_baselines import allin_agglom, sig_agglom
from frontier_fig import score, P

DENS = [0.1, 0.3, 0.5, 0.7, 0.9]
NINST = 10


def job(a):
    dens, seed = a
    try:
        A, *_ = sample_low_level_graph(n_high=10, n_low=P, density=dens, shuffle=True,
            blockiness=0.5, heterogeneity=1., internal_density=0., graph_type='random',
            seed=seed, connected=True, max_attempts=300)
    except Exception:
        return []
    absA = np.abs(A); tot = absA.sum()
    E_low = np.linalg.inv(np.eye(P) - A); su = np.ones(P)
    g0, T0, ci = occ(A, seed=seed); K = max(ci['k'], 2)
    com = dict(k=K, seed=seed, density=dens)
    out = [score(A, T0, E_low, su, absA, tot, method='OCC', secs=0., **com)]
    for nm, fn in (('all-in greedy', allin_agglom), ('sig-agglom', sig_agglom)):
        t0 = time.time()
        try:
            T = fn(A, K)
            out.append(score(A, T, E_low, su, absA, tot, method=nm,
                             secs=time.time() - t0, **com))
        except Exception as e:
            print('  %s failed d=%.1f s=%d: %s' % (nm, dens, seed, e), flush=True)
    return out


if __name__ == '__main__':
    jobs = [(d, i) for d in DENS for i in range(NINST)]
    print('%d instances x 3 methods through process_output' % len(jobs), flush=True)
    t0 = time.time(); rows = []
    path = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/greedy_pipeline.csv'
    with mp.Pool(6) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs); pd.DataFrame(rows).to_csv(path, index=False)
            if i % 10 == 0:
                print('%d/%d %.1fmin' % (i, len(jobs), (time.time()-t0)/60), flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print('done %.1f min' % ((time.time()-t0)/60), flush=True)
