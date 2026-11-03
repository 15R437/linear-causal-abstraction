"""Does the All-In violation WEIGHTING matter? pairs vs nodes vs dilution.

Same greedy, same candidate shortlist, same tie-break, same budget -- only the violation
cost differs. 'pairs' (current) penalises large-cluster violations MORE; 'dilution'
penalises them LESS; 'nodes' is size-neutral. Scored through the full pipeline, because
all three reach 0 violations on easy instances and can only be separated downstream.
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
from partition_baselines import allin_agglom_mode, violations
from frontier_fig import score, P


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
    K = max(occ(A, seed=seed)[2]['k'], 2)
    out = []
    for mode in ('pairs', 'nodes', 'dilution'):
        t0 = time.time()
        T = allin_agglom_mode(A, K, mode)
        r = score(A, T, E_low, su, absA, tot, method=mode, k=K, seed=seed,
                  density=dens, secs=time.time() - t0)
        r.update(violations(A, T)); out.append(r)
    return out


if __name__ == '__main__':
    jobs = [(d, i) for d in (0.1, 0.3, 0.5, 0.7, 0.9) for i in range(10)]
    print('%d instances x 3 violation weightings' % len(jobs), flush=True)
    t0 = time.time(); rows = []
    path = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/allin_objective.csv'
    with mp.Pool(6) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs); pd.DataFrame(rows).to_csv(path, index=False)
            if i % 10 == 0:
                print('%d/%d %.1fmin' % (i, len(jobs), (time.time()-t0)/60), flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print('done %.1f min' % ((time.time()-t0)/60), flush=True)
