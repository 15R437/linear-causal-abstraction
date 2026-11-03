"""Are the graph conditions reachable by an obvious scalarised greedy?

Defensive benchmark: min-intra, one-child-only, and a JOINT greedy that scalarises all
three quantities. Scored on the partition conditions AND through the full pipeline
(process_output -> RAE, info_eff), because the earlier round showed that dominating on
raw partition quality does not imply an exact abstraction exists.
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
from partition_baselines import greedy_merge, allin_agglom, violations
from frontier_fig import score, P

DENS = [0.1, 0.3, 0.5, 0.7, 0.9]
CFG = [('min-intra greedy', 0., 0., 1.), ('one-child greedy', 0., 1., 0.),
       ('joint greedy', 1., 1., 1.), ('all-in greedy', 1., 0., 0.)]


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
    out = []
    r = score(A, T0, E_low, su, absA, tot, method='OCC', secs=0., **com)
    r.update(violations(A, T0)); out.append(r)
    for nm, wa, wo, wi in CFG:
        t0 = time.time()
        try:
            T = greedy_merge(A, K, w_allin=wa, w_oc=wo, w_intra=wi)
            r = score(A, T, E_low, su, absA, tot, method=nm, secs=time.time()-t0, **com)
            r.update(violations(A, T)); out.append(r)
        except Exception as e:
            print('  %s failed d=%.1f s=%d: %s' % (nm, dens, seed, e), flush=True)
    return out


if __name__ == '__main__':
    jobs = [(d, i) for d in DENS for i in range(10)]
    print('%d instances x (OCC + %d greedies), conditions + full pipeline'
          % (len(jobs), len(CFG)), flush=True)
    t0 = time.time(); rows = []
    path = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/greedy_variants.csv'
    with mp.Pool(6) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs); pd.DataFrame(rows).to_csv(path, index=False)
            if i % 10 == 0:
                print('%d/%d %.1fmin' % (i, len(jobs), (time.time()-t0)/60), flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print('done %.1f min' % ((time.time()-t0)/60), flush=True)
