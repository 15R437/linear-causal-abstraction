"""Ablation of AOC Closure's All-In stage: one_child_closure alone vs the full method.

OC Closure is stage 1 only -- one-child guaranteed, All-In never enforced. It selects the
min-information-loss draw (the adopted criterion); AOC Closure selects the widest All-In
valid draw and repairs if none is valid. Same instances, same n_draws, so the difference
IS the All-In stage.

Note the budget is NOT controlled here and cannot be: the closure returns whatever k its
fixed point gives, which is the point -- you cannot impose both a budget and the condition.
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
from occ import occ, oc_closure
from partition_baselines import violations
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
    out = []
    g0, T0, ci = occ(A, seed=seed)
    r = score(A, T0, E_low, su, absA, tot, method='AOC Closure', seed=seed, density=dens,
              k=len(g0), repaired=int(ci['repaired']), secs=0.)
    r.update(violations(A, T0)); out.append(r)
    # stage 1 only: one-child fixed point, min-information-loss draw, no All-In
    # selection and no repair. Uses the adopted oc_closure criterion, so the ablation
    # measures the All-In stage against the OC Closure we actually ship.
    _, T1, _ = oc_closure(A, n_draws=12, seed=seed)
    r = score(A, T1, E_low, su, absA, tot, method='one-child closure', seed=seed,
              density=dens, k=T1.shape[1], repaired=0, secs=0.)
    r.update(violations(A, T1)); out.append(r)
    return out


if __name__ == '__main__':
    jobs = [(d, i) for d in (0.1, 0.3, 0.5, 0.7, 0.9) for i in range(10)]
    print('%d instances x 2 (closure stage-1 vs full AOC)' % len(jobs), flush=True)
    t0 = time.time(); rows = []
    path = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/occ_ablation.csv'
    with mp.Pool(6) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs); pd.DataFrame(rows).to_csv(path, index=False)
            if i % 10 == 0:
                print('%d/%d %.1fmin' % (i, len(jobs), (time.time()-t0)/60), flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print('done %.1f min' % ((time.time()-t0)/60), flush=True)
