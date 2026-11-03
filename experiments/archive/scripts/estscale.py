"""Estimator alone on random DAGs: failure rate and empirical time scaling.

Records, per instance: whether ANY of the n_orders closure draws yielded a valid
(All-In + one-child + acyclic) partition, how many of the draws individually failed,
the resulting ceiling, and wall-time -- so we can fit t ~ p^a * k_max^b.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, importlib.util, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, multiprocessing as mp
from src.utils.generate_data import sample_low_level_graph
_s = importlib.util.spec_from_file_location(
    'mk', '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts/max_k_estimate.py')
mk = importlib.util.module_from_spec(_s); _s.loader.exec_module(mk)
SP = '/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad'
N_ORDERS = 12


def job(a):
    gt, d, p, nh, seed = a
    A, _, _, _ = sample_low_level_graph(
        n_high=nh, n_low=p, density=d, shuffle=True, blockiness=0.5, heterogeneity=1.,
        internal_density=0., graph_type=gt, seed=seed, connected=True, max_attempts=400)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    draws_ok = 0
    best, best_g = 0, None
    t_closure = 0.0
    for _ in range(N_ORDERS):
        t1 = time.time(); g = mk.closure(A, rng); t_closure += time.time() - t1
        if not all(mk.conditions(A, g)):
            continue
        draws_ok += 1
        g = mk.refine_by_splitting(A, g, rng)
        if len(g) > best:
            best, best_g = len(g), g
    el = time.time() - t0
    info = np.nan
    if best_g is not None:
        T = np.zeros((p, len(best_g)))
        for j, m in enumerate(best_g):
            T[m, j] = 1
        co = T @ T.T; np.fill_diagonal(co, 0)
        info = (np.abs(A) * co).sum() / np.abs(A).sum()
    return dict(graph_type=gt, p=p, true_nhigh=nh, seed=seed, returned=int(best > 0),
                draws_ok=draws_ok, draw_fail_rate=1 - draws_ok / N_ORDERS,
                ceiling=best, info=info, secs=el, secs_closure=t_closure,
                edges=int((A != 0).sum()), max_outdeg=int((A != 0).sum(1).max()))


if __name__ == '__main__':
    jobs = [('random', 0.5, p, nh, s)
            for p in (50, 100, 200, 400) for nh in (5, 10, 20, 40)
            for s in range(20) if nh <= max(5, p // 5)]
    print(f'{len(jobs)} instances', flush=True)
    t0 = time.time(); rows = []
    with mp.Pool(3) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.append(r)
            if i % 25 == 0:
                print(f'{i}/{len(jobs)}  {(time.time()-t0)/60:.1f}min', flush=True)
                pd.DataFrame(rows).to_csv(f'{SP}/estscale.csv', index=False)
    pd.DataFrame(rows).to_csv(f'{SP}/estscale.csv', index=False)
    print(f'done {(time.time()-t0)/60:.1f} min')
