"""Does closure's success fall as random-DAG density drops?

Hypothesis (user's): high density means that satisfying one-child also satisfies All-In
with high probability, because every node has many out-edges and is likely to hit the
child cluster. Lower density -> nodes more often 'locally sunk' -> All-In fails.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, importlib.util, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, multiprocessing as mp
from src.utils.generate_data import sample_low_level_graph
_s = importlib.util.spec_from_file_location(
    'mk', '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts/max_k_estimate.py')
mk = importlib.util.module_from_spec(_s); _s.loader.exec_module(mk)
SP = '/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad'


def job(a):
    p, nh, dens, seed = a
    try:
        A, _, _, _ = sample_low_level_graph(
            n_high=nh, n_low=p, density=dens, shuffle=True, blockiness=0.5,
            heterogeneity=1., internal_density=0., graph_type='random', seed=seed,
            connected=True, max_attempts=400)
    except Exception:
        return None
    Ab = (A != 0)
    rng = np.random.default_rng(seed)
    ok = 0
    best, bg = 0, None
    for _ in range(12):
        g = mk.closure(A, rng)
        if all(mk.conditions(A, g)):
            ok += 1
            if len(g) > best:
                best, bg = len(g), g
    info = np.nan
    if bg is not None:
        T = np.zeros((p, len(bg)))
        for j, m in enumerate(bg):
            T[m, j] = 1
        co = T @ T.T; np.fill_diagonal(co, 0)
        info = (np.abs(A) * co).sum() / np.abs(A).sum()
    return dict(p=p, true_nhigh=nh, density=dens, seed=seed, returned=int(best > 0),
                draws_ok=ok, ceiling=best, info=info,
                real_density=float(Ab.mean()), mean_outdeg=float(Ab.sum(1).mean()),
                frac_zero_out=float((Ab.sum(1) == 0).mean()))


if __name__ == '__main__':
    jobs = [(p, nh, d, s) for p in (50, 100) for nh in (10,)
            for d in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9) for s in range(15)]
    with mp.Pool(6) as pool:
        rows = [r for r in pool.map(job, jobs) if r]
    d = pd.DataFrame(rows); d.to_csv(f'{SP}/density.csv', index=False)
    print('%d instances\n' % len(d))
    print(d.groupby(['p', 'density']).agg(
        n=('returned', 'size'), success=('returned', 'mean'),
        draws_ok=('draws_ok', 'mean'), real_dens=('real_density', 'mean'),
        outdeg=('mean_outdeg', 'mean'), sinks=('frac_zero_out', 'mean'),
        ceiling=('ceiling', 'mean'), info=('info', 'mean')).round(3).to_string())
