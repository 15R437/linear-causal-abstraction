"""Score the max-k estimator ITSELF as a training-free baseline, on the same instances
as kpolicy.py. It returns a certified-valid partition (or nothing), so the question is
not validity but information loss and how often it returns nothing."""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
import sys, time, importlib.util, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, multiprocessing as mp
from src.utils.generate_data import sample_low_level_graph
from src.utils.losses import graph_loss, reconstruction_loss
from src.utils.helper import fit_abstraction

_s = importlib.util.spec_from_file_location(
    'mk', '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts/max_k_estimate.py')
mk = importlib.util.module_from_spec(_s); _s.loader.exec_module(mk)
SP = '/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad'


def job(a):
    gt, d, p, nh, seed, refine = a
    A, _, _, _ = sample_low_level_graph(
        n_high=nh, n_low=p, density=d, shuffle=True, blockiness=0.5, heterogeneity=1.,
        internal_density=0., graph_type=gt, seed=seed, connected=True, max_attempts=400)
    t0 = time.time()
    est, g = mk.estimate_max_k(A, n_orders=12, seed=seed, refine=refine)
    el = time.time() - t0
    row = dict(policy='closure+split' if refine else 'closure_only', graph_type=gt, p=p,
               true_nhigh=nh, seed=seed, returned=int(g is not None), k=est, secs=el)
    if g is None:
        return {**row, 'success': 0, 'info': np.nan, 'recon': np.nan, 'max_clust': np.nan}
    T = np.zeros((p, len(g)))
    for j, m in enumerate(g):
        T[m, j] = 1
    M = (T.T @ (A != 0).astype(float) @ T) > 0
    np.fill_diagonal(M, False)
    co = T @ T.T; np.fill_diagonal(co, 0)
    gl = graph_loss(A_low=A, T_hard=T)
    ok = int(gl < 1e-8 and M.sum(1).max() <= 1)
    recon = np.nan
    if ok:
        try:
            A_high, Tc = fit_abstraction(A, T, n_restarts=3, seed=seed, maxiter=60)
            recon = reconstruction_loss(A, Tc, A_high)
        except Exception:
            pass
    return {**row, 'success': ok, 'info': (np.abs(A) * co).sum() / np.abs(A).sum(),
            'recon': recon, 'max_clust': int(T.sum(0).max())}


if __name__ == '__main__':
    jobs = []
    for gt, d in [('random', 0.5), ('one_child', 1.0)]:
        for p in (50, 100, 200):
            for nh in sorted({max(3, p // 20), max(3, p // 10), max(3, p // 5)}):
                for s in range(5):
                    for refine in (True, False):
                        jobs.append((gt, d, p, nh, s, refine))
    print(f'{len(jobs)} runs', flush=True)
    t0 = time.time()
    rows = []
    with mp.Pool(6) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.append(r)
            if i % 10 == 0:
                print(f'{i}/{len(jobs)}  {(time.time()-t0)/60:.1f}min', flush=True)
                pd.DataFrame(rows).to_csv(f'{SP}/ceilbase.csv', index=False)
    df = pd.DataFrame(rows); df.to_csv(f'{SP}/ceilbase.csv', index=False)
    print(f'done {(time.time()-t0)/60:.1f} min')
