"""UNTESTED CLAIM #1: a targeted-merge fallback for closure's structural failures.

When every closure draw violates All-In, the violation is always "node i in cluster X has
no edge into X's child Y". Merging X and Y removes the offending edge, and is safe because
the merged cluster inherits Y's children, of which there is at most one. Iterating should
terminate at a certified-valid partition, coarsening ONLY where forced -- far better than
collapsing to a single cluster (info = 1.0).

Measured here against: (a) does it terminate valid, (b) how coarse, (c) info vs the trivial
one-cluster fallback, (d) does it hurt on instances where closure already succeeds.
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
OUT = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/fallback.csv'


def high_mask(Ab, groups):
    K = len(groups)
    M = np.zeros((K, K), bool)
    for x in range(K):
        for y in range(K):
            if x != y and Ab[np.ix_(groups[x], groups[y])].any():
                M[x, y] = True
    return M


def first_violation(Ab, groups):
    """Return (x, y) for some cluster x whose child y is not reached by every member."""
    M = high_mask(Ab, groups)
    for x in range(len(groups)):
        for y in np.where(M[x])[0]:
            for i in groups[x]:
                if not Ab[i, groups[y]].any():
                    return x, y
    return None


def targeted_merge(Ab, groups, max_iter=10000):
    """Merge a violating cluster with its child, repeatedly, until All-In holds."""
    groups = [list(g) for g in groups]
    for it in range(max_iter):
        v = first_violation(Ab, groups)
        if v is None:
            return groups, it
        x, y = v
        merged = groups[x] + groups[y]
        groups = [g for j, g in enumerate(groups) if j not in (x, y)] + [merged]
    return groups, max_iter


def info_of(A, groups, p):
    T = np.zeros((p, len(groups)))
    for j, m in enumerate(groups):
        T[m, j] = 1
    co = T @ T.T; np.fill_diagonal(co, 0)
    return (np.abs(A) * co).sum() / np.abs(A).sum()


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
    best_valid, bg_valid = 0, None
    best_any, bg_any = 0, None
    for _ in range(12):
        g = mk.closure(A, rng)
        if len(g) > best_any:
            best_any, bg_any = len(g), g
        if all(mk.conditions(A, g)) and len(g) > best_valid:
            best_valid, bg_valid = len(g), g
    closure_ok = int(bg_valid is not None)

    t0 = time.time()
    fb, iters = targeted_merge(Ab, bg_any)
    t_fb = time.time() - t0
    ai, oc, dag = mk.conditions(A, fb)
    return dict(p=p, true_nhigh=nh, density=dens, seed=seed,
                closure_ok=closure_ok, closure_k=best_valid,
                closure_info=info_of(A, bg_valid, p) if bg_valid else np.nan,
                start_k=best_any,
                fb_valid=int(ai and oc and dag), fb_allin=int(ai), fb_onechild=int(oc),
                fb_k=len(fb), fb_info=info_of(A, fb, p), fb_merges=iters, fb_secs=t_fb,
                trivial_info=1.0)


if __name__ == '__main__':
    jobs = [(p, nh, 0.5, s) for p in (50, 100, 200) for nh in (5, 10, 20, 40)
            for s in range(10) if nh <= max(5, p // 5)]
    print(f'{len(jobs)} instances', flush=True)
    with mp.Pool(6) as pool:
        rows = [r for r in pool.map(job, jobs) if r]
    d = pd.DataFrame(rows); d.to_csv(OUT, index=False)
    print('%d instances\n' % len(d))
    f = d[d.closure_ok == 0]
    print('=== ON INSTANCES WHERE CLOSURE RETURNED NOTHING (n=%d) ===' % len(f))
    if len(f):
        print('  fallback terminates VALID : %.3f' % f.fb_valid.mean())
        print('  clusters recovered        : %.1f  (trivial fallback would give 1)' % f.fb_k.mean())
        print('  info                      : %.3f  (trivial fallback = 1.000)' % f.fb_info.mean())
        print('  merges needed             : %.1f' % f.fb_merges.mean())
        print('  wall time                 : %.4f s' % f.fb_secs.mean())
    s = d[d.closure_ok == 1]
    print('\n=== ON INSTANCES WHERE CLOSURE ALREADY SUCCEEDED (n=%d) ===' % len(s))
    print('  closure   : k %.1f  info %.3f' % (s.closure_k.mean(), s.closure_info.mean()))
    print('  fallback  : k %.1f  info %.3f  valid %.3f' % (s.fb_k.mean(), s.fb_info.mean(), s.fb_valid.mean()))
    print('\n=== COVERAGE ===')
    print('  closure alone            : %.3f' % d.closure_ok.mean())
    print('  closure + fallback       : %.3f' % ((d.closure_ok == 1) | (d.fb_valid == 1)).mean())
