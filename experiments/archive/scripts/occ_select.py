"""OC Closure draw selection: widest valid draw (current) vs minimum-intra valid draw.

Computing intra-cluster edge mass for a draw is O(sum_x |X_x|^2) <= O(p^2) -- microseconds
at p=100, negligible beside generating the draw. Width is only a proxy for information
loss, so selecting directly on intra should dominate it. This measures by how much.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, multiprocessing as mp
from src.utils.generate_data import sample_low_level_graph
from occ import one_child_closure, conditions

P, N_DRAWS = 100, 12


def intra_of(absA, tot, groups):
    s = 0.0
    for g in groups:
        if len(g) > 1:
            sub = absA[np.ix_(g, g)]
            s += sub.sum() - np.trace(sub)
    return s / tot


def job(a):
    dens, seed = a
    try:
        A, *_ = sample_low_level_graph(n_high=10, n_low=P, density=dens, shuffle=True,
            blockiness=0.5, heterogeneity=1., internal_density=0., graph_type='random',
            seed=seed, connected=True, max_attempts=300)
    except Exception:
        return []
    absA = np.abs(A); tot = absA.sum()
    rng = np.random.default_rng(seed)
    draws = []
    t_draw = t_intra = 0.0
    for _ in range(N_DRAWS):
        t0 = time.time(); g = one_child_closure(A, rng); t_draw += time.time() - t0
        t0 = time.time(); iv = intra_of(absA, tot, g); t_intra += time.time() - t0
        draws.append((g, len(g), iv, all(conditions(A, g))))
    valid = [d for d in draws if d[3]]
    pool = valid if valid else draws
    widest = max(pool, key=lambda d: d[1])
    minintra = min(pool, key=lambda d: d[2])
    return [dict(density=dens, seed=seed, n_valid=len(valid),
                 k_widest=widest[1], intra_widest=widest[2],
                 k_minintra=minintra[1], intra_minintra=minintra[2],
                 intra_best_any=min(d[2] for d in draws),
                 k_spread=max(d[1] for d in pool) - min(d[1] for d in pool),
                 intra_spread=max(d[2] for d in pool) - min(d[2] for d in pool),
                 t_draw=t_draw, t_intra=t_intra)]


if __name__ == '__main__':
    jobs = [(d, i) for d in (0.1, 0.3, 0.5, 0.7, 0.9) for i in range(10)]
    rows = []
    with mp.Pool(6) as pool:
        for rs in pool.imap_unordered(job, jobs):
            rows.extend(rs)
    d = pd.DataFrame(rows)
    d.to_csv('/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/occ_select.csv', index=False)
    print('instances %d\n' % len(d))
    print('COST: intra scoring as a fraction of draw generation: %.4f%%'
          % (100 * d.t_intra.sum() / d.t_draw.sum()))
    print('  (%.1f ms total for %d draws vs %.1f s generating them)'
          % (1000 * d.t_intra.sum(), N_DRAWS * len(d), d.t_draw.sum()))
    print('\nSELECTION: widest valid draw (current) vs min-intra valid draw')
    t = d.groupby('density').agg(n=('seed', 'size'), n_valid=('n_valid', 'median'),
        k_widest=('k_widest', 'median'), intra_widest=('intra_widest', 'median'),
        k_minintra=('k_minintra', 'median'), intra_minintra=('intra_minintra', 'median'),
        intra_spread=('intra_spread', 'median')).round(3)
    t['improvement'] = (t.intra_widest - t.intra_minintra).round(3)
    print(t.to_string())
    imp = d.intra_widest - d.intra_minintra
    print('\npooled: intra %.3f -> %.3f  (median improvement %+.3f, better on %d/%d, worse on %d)'
          % (d.intra_widest.median(), d.intra_minintra.median(), np.median(imp),
             (imp > 1e-9).sum(), len(d), (imp < -1e-9).sum()))
    print('k cost of switching: %.1f -> %.1f clusters' % (d.k_widest.median(), d.k_minintra.median()))
