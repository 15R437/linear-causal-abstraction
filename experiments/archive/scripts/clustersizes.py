"""What do closure's partitions actually look like? Full cluster-size distributions.

Records every cluster size (not just the max), plus shape summaries: fraction of nodes in
the largest cluster, singleton count, Gini, and the normalised entropy of the size
distribution (1 = perfectly balanced, 0 = one blob).
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, importlib.util, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, multiprocessing as mp
from src.utils.generate_data import sample_low_level_graph
B = '/Users/issy/Projects/linear-causal-abstraction'
_s = importlib.util.spec_from_file_location('mk', f'{B}/experiments/abstraction_study/scripts/max_k_estimate.py')
mk = importlib.util.module_from_spec(_s); _s.loader.exec_module(mk)
_f = importlib.util.spec_from_file_location('fb', f'{B}/experiments/abstraction_study/scripts/fallback.py')
fb = importlib.util.module_from_spec(_f); _f.loader.exec_module(fb)


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if x.sum() else 0.


def job(a):
    gt, dens, p, nh, seed = a
    try:
        A, _, _, _ = sample_low_level_graph(
            n_high=nh, n_low=p, density=dens, shuffle=True, blockiness=0.5,
            heterogeneity=1., internal_density=0., graph_type=gt, seed=seed,
            connected=True, max_attempts=400)
    except Exception:
        return None
    Ab = (A != 0)
    rng = np.random.default_rng(seed)
    best, bg, bany, ba = 0, None, 0, None
    for _ in range(12):
        g = mk.closure(A, rng)
        if len(g) > bany:
            bany, ba = len(g), g
        if all(mk.conditions(A, g)) and len(g) > best:
            best, bg = len(g), g
    used_fb = int(bg is None)
    if bg is None:
        bg, _ = fb.targeted_merge(Ab, ba)
    sizes = np.array(sorted(len(x) for x in bg), dtype=float)
    q = sizes / sizes.sum()
    ent = float(-(q * np.log(q)).sum() / np.log(len(q))) if len(q) > 1 else 0.
    return dict(graph_type=gt, density=dens, p=p, true_nhigh=nh, seed=seed,
                k=len(sizes), used_fallback=used_fb,
                largest=int(sizes.max()), largest_frac=float(sizes.max() / p),
                median_size=float(np.median(sizes)), mean_size=float(sizes.mean()),
                singletons=int((sizes == 1).sum()),
                singleton_frac=float((sizes == 1).mean()),
                q25=float(np.percentile(sizes, 25)), q75=float(np.percentile(sizes, 75)),
                gini=gini(sizes), entropy=ent,
                sizes=';'.join(str(int(x)) for x in sizes))


if __name__ == '__main__':
    jobs = [(gt, d, p, nh, s)
            for gt, d in [('random', 0.1), ('random', 0.3), ('random', 0.5),
                          ('random', 0.7), ('random', 0.9), ('one_child', 1.0)]
            for p in (50, 100, 200) for nh in (5, 10, 20) for s in range(8)
            if nh <= max(5, p // 5)]
    print(f'{len(jobs)} instances', flush=True)
    with mp.Pool(6) as pool:
        rows = [r for r in pool.map(job, jobs) if r]
    d = pd.DataFrame(rows)
    d.to_csv(f'{B}/experiments/abstraction_study/data/clustersizes.csv', index=False)
    print('%d instances\n' % len(d))
    print('=== shape of closure partitions ===')
    print(d.groupby(['graph_type', 'density']).agg(
        n=('k', 'size'), k=('k', 'mean'), mean_size=('mean_size', 'mean'),
        median_size=('median_size', 'mean'), largest=('largest', 'mean'),
        largest_frac=('largest_frac', 'mean'), singleton_frac=('singleton_frac', 'mean'),
        gini=('gini', 'mean'), entropy=('entropy', 'mean')).round(3).to_string())
    print('\n=== by p (random only) ===')
    r = d[d.graph_type == 'random']
    print(r.groupby('p').agg(n=('k', 'size'), k=('k', 'mean'), mean_size=('mean_size', 'mean'),
                             median_size=('median_size', 'mean'), largest=('largest', 'mean'),
                             largest_frac=('largest_frac', 'mean'),
                             singleton_frac=('singleton_frac', 'mean')).round(3).to_string())
    print('\n=== pooled size histogram (all instances) ===')
    allsz = np.concatenate([np.array(s.split(';'), int) for s in d.sizes])
    for lo, hi in [(1, 1), (2, 2), (3, 5), (6, 10), (11, 25), (26, 50), (51, 100), (101, 10**9)]:
        m = (allsz >= lo) & (allsz <= hi)
        lab = f'{lo}' if lo == hi else (f'{lo}+' if hi > 10**8 else f'{lo}-{hi}')
        print('  size %-8s %6d clusters  %5.1f%% of clusters  %5.1f%% of nodes'
              % (lab, m.sum(), 100 * m.mean(), 100 * allsz[m].sum() / allsz.sum()))
