"""IS THE BLOB INHERENT? Exhaustive answer at small p.

Enumerate EVERY partition of a small graph, keep the ones satisfying All-In + one-child,
and compare the shape of the best-balanced valid partition against the one closure returns.

  * if every valid partition is blobby        -> the blob is forced by the conditions
  * if balanced valid partitions exist        -> closure is leaving quality on the table,
                                                 and a better search could find it
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, importlib.util, warnings, itertools
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, multiprocessing as mp
from src.utils.generate_data import sample_low_level_graph
B = '/Users/issy/Projects/linear-causal-abstraction'
_s = importlib.util.spec_from_file_location('mk', f'{B}/experiments/abstraction_study/scripts/max_k_estimate.py')
mk = importlib.util.module_from_spec(_s); _s.loader.exec_module(mk)
_f = importlib.util.spec_from_file_location('fb', f'{B}/experiments/abstraction_study/scripts/fallback.py')
fb = importlib.util.module_from_spec(_f); _f.loader.exec_module(fb)


def partitions(collection):
    collection = list(collection)
    if len(collection) == 1:
        yield [collection]
        return
    first = collection[0]
    for smaller in partitions(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n+1:]
        yield [[first]] + smaller


def balance(sizes):
    """normalised entropy: 1 = perfectly even, 0 = one blob"""
    q = np.asarray(sizes, float); q = q / q.sum()
    return float(-(q * np.log(q)).sum() / np.log(len(q))) if len(q) > 1 else 0.


def job(a):
    p, nh, dens, seed = a
    try:
        A, _, _, _ = sample_low_level_graph(
            n_high=nh, n_low=p, density=dens, shuffle=True, blockiness=0.5,
            heterogeneity=1., internal_density=0., graph_type='random', seed=seed,
            connected=True, max_attempts=300)
    except Exception:
        return None
    Ab = (A != 0); absA = np.abs(A); tot = absA.sum()

    def info_of(groups):
        T = np.zeros((p, len(groups)))
        for j, m in enumerate(groups):
            T[m, j] = 1
        co = T @ T.T; np.fill_diagonal(co, 0)
        return float((absA * co).sum() / tot)

    # --- closure's answer ---
    rng = np.random.default_rng(seed)
    best, bg, bany, ba = 0, None, 0, None
    for _ in range(12):
        g = mk.closure(A, rng)
        if len(g) > bany:
            bany, ba = len(g), g
        if all(mk.conditions(A, g)) and len(g) > best:
            best, bg = len(g), g
    if bg is None:
        bg, _ = fb.targeted_merge(Ab, ba)
    clo_sizes = sorted(len(x) for x in bg)
    clo_k = len(bg)

    # --- every valid partition ---
    valid = []
    for part in partitions(range(p)):
        ai, oc, dag = mk.conditions(A, [np.array(x) for x in part])
        if ai and oc and dag:
            sizes = sorted(len(x) for x in part)
            valid.append((len(part), max(sizes), balance(sizes), info_of([np.array(x) for x in part])))
    if not valid:
        return None
    V = pd.DataFrame(valid, columns=['k', 'largest', 'bal', 'info'])
    same_k = V[V.k == clo_k]
    max_k = V.k.max()
    at_max = V[V.k == max_k]
    return dict(p=p, true_nhigh=nh, density=dens, seed=seed,
                n_valid=len(V), max_k=int(max_k), clo_k=clo_k,
                clo_largest=max(clo_sizes), clo_bal=balance(clo_sizes), clo_info=info_of(bg),
                # best available AT CLOSURE'S k
                bestbal_at_clo_k=float(same_k.bal.max()) if len(same_k) else np.nan,
                minlargest_at_clo_k=int(same_k.largest.min()) if len(same_k) else -1,
                bestinfo_at_clo_k=float(same_k['info'].min()) if len(same_k) else np.nan,
                n_at_clo_k=len(same_k),
                # best available anywhere
                bestbal_any=float(V.bal.max()), minlargest_any=int(V.largest.min()),
                bestinfo_any=float(V['info'].min()),
                # at the true ceiling
                bestbal_at_maxk=float(at_max.bal.max()), minlargest_at_maxk=int(at_max.largest.min()))


if __name__ == '__main__':
    jobs = [(p, nh, d, s) for p in (8, 9, 10) for nh in (3, 4)
            for d in (0.3, 0.5, 0.9) for s in range(6)]
    print(f'{len(jobs)} instances (Bell(10)=115975 partitions each)', flush=True)
    with mp.Pool(6) as pool:
        rows = [r for r in pool.map(job, jobs) if r]
    d = pd.DataFrame(rows)
    d.to_csv(f'{B}/experiments/abstraction_study/data/blob_bruteforce.csv', index=False)
    print('%d instances\n' % len(d))
    print('=== does closure find the MOST BALANCED valid partition at its own k? ===')
    print('  closure balance           %.3f' % d.clo_bal.mean())
    print('  best balance at same k    %.3f' % d.bestbal_at_clo_k.mean())
    print('  closure is optimal in     %.3f of instances'
          % np.isclose(d.clo_bal, d.bestbal_at_clo_k, atol=1e-9).mean())
    print()
    print('  closure largest cluster   %.2f' % d.clo_largest.mean())
    print('  smallest possible at k    %.2f' % d.minlargest_at_clo_k.mean())
    print()
    print('=== is a BALANCED valid partition available at all? ===')
    print('  best balance over ALL valid partitions   %.3f' % d.bestbal_any.mean())
    print('  smallest achievable largest cluster      %.2f  (p = %.1f)'
          % (d.minlargest_any.mean(), d.p.mean()))
    print('  best balance at the true ceiling k       %.3f' % d.bestbal_at_maxk.mean())
    print()
    print('=== information loss ===')
    print('  closure info              %.3f' % d.clo_info.mean())
    print('  best info at same k       %.3f' % d.bestinfo_at_clo_k.mean())
    print('  best info over all valid  %.3f' % d.bestinfo_any.mean())
    print()
    print('  closure k %.2f   true max k %.2f   valid partitions %.0f'
          % (d.clo_k.mean(), d.max_k.mean(), d.n_valid.mean()))
