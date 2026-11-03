"""Does HailMary produce a LESS blobby partition than closure, at the same k?

Closure's shape is extreme: largest cluster 58-82% of nodes on random DAGs. HailMary
optimises intra-cluster edge mass directly, and balance is one way to reduce it -- so if
anything can break up the blob, it should. Paired on identical instances at n_high = K.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, importlib.util, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary
from src.utils.generate_data import sample_low_level_graph
B = '/Users/issy/Projects/linear-causal-abstraction'
_s = importlib.util.spec_from_file_location('mk', f'{B}/experiments/abstraction_study/scripts/max_k_estimate.py')
mk = importlib.util.module_from_spec(_s); _s.loader.exec_module(mk)
_f = importlib.util.spec_from_file_location('fb', f'{B}/experiments/abstraction_study/scripts/fallback.py')
fb = importlib.util.module_from_spec(_f); _f.loader.exec_module(fb)


def shape(sizes, p):
    sizes = np.array(sorted(sizes), float)
    q = sizes / sizes.sum()
    ent = float(-(q * np.log(q)).sum() / np.log(len(q))) if len(q) > 1 else 0.
    x = np.sort(sizes); n = len(x)
    g = float((2*np.arange(1, n+1) - n - 1).dot(x) / (n*x.sum())) if x.sum() else 0.
    return dict(k=len(sizes), largest=int(sizes.max()), largest_frac=float(sizes.max()/p),
                singleton_frac=float((sizes == 1).mean()), entropy=ent, gini=g,
                median_size=float(np.median(sizes)))


def job(a):
    torch.set_num_threads(1)
    gt, dens, p, nh, seed = a
    try:
        A, _, _, _ = sample_low_level_graph(
            n_high=nh, n_low=p, density=dens, shuffle=True, blockiness=0.5,
            heterogeneity=1., internal_density=0., graph_type=gt, seed=seed,
            connected=True, max_attempts=400)
    except Exception:
        return None
    Ab = (A != 0); absA = np.abs(A); tot = absA.sum()
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
    K = len(bg)
    T0 = np.zeros((p, K))
    for j, m in enumerate(bg):
        T0[m, j] = 1
    co0 = T0 @ T0.T; np.fill_diagonal(co0, 0)
    row = {('clo_' + k): v for k, v in shape([len(x) for x in bg], p).items()}
    row['clo_info'] = (absA * co0).sum() / tot

    gen = torch.Generator().manual_seed(31337 + seed)
    li = torch.from_numpy(T0).float() * 5.0 + torch.randn(p, K, generator=gen) * 0.1
    m = HailMary(n_low=p, n_high=K, A_low=torch.from_numpy(A).float(), logits_init=li)
    m.train(num_epochs=6000, lambda_recon=0., lambda_graph=10., lambda_one_child=100.,
            lambda_surj=0., lambda_info=10., verbose=False)
    T = m.get_partition()[0].detach().numpy()
    gh = [g for g in (np.where(T[:, j] == 1)[0] for j in range(K)) if len(g)]
    ai, oc, dag = mk.conditions(A, gh)
    co = T @ T.T; np.fill_diagonal(co, 0)
    row.update({('hm_' + k): v for k, v in shape([len(x) for x in gh], p).items()})
    row['hm_info'] = (absA * co).sum() / tot
    row['hm_valid'] = int(ai and oc and dag)
    row.update(graph_type=gt, density=dens, p=p, true_nhigh=nh, seed=seed, K=K)
    return row


if __name__ == '__main__':
    jobs = [(gt, d, p, nh, s)
            for gt, d in [('random', 0.1), ('random', 0.3), ('random', 0.5),
                          ('random', 0.7), ('random', 0.9), ('one_child', 1.0)]
            for p in (50, 100) for nh in (5, 10, 20) for s in range(5)
            if nh <= max(5, p // 5)]
    print(f'{len(jobs)} instances', flush=True)
    with mp.Pool(6) as pool:
        rows = [r for r in pool.map(job, jobs) if r]
    d = pd.DataFrame(rows)
    d.to_csv(f'{B}/experiments/abstraction_study/data/hm_sizes.csv', index=False)
    print('%d instances,  HailMary stayed valid in %.3f\n' % (len(d), d.hm_valid.mean()))
    v = d[d.hm_valid == 1]
    print('=== PAIRED SHAPE, on instances where HailMary stayed valid (n=%d) ===' % len(v))
    print(v.groupby(['graph_type', 'density']).agg(
        n=('K', 'size'), K=('K', 'mean'),
        clo_largest=('clo_largest_frac', 'mean'), hm_largest=('hm_largest_frac', 'mean'),
        clo_ent=('clo_entropy', 'mean'), hm_ent=('hm_entropy', 'mean'),
        clo_k=('clo_k', 'mean'), hm_k=('hm_k', 'mean'),
        clo_info=('clo_info', 'mean'), hm_info=('hm_info', 'mean')).round(3).to_string())
    print('\n=== overall paired deltas (HailMary - closure) ===')
    for c in ('largest_frac', 'entropy', 'singleton_frac', 'k', 'info'):
        a, b = v['clo_' + c] if c != 'info' else v['clo_info'], v['hm_' + c] if c != 'info' else v['hm_info']
        dd = b - a
        print('  %-15s closure %7.3f -> hailmary %7.3f   delta %+.4f  (better on %d/%d)'
              % (c, a.mean(), b.mean(), dd.mean(),
                 (dd < -1e-9).sum() if c in ('largest_frac', 'singleton_frac', 'info') else (dd > 1e-9).sum(),
                 len(dd)))
