"""How tightly should HailMary be pinned to the closure partition?

Pipeline as intended: take the best VALID closure partition (K clusters), hand HailMary
that exact partition at n_high=K, train. The init is `T0*margin + noise*randn`, and every
run today used margin=5.0, noise=0.1 -- a 50:1 ratio that starts training essentially at
the hard partition. 23 of 32 valid runs never moved. This sweeps that ratio.

margin=0 is a pure random init (no closure information at all) as the floor.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, importlib.util, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary
from src.utils.generate_data import sample_low_level_graph
_s = importlib.util.spec_from_file_location(
    'mk', '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts/max_k_estimate.py')
mk = importlib.util.module_from_spec(_s); _s.loader.exec_module(mk)
SP = '/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad'

CONFIGS = [(5.0, 0.1), (2.0, 0.1), (1.0, 0.1), (5.0, 0.5), (2.0, 0.5), (0.0, 1.0)]


def job(a):
    torch.set_num_threads(1)
    p, nh, seed, margin, noise = a
    A, _, _, _ = sample_low_level_graph(
        n_high=nh, n_low=p, density=0.5, shuffle=True, blockiness=0.5, heterogeneity=1.,
        internal_density=0., graph_type='random', seed=seed, connected=True, max_attempts=400)
    Ab = (A != 0); absA = np.abs(A); tot = absA.sum()
    rng = np.random.default_rng(seed)
    best, bg = 0, None
    for _ in range(12):
        g = mk.closure(A, rng)
        if all(mk.conditions(A, g)) and len(g) > best:
            best, bg = len(g), g
    if bg is None:
        return None
    K = len(bg)
    T0 = np.zeros((p, K))
    for j, mem in enumerate(bg):
        T0[mem, j] = 1.0
    co0 = T0 @ T0.T; np.fill_diagonal(co0, 0)
    info_start = (absA * co0).sum() / tot

    gen = torch.Generator().manual_seed(31337 + seed)
    li = torch.from_numpy(T0).float() * margin + torch.randn(p, K, generator=gen) * noise
    m = HailMary(n_low=p, n_high=K, A_low=torch.from_numpy(A).float(), logits_init=li)
    m.train(num_epochs=6000, lambda_recon=0., lambda_graph=10., lambda_one_child=100.,
            lambda_surj=0., lambda_info=10., verbose=False)
    T = m.get_partition()[0].detach().numpy()
    lab = T.argmax(1)
    groups = [np.where(lab == j)[0] for j in range(K)]
    groups = [g for g in groups if len(g)]
    Kc = len(groups)
    M = np.zeros((Kc, Kc), bool)
    for x in range(Kc):
        for y in range(Kc):
            if x != y and Ab[np.ix_(groups[x], groups[y])].any():
                M[x, y] = True
    bad = set()
    for x in range(Kc):
        for y in np.where(M[x])[0]:
            for i in groups[x]:
                if not Ab[i, groups[y]].any():
                    bad.add(int(i))
    co = T @ T.T; np.fill_diagonal(co, 0)
    ok = int(len(bad) == 0 and (M.sum(1).max() <= 1 if Kc else True))
    return dict(p=p, true_nhigh=nh, seed=seed, margin=margin, noise=noise, K=K,
                success=ok, viol_nodes=len(bad), one_child=int(M.sum(1).max() <= 1) if Kc else 1,
                info=(absA * co).sum() / tot, info_start=info_start,
                moved=int((T0.argmax(1) != lab).sum()), used=Kc)


if __name__ == '__main__':
    jobs = [(p, nh, s, mg, nz) for p in (50, 100) for nh in (10, 20)
            for s in range(8) for mg, nz in CONFIGS]
    print(f'{len(jobs)} runs', flush=True)
    t0 = time.time(); rows = []
    with mp.Pool(6) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            if r:
                rows.append(r)
            if i % 40 == 0:
                print(f'{i}/{len(jobs)} {(time.time()-t0)/60:.1f}min', flush=True)
                pd.DataFrame(rows).to_csv(f'{SP}/marginsweep.csv', index=False)
    d = pd.DataFrame(rows); d.to_csv(f'{SP}/marginsweep.csv', index=False)
    d['cfg'] = 'm' + d.margin.astype(str) + '/n' + d.noise.astype(str)
    print('\n%d runs (closure returned a valid partition)\n' % len(d))
    print('closure baseline: success 1.000, info %.3f\n' % d.info_start.mean())
    g = d.groupby('cfg').agg(n=('success', 'size'), success=('success', 'mean'),
                             info=('info', 'mean'), moved=('moved', 'mean'),
                             viol=('viol_nodes', 'mean'), used=('used', 'mean'))
    print(g.round(3).to_string())
    print('\nconditional on staying valid:')
    s = d[d.success == 1]
    print(s.groupby('cfg').agg(n=('info', 'size'), info=('info', 'mean'),
                               start=('info_start', 'mean'), moved=('moved', 'mean')).round(3).to_string())
    print('\nwith fallback to the closure partition when invalid:')
    d['hyb'] = np.where(d.success == 1, np.minimum(d.info, d.info_start), d.info_start)
    print(d.groupby('cfg').hyb.mean().round(4).to_string())
