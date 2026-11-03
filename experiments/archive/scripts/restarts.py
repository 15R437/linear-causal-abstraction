"""Closure (correctly selected) vs HailMary with RESTARTS, across density.

The single-restart comparison understates HailMary badly, because validity is CHEAP TO
VERIFY (O(p^2)). You never have to ship an invalid partition: run r restarts, keep the
first valid one. So the honest metric is success WITHIN A BUDGET, not success per restart
-- and the earlier runs all used n_restarts=1.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, importlib.util, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary, closure_logits
from src.utils.generate_data import sample_low_level_graph
from src.utils.losses import graph_loss
_s = importlib.util.spec_from_file_location(
    'mk', '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts/max_k_estimate.py')
mk = importlib.util.module_from_spec(_s); _s.loader.exec_module(mk)
SP = '/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad'
R = 5


def score(A, T):
    M = (T.T @ (A != 0).astype(float) @ T) > 0
    np.fill_diagonal(M, False)
    co = T @ T.T; np.fill_diagonal(co, 0)
    ok = int(graph_loss(A_low=A, T_hard=T) < 1e-8 and M.sum(1).max() <= 1)
    return ok, (np.abs(A) * co).sum() / np.abs(A).sum()


def job(a):
    torch.set_num_threads(1)
    p, dens, seed = a
    try:
        A, _, _, _ = sample_low_level_graph(
            n_high=10, n_low=p, density=dens, shuffle=True, blockiness=0.5,
            heterogeneity=1., internal_density=0., graph_type='random', seed=seed,
            connected=True, max_attempts=400)
    except Exception:
        return None
    # ---- closure, selecting the best VALID draw ----
    rng = np.random.default_rng(seed)
    best, bg = 0, None
    for _ in range(12):
        g = mk.closure(A, rng)
        if all(mk.conditions(A, g)) and len(g) > best:
            best, bg = len(g), g
    clo_ok, clo_info = (0, np.nan)
    if bg is not None:
        T = np.zeros((p, len(bg)))
        for j, m in enumerate(bg):
            T[m, j] = 1
        clo_ok, clo_info = score(A, T)
    k = max(2, best if best else 10)
    # ---- HailMary, R independent restarts; keep valid ones ----
    t0 = time.time()
    firsts, infos = [], []
    for r in range(R):
        gen = torch.Generator().manual_seed(31337 + 1000 * seed + r)
        li = closure_logits(A, k, rng=np.random.default_rng(31337 + 1000 * seed + r),
                            generator=gen)
        m = HailMary(n_low=p, n_high=k, A_low=torch.from_numpy(A).float(), logits_init=li)
        m.train(num_epochs=6000, lambda_recon=0., lambda_graph=10., lambda_one_child=100.,
                lambda_surj=0., lambda_info=10., verbose=False)
        ok, inf = score(A, m.get_partition()[0].detach().numpy())
        firsts.append(ok)
        if ok:
            infos.append(inf)
    return dict(p=p, density=dens, seed=seed, k=k, clo_ok=clo_ok, clo_info=clo_info,
                hm1=firsts[0], hm_any=int(any(firsts)), n_valid=int(sum(firsts)),
                hm_info_best=(min(infos) if infos else np.nan), secs=time.time() - t0)


if __name__ == '__main__':
    jobs = [(p, d, s) for p in (50, 100) for d in (0.1, 0.2, 0.3, 0.5) for s in range(10)]
    print(f'{len(jobs)} instances x {R} restarts', flush=True)
    t0 = time.time(); rows = []
    with mp.Pool(6) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            if r:
                rows.append(r)
            if i % 10 == 0:
                print(f'{i}/{len(jobs)} {(time.time()-t0)/60:.1f}min', flush=True)
                pd.DataFrame(rows).to_csv(f'{SP}/restarts.csv', index=False)
    d = pd.DataFrame(rows); d.to_csv(f'{SP}/restarts.csv', index=False)
    print('\n%d instances\n' % len(d))
    print(d.groupby(['p', 'density']).agg(
        n=('clo_ok', 'size'), clo_succ=('clo_ok', 'mean'),
        hm_1restart=('hm1', 'mean'), hm_5restarts=('hm_any', 'mean'),
        clo_info=('clo_info', 'mean'), hm_info=('hm_info_best', 'mean'),
        secs=('secs', 'mean')).round(3).to_string())
