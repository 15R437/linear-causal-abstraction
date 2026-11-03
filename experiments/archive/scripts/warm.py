"""Arm 3 of the baseline comparison: HailMary WARM-STARTED from the refined
(closure+greedy-split) partition, at k = size of that partition.

Arm 1 (baseline alone)      -> ceilbase.csv, policy='closure+split'
Arm 2 (HailMary alone)      -> kpolicy.csv,  policy='ceiling'  (closure init, same k)
Arm 3 (baseline + HailMary) -> this file
Identical instances and seeds across all three, so they join on
(graph_type, p, true_nhigh, seed).
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
import sys, time, importlib.util, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary
from src.utils.generate_data import sample_low_level_graph
from src.utils.losses import graph_loss, reconstruction_loss
from src.utils.helper import fit_abstraction

_s = importlib.util.spec_from_file_location(
    'mk', '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts/max_k_estimate.py')
mk = importlib.util.module_from_spec(_s); _s.loader.exec_module(mk)
SP = '/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad'
EPOCHS = 6000


def groups_logits(groups, p, n_high, margin=5.0, noise=0.1, generator=None):
    """Same encoding as closure_logits but from an EXPLICIT partition."""
    T0 = np.zeros((p, n_high))
    for j, mem in enumerate(groups[:n_high]):
        T0[mem, j] = 1.0
    return (torch.from_numpy(T0).float() * margin
            + torch.randn(p, n_high, generator=generator) * noise)


def job(a):
    torch.set_num_threads(1)
    gt, d, p, nh, seed = a
    A, _, _, _ = sample_low_level_graph(
        n_high=nh, n_low=p, density=d, shuffle=True, blockiness=0.5, heterogeneity=1.,
        internal_density=0., graph_type=gt, seed=seed, connected=True, max_attempts=400)
    est, g = mk.estimate_max_k(A, n_orders=12, seed=seed)
    if g is None:
        return dict(policy='baseline+hm', graph_type=gt, p=p, true_nhigh=nh, seed=seed,
                    ceiling=est, k=0, returned=0, success=0, graph_ok=0, one_child=0,
                    used=0, info=np.nan, recon=np.nan, max_clust=np.nan,
                    info_start=np.nan, k_start=0)
    k = len(g)
    T0 = np.zeros((p, k))
    for j, mem in enumerate(g):
        T0[mem, j] = 1.0
    co0 = T0 @ T0.T; np.fill_diagonal(co0, 0)
    info_start = (np.abs(A) * co0).sum() / np.abs(A).sum()

    gen = torch.Generator().manual_seed(31337 + seed)
    li = groups_logits(g, p, k, generator=gen)
    m = HailMary(n_low=p, n_high=k, A_low=torch.from_numpy(A).float(), logits_init=li)
    m.train(num_epochs=EPOCHS, lambda_recon=0., lambda_graph=10., lambda_one_child=100.,
            lambda_surj=0., lambda_info=10., verbose=False)
    T = m.get_partition()[0].detach().numpy()
    M = (T.T @ (A != 0).astype(float) @ T) > 0
    np.fill_diagonal(M, False)
    co = T @ T.T; np.fill_diagonal(co, 0)
    gl = graph_loss(A_low=A, T_hard=T)
    oc = int(M.sum(1).max() <= 1)
    ok = int(gl < 1e-8 and oc)
    recon = np.nan
    if ok:
        try:
            Tnz = T[:, T.sum(0) > 0]
            A_high, Tc = fit_abstraction(A, Tnz, n_restarts=3, seed=seed, maxiter=60)
            recon = reconstruction_loss(A, Tc, A_high)
        except Exception:
            pass
    return dict(policy='baseline+hm', graph_type=gt, p=p, true_nhigh=nh, seed=seed,
                ceiling=est, k=k, returned=1, success=ok, graph_ok=int(gl < 1e-8),
                one_child=oc, used=int((T.sum(0) > 0).sum()),
                info=(np.abs(A) * co).sum() / np.abs(A).sum(), recon=recon,
                max_clust=int(T.sum(0).max()), info_start=info_start, k_start=k)


if __name__ == '__main__':
    jobs = []
    for gt, d in [('random', 0.5), ('one_child', 1.0)]:
        for p in (50, 100, 200):
            for nh in sorted({max(3, p // 20), max(3, p // 10), max(3, p // 5)}):
                for s in range(5):
                    jobs.append((gt, d, p, nh, s))
    jobs.sort(key=lambda a: -a[2])
    print(f'{len(jobs)} runs', flush=True)
    t0 = time.time(); rows = []
    with mp.Pool(6) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.append(r)
            if i % 10 == 0:
                el = (time.time() - t0) / 60
                print(f'{i}/{len(jobs)}  {el:.1f}min', flush=True)
                pd.DataFrame(rows).to_csv(f'{SP}/warm.csv', index=False)
    pd.DataFrame(rows).to_csv(f'{SP}/warm.csv', index=False)
    print(f'done {(time.time()-t0)/60:.1f} min')
