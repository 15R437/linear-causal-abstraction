"""Head-to-head on how to CHOOSE k: per-instance ceiling estimate vs fixed fractions of p.

Policies:  ceiling (merge-closure + greedy split estimator, ~ms)
           0.25p, 0.4p   (fixed fractions)
Families:  random (d=0.5), one_child (d=1.0)
Varies:    p in {50,100,200}  x  true n_high in {p/20, p/10, p/5}  x  5 seeds
Model:     current src HailMary, closure init, recon off, info on, 6000 epochs fixed
           (the epoch rule degenerates near k=p, so it is pinned).
Scores:    success = All-In AND one-child (guarantees an abstraction exists)
           info    = intra-cluster edge mass fraction
           recon   = reconstruction_loss after fit_abstraction, successful runs only
"""
import sys, time, importlib.util, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary, closure_logits
from src.utils.generate_data import sample_low_level_graph
from src.utils.losses import graph_loss, reconstruction_loss
from src.utils.helper import fit_abstraction

_s = importlib.util.spec_from_file_location(
    'mk', '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts/max_k_estimate.py')
mk = importlib.util.module_from_spec(_s); _s.loader.exec_module(mk)

SP = '/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad'
OUT = f'{SP}/kpolicy.csv'
EPOCHS = 6000


def job(a):
    torch.set_num_threads(1)
    gt, d, p, nh, seed, policy = a
    A, _, _, _ = sample_low_level_graph(
        n_high=nh, n_low=p, density=d, shuffle=True, blockiness=0.5, heterogeneity=1.,
        internal_density=0., graph_type=gt, seed=seed, connected=True, max_attempts=400)

    t0 = time.time()
    ceil_est, _ = mk.estimate_max_k(A, n_orders=12, seed=seed)
    t_ceil = time.time() - t0

    k = {'ceiling': ceil_est, '0.25p': int(round(0.25 * p)), '0.4p': int(round(0.4 * p))}[policy]
    k = int(min(max(k, 2), p))

    gen = torch.Generator().manual_seed(31337 + seed)
    li = closure_logits(A, k, rng=np.random.default_rng(31337 + seed), generator=gen)
    m = HailMary(n_low=p, n_high=k, A_low=torch.from_numpy(A).float(), logits_init=li)
    t1 = time.time()
    m.train(num_epochs=EPOCHS, lambda_recon=0., lambda_graph=10., lambda_one_child=100.,
            lambda_surj=0., lambda_info=10., verbose=False)
    t_fit = time.time() - t1

    T = m.get_partition()[0].detach().numpy()
    M = (T.T @ (A != 0).astype(float) @ T) > 0
    np.fill_diagonal(M, False)
    co = T @ T.T; np.fill_diagonal(co, 0)
    gl = graph_loss(A_low=A, T_hard=T)
    oc = int(M.sum(1).max() <= 1)
    ok = int(gl < 1e-8 and oc)
    used = int((T.sum(0) > 0).sum())

    recon = np.nan
    if ok:
        try:
            Tnz = T[:, T.sum(0) > 0]
            A_high, Tc = fit_abstraction(A, Tnz, n_restarts=3, seed=seed, maxiter=60)
            recon = reconstruction_loss(A, Tc, A_high)
        except Exception as e:
            recon = np.nan
            err = type(e).__name__

    return dict(policy=policy, graph_type=gt, p=p, true_nhigh=nh, seed=seed,
                ceiling=ceil_est, k=k, k_over_ceiling=k / max(ceil_est, 1),
                graph_ok=int(gl < 1e-8), one_child=oc, success=ok,
                used=used, used_over_ceiling=used / max(ceil_est, 1),
                info=(np.abs(A) * co).sum() / np.abs(A).sum(),
                max_clust=int(T.sum(0).max()), high_edges=int(M.sum()),
                recon=recon, t_ceil=t_ceil, t_fit=t_fit)


if __name__ == '__main__':
    jobs = []
    for gt, d in [('random', 0.5), ('one_child', 1.0)]:
        for p in (50, 100, 200):
            for nh in sorted({max(3, p // 20), max(3, p // 10), max(3, p // 5)}):
                for s in range(5):
                    for pol in ('ceiling', '0.25p', '0.4p'):
                        jobs.append((gt, d, p, nh, s, pol))
    # big jobs first so the tail is short
    jobs.sort(key=lambda a: -a[2])
    print(f'{len(jobs)} runs', flush=True)
    t0 = time.time()
    rows = []
    with mp.Pool(6) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.append(r)
            if i % 15 == 0:
                el = (time.time() - t0) / 60
                print(f'{i}/{len(jobs)}  {el:.1f}min  eta {el/i*(len(jobs)-i):.0f}min', flush=True)
                pd.DataFrame(rows).to_csv(OUT, index=False)
    df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
    print(f'\ndone {(time.time()-t0)/60:.1f} min -> {OUT}')
