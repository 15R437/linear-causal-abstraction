"""CONTROL: is All-In satisfied by the OBJECTIVE, or inherited from the closure init?

approximate.py initialised HailMary from OCC's partition, which already satisfies All-In.
So its 0.94 All-In rate at lambda_graph=0 may be inherited rather than earned. This reruns
the key cells with a RANDOM init as well, so the two explanations separate.

Original docstring:
APPROXIMATE abstractions: keep one-child, relax All-In, minimise information loss.

The finding this tests: one-child violations are catastrophic (relative observation error
~0.84) because a multi-child cluster makes the Massidda equations inconsistent. All-In
violations degrade GRACEFULLY, because fit_abstraction marginalises the offenders -- the
recon arm's failures sit at ~2% relative error.

So the well-posed objective for a learned method is: find a ONE-CHILD partition that
minimises information loss, tolerating All-In violations. That is NOT what HailMary has
been running (it fights hard for All-In via lambda_graph=10).

Sweep lambda_graph down to 0 with lambda_one_child pinned at 100, and trace the frontier
of (information retained) vs (relative observation error).
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary
from src.utils.generate_data import sample_low_level_graph
from src.utils.helper import process_output
from occ import occ, conditions
B = '/Users/issy/Projects/linear-causal-abstraction'
LG = [10.0, 0.0]
INITS = ['closure', 'random']


def score(A, T, E_low, absA, tot, label, **extra):
    p = A.shape[0]
    gs = [x for x in (np.where(T[:, j] == 1)[0] for j in range(T.shape[1])) if len(x)]
    ai, oc, dag = conditions(A, gs)
    co = T @ T.T; np.fill_diagonal(co, 0)
    row = dict(arm=label, allin=int(ai), one_child=int(oc), valid=int(ai and oc and dag),
               k=len(gs), info=float((absA * co).sum() / tot),
               largest=int(max(len(x) for x in gs)), **extra)
    try:
        S, Tp, Ah = process_output(T_hard=T, A_low=A, E_low=E_low, n_restarts=3)
        Eh = np.linalg.inv(np.eye(Ah.shape[0]) - Ah)
        lhs, rhs = S @ Eh, E_low @ Tp
        rel = float(np.mean((lhs - rhs) ** 2) / max(np.mean(rhs ** 2), 1e-30))
        marg = (np.abs(Tp).sum(1) == 0)
        lost = co.astype(bool); lost[marg, :] = True; lost[:, marg] = True
        np.fill_diagonal(lost, False)
        row.update(rel_obs=rel, marginalised=int(marg.sum()),
                   info_eff=float((absA * lost).sum() / tot))
    except Exception:
        row.update(rel_obs=np.nan, marginalised=np.nan, info_eff=np.nan)
    return row


def job(a):
    torch.set_num_threads(1)
    dens, seed = a
    p = 100
    try:
        A, *_ = sample_low_level_graph(n_high=10, n_low=p, density=dens, shuffle=True,
                                       blockiness=0.5, heterogeneity=1., internal_density=0.,
                                       graph_type='random', seed=seed, connected=True,
                                       max_attempts=300)
    except Exception:
        return []
    absA = np.abs(A); tot = absA.sum(); E_low = np.linalg.inv(np.eye(p) - A)
    g, T0, ci = occ(A, seed=seed)
    K = ci['k']
    out = [score(A, T0, E_low, absA, tot, 'OCC', lam_graph=np.nan, loss='-', init='-')]
    for loss, kw in [('info', dict(lambda_info=10., lambda_recon=0.)),
                     ('recon', dict(lambda_info=0., lambda_recon=10.))]:
        for lg in LG:
            for init in INITS:
                gen = torch.Generator().manual_seed(31337 + seed)
                if init == 'closure':
                    li = torch.from_numpy(T0).float() * 5.0 + torch.randn(p, K, generator=gen) * 0.1
                else:
                    li = torch.randn(p, K, generator=gen)
                m = HailMary(n_low=p, n_high=K, A_low=torch.from_numpy(A).float(), logits_init=li)
                m.train(num_epochs=6000, lambda_graph=lg, lambda_one_child=100.,
                        lambda_surj=0., verbose=False, **kw)
                T = m.get_partition()[0].detach().numpy()
                out.append(score(A, T, E_low, absA, tot, f'{loss}_lg{lg}_{init}',
                                 lam_graph=lg, loss=loss, init=init))
    for r in out:
        r.update(density=dens, seed=seed, K=K)
    return out


if __name__ == '__main__':
    jobs = [(d, s) for d in (0.5, 0.9) for s in range(8)]
    print(f'{len(jobs)} instances x 11 arms', flush=True)
    t0 = time.time()
    with mp.Pool(6) as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    d = pd.DataFrame(rows)
    d.to_csv(f'{B}/experiments/abstraction_study/data/approximate_init_control.csv', index=False)
    print('done %.1f min\n' % ((time.time() - t0) / 60))
    print('=== IS ALL-IN EARNED OR INHERITED?  closure init vs random init ===')
    print(d.groupby(['loss', 'lam_graph', 'init'], dropna=False).agg(
        n=('valid', 'size'), allin=('allin', 'mean'), one_child=('one_child', 'mean'),
        valid=('valid', 'mean'),
        rel_obs_med=('rel_obs', 'median'), rel_obs_p90=('rel_obs', lambda s: s.quantile(.9)),
        info=('info', 'mean'), info_eff=('info_eff', 'mean'),
        k=('k', 'mean'), largest=('largest', 'mean')).round(4).to_string())
    occ_row = d[d.arm == 'OCC']
    print('\nOCC reference: valid %.3f  rel_obs %.4f  info %.4f  k %.1f  largest %.1f'
          % (occ_row.valid.mean(), occ_row.rel_obs.median(), occ_row['info'].mean(),
             occ_row.k.mean(), occ_row.largest.mean()))
    print('\n=== the verdict ===')
    h = d[d.arm != 'OCC']
    for (l, lg, i), g in h.groupby(['loss', 'lam_graph', 'init']):
        print('  %-6s lg=%-4s %-8s  All-In %.3f  valid %.3f  info %.3f  rel_err(p90) %.4f'
              % (l, lg, i, g.allin.mean(), g.valid.mean(), g['info'].mean(),
                 g.rel_obs.quantile(.9)))
