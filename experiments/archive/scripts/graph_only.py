"""SANITY CHECK: does graph_loss ALONE satisfy All-In, and non-trivially?

Every previous test ran graph_loss alongside one_child and a utility term. This runs it as
the SOLE objective (lambda_graph=10, everything else 0) from both random and closure inits.

"Non-trivially" matters: k=1 satisfies All-In vacuously (no high-level edges), so a collapse
to one cluster is not evidence the loss works. We therefore report k_used and largest/p
alongside the All-In rate.
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
from src.utils.generate_data import sample_low_level_graph, sample_by_noise_adjustment_any_order
from src.utils.helper import process_output
from occ import occ, conditions


def job(a):
    torch.set_num_threads(1)
    p, dens, seed = a
    try:
        A, *_ = sample_low_level_graph(n_high=10, n_low=p, density=dens, shuffle=True,
                                       blockiness=0.5, heterogeneity=1., internal_density=0.,
                                       graph_type='random', seed=seed, connected=True,
                                       max_attempts=300)
    except Exception:
        return []
    absA = np.abs(A); tot = absA.sum(); E_low = np.linalg.inv(np.eye(p) - A)
    X = sample_by_noise_adjustment_any_order(A, n_samples=2000, gennorm_param=1.8, seed=seed+991)
    su = ((np.eye(p) - A.T) @ X.T).var(axis=1)

    def err(T):
        try:
            S, Tp, Ah = process_output(T_hard=T, A_low=A, E_low=E_low, n_restarts=3)
            Eh = np.linalg.inv(np.eye(Ah.shape[0]) - Ah)
            Dm = S @ Eh - E_low @ Tp; R = E_low @ Tp
            co2 = T @ T.T; np.fill_diagonal(co2, 0)
            marg = (np.abs(Tp).sum(1) == 0)
            lost = co2.astype(bool); lost[marg, :] = True; lost[:, marg] = True
            np.fill_diagonal(lost, False)
            return (float(np.sum(su[:,None]*Dm**2)/max(np.sum(su[:,None]*R**2),1e-30)),
                    float((absA*lost).sum()/tot), int(marg.sum()))
        except Exception:
            return np.nan, np.nan, np.nan
    g0, T0, ci = occ(A, seed=seed)
    K = ci['k']
    out = [dict(arm='OCC', init='-', allin=1, one_child=1, k_used=len(g0),
                largest=max(len(x) for x in g0),
                info=float((absA*((T0@T0.T)-np.diag(np.diag(T0@T0.T)))).sum()/tot),
                **dict(zip(('rel_err','info_eff','marginalised'), err(T0))))]
    for init in ('closure', 'random'):
        gen = torch.Generator().manual_seed(31337 + seed)
        li = (torch.from_numpy(T0).float()*5.0 + torch.randn(p, K, generator=gen)*0.1
              if init == 'closure' else torch.randn(p, K, generator=gen))
        m = HailMary(n_low=p, n_high=K, A_low=torch.from_numpy(A).float(), logits_init=li)
        m.train(num_epochs=6000, lambda_graph=10., lambda_one_child=0.,
                lambda_info=0., lambda_recon=0., lambda_surj=0., verbose=False)
        T = m.get_partition()[0].detach().numpy()
        gs = [x for x in (np.where(T[:, j] == 1)[0] for j in range(K)) if len(x)]
        ai, oc, dag = conditions(A, gs)
        co = T @ T.T; np.fill_diagonal(co, 0)
        out.append(dict(arm='graph_only', init=init, allin=int(ai), one_child=int(oc),
                        k_used=len(gs), largest=max(len(x) for x in gs),
                        info=float((absA*co).sum()/tot),
                        **dict(zip(('rel_err','info_eff','marginalised'), err(T)))))
    for r in out:
        r.update(p=p, density=dens, seed=seed, K=K)
    return out


if __name__ == '__main__':
    jobs = [(p, d, s) for p in (50,) for d in (0.1, 0.5, 0.9) for s in range(5)]
    print(f'{len(jobs)} instances', flush=True)
    with mp.Pool(6) as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    d = pd.DataFrame(rows)
    d.to_csv('/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/graph_only_err.csv', index=False)
    d['largest_frac'] = d.largest / d.p
    d['collapsed'] = (d.k_used <= 2).astype(int)
    print('\n=== graph_loss as the SOLE objective ===')
    print(d.groupby(['arm', 'init']).agg(n=('allin', 'size'), allin=('allin', 'mean'),
        one_child=('one_child', 'mean'), k_used=('k_used', 'mean'),
        largest_frac=('largest_frac', 'mean'), info=('info', 'mean'),
        info_eff=('info_eff','mean'), rel_err_med=('rel_err','median'),
        rel_err_p90=('rel_err', lambda s: s.quantile(.9)),
        marg=('marginalised','mean')).round(4).to_string())
    print('\n=== All-In among NON-COLLAPSED runs only (k_used >= 3) ===')
    nc = d[(d.arm == 'graph_only') & (d.k_used >= 3)]
    if len(nc):
        print(nc.groupby('init').agg(n=('allin', 'size'), allin=('allin', 'mean'),
            k_used=('k_used', 'mean'), largest_frac=('largest_frac', 'mean')).round(3).to_string())
    else:
        print('  (none -- every run collapsed)')
    print('\n=== by density ===')
    print(d[d.arm == 'graph_only'].groupby(['density', 'init']).agg(
        n=('allin', 'size'), allin=('allin', 'mean'), k_used=('k_used', 'mean'),
        largest_frac=('largest_frac', 'mean')).round(3).to_string())
