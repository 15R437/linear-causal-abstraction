"""Is fit_abstraction_alternative's error floor real, or just optimiser budget?

lossfit.csv scores both fitters at n_restarts=3 / maxiter=100, the settings every prior
study used through process_output. fit_abstraction solves its equations analytically and
lands at machine precision; the alternative runs SLSQP on a residual, so it stops wherever
its budget runs out. If its nonzero rel_err is budget rather than a real modelling gap, the
exactness comparison at 1e-8 is measuring the optimiser, not the method -- and the alpha
rule (which tolerates 0.05) is the only fair lens.

Holds the partitions fixed (regenerated from the same seeds as lossfit.py) and re-fits at
increasing budget.
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
from src.utils.helper import fit_abstraction, fit_abstraction_alternative, get_S
from src.utils.losses import observation_loss, intervention_loss
from occ import occ, conditions

B = '/Users/issy/Projects/linear-causal-abstraction'
BUDGETS = [(3, 100), (10, 100), (10, 500), (30, 500)]
ARMS = [('oc_info',  dict(lambda_graph=0., lambda_one_child=100., lambda_info=10., lambda_recon=0.)),
        ('oc_recon', dict(lambda_graph=0., lambda_one_child=100., lambda_info=0.,  lambda_recon=10.)),
        ('graph_only', dict(lambda_graph=10., lambda_one_child=0., lambda_info=0., lambda_recon=0.))]


def err(A, T_hard, T, A_high, E_low, su):
    k = A_high.shape[0]
    E_high = np.linalg.inv(np.eye(k) - A_high)
    S = get_S(T=T, E_low=E_low, T_hard=T_hard)
    D = S @ E_high - E_low @ T; R = E_low @ T
    rowmax = np.abs(T).max(1)
    return dict(rel_err=float(np.sum(su[:, None]*D**2)/max(np.sum(su[:, None]*R**2), 1e-30)),
                obs=float(observation_loss(E_low=E_low, E_high=E_high, S=S, T=T)),
                intv=float(intervention_loss(E_low=E_low, E_high=E_high, S=S, T=T, T_hard=T_hard)),
                marg_t2=int((rowmax <= 1e-2).sum()), marg_0=int((rowmax == 0).sum()))


def job(a):
    torch.set_num_threads(1)
    p, dens, seed = a
    A, *_ = sample_low_level_graph(n_high=10, n_low=p, density=dens, shuffle=True,
                                   blockiness=0.5, heterogeneity=1., internal_density=0.,
                                   graph_type='random', seed=seed, connected=True, max_attempts=300)
    E_low = np.linalg.inv(np.eye(p) - A)
    X = sample_by_noise_adjustment_any_order(A, n_samples=2000, gennorm_param=1.8, seed=seed+991)
    su = ((np.eye(p) - A.T) @ X.T).var(axis=1)
    g0, T0, ci = occ(A, seed=seed); K = ci['k']; k = int(min(2*K, p))

    parts = [('occ', T0)]
    for name, kw in ARMS:
        gen = torch.Generator().manual_seed(31337 + seed)
        base = np.zeros((p, k)); base[:, :min(k, T0.shape[1])] = T0[:, :k]
        init = torch.from_numpy(base).float()*5.0 + torch.randn(p, k, generator=gen)*0.1
        m = HailMary(n_low=p, n_high=k, A_low=torch.from_numpy(A).float(), logits_init=init)
        m.train(num_epochs=6000, lambda_surj=0., verbose=False, **kw)
        parts.append((name, m.get_partition()[0].detach().numpy()))

    rows = []
    for arm, T_hard in parts:
        meta = dict(p=p, density=dens, seed=seed, arm=arm, k_used=T_hard.shape[1],
                    valid=int(all(conditions(A, [x for x in (np.where(T_hard[:, j] == 1)[0]
                                for j in range(T_hard.shape[1])) if len(x)]))))
        t0 = time.perf_counter()
        Ar, Tr = fit_abstraction(A_low=A, T_hard=T_hard, E_low=E_low, n_restarts=3)[:2]
        rows.append(dict(meta, fitter='reg', n_restarts=3, maxiter=100,
                         secs=time.perf_counter()-t0, **err(A, T_hard, Tr, Ar, E_low, su)))
        for nr, mi in BUDGETS:
            t0 = time.perf_counter()
            Aa, Ta = fit_abstraction_alternative(A_low=A, E_low=E_low, T_hard=T_hard,
                                                 n_restarts=nr, maxiter=mi)[:2]
            rows.append(dict(meta, fitter='alt', n_restarts=nr, maxiter=mi,
                             secs=time.perf_counter()-t0, **err(A, T_hard, Ta, Aa, E_low, su)))
    return rows


if __name__ == '__main__':
    jobs = [(p, d, s) for p in (50, 100) for d in (0.1, 0.5, 0.9) for s in range(3)]
    print(f'{len(jobs)} instances x 4 partitions x (1 reg + {len(BUDGETS)} alt budgets)', flush=True)
    t0 = time.time(); rows = []
    with mp.Pool(6) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs)
            print(f'{i}/{len(jobs)} {(time.time()-t0)/60:.1f}min', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(f'{B}/experiments/abstraction_study/data/altbudget.csv', index=False)
    pd.set_option('display.width', 220)
    print('\ndone %.1f min\n' % ((time.time()-t0)/60))
    d['exact'] = (d.obs < 1e-8) & (d.intv < 1e-8)
    print(d.groupby(['fitter', 'n_restarts', 'maxiter']).agg(
        n=('rel_err', 'size'), exact=('exact', 'mean'),
        rel_err_med=('rel_err', 'median'), rel_err_p90=('rel_err', lambda s: s.quantile(.9)),
        obs_med=('obs', 'median'), marg_t2=('marg_t2', 'mean'),
        secs=('secs', 'mean')).round(6).to_string())
    print('\nby arm (median rel_err):')
    print(d.pivot_table(index='arm', columns=['fitter', 'n_restarts', 'maxiter'],
                        values='rel_err', aggfunc='median').round(6).to_string())
