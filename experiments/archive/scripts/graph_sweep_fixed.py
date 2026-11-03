"""Validate graph_only and the lambda_graph sweep with the FIXED graph_loss.

rerun_fixed.py (n=20, p=50) found graph_only reaching info_eff 0.041 vs OCC's 0.549 and a
-52% hybrid -- the best measured. This widens it: more lambda values, both p, all densities,
and BOTH noise weightings for rel_err (graph_only is inexact, and that is exactly where the
weighting swings by up to 30x).
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

ARMS = ([('graph_only_l%g' % l, dict(lambda_graph=l, lambda_one_child=0., lambda_info=0.))
         for l in (1., 10., 100.)] +
        [('g%g_oc_info' % l, dict(lambda_graph=l, lambda_one_child=100., lambda_info=10.))
         for l in (0., 0.3, 1., 3., 10.)])


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

    def ev(T, name, secs):
        gs = [x for x in (np.where(T[:, j] == 1)[0] for j in range(T.shape[1])) if len(x)]
        ai, oc, dag = conditions(A, gs)
        co = T @ T.T; np.fill_diagonal(co, 0)
        row = dict(arm=name, allin=int(ai), one_child=int(oc), valid=int(ai and oc and dag),
                   k_used=len(gs), largest=max(len(x) for x in gs),
                   info=float((absA*co).sum()/tot), secs=secs)
        try:
            S, Tp, Ah = process_output(T_hard=T, A_low=A, E_low=E_low, n_restarts=3)
            Eh = np.linalg.inv(np.eye(Ah.shape[0]) - Ah)
            Dm = S @ Eh - E_low @ Tp; R = E_low @ Tp
            marg = (np.abs(Tp).sum(1) == 0)
            lost = co.astype(bool); lost[marg, :] = True; lost[:, marg] = True
            np.fill_diagonal(lost, False)
            row.update(rel_err=float(np.sum(su[:,None]*Dm**2)/max(np.sum(su[:,None]*R**2),1e-30)),
                       rel_err_iso=float(np.mean(Dm**2)/max(np.mean(R**2),1e-30)),
                       marginalised=int(marg.sum()),
                       info_eff=float((absA*lost).sum()/tot))
        except Exception:
            row.update(rel_err=np.nan, rel_err_iso=np.nan, marginalised=-1, info_eff=row['info'])
        return row

    g0, T0, ci = occ(A, seed=seed); K = ci['k']; k = int(min(2*K, p))
    out = [ev(T0, 'occ', 0.0)]
    for name, kw in ARMS:
        t0 = time.time()
        gen = torch.Generator().manual_seed(31337+seed)
        base = np.zeros((p, k)); base[:, :min(k, T0.shape[1])] = T0[:, :k]
        init = torch.from_numpy(base).float()*5.0 + torch.randn(p, k, generator=gen)*0.1
        m = HailMary(n_low=p, n_high=k, A_low=torch.from_numpy(A).float(), logits_init=init)
        m.train(num_epochs=6000, lambda_recon=0., lambda_surj=0., verbose=False, **kw)
        out.append(ev(m.get_partition()[0].detach().numpy(), name, time.time()-t0))
    for r in out:
        r.update(p=p, density=dens, seed=seed, K=K, k_offered=k)
    return out


if __name__ == '__main__':
    jobs = [(p, d, s) for p in (50, 100) for d in (0.1, 0.3, 0.5, 0.7, 0.9) for s in range(5)]
    print(f'{len(jobs)} instances x 9 arms', flush=True)
    t0 = time.time(); rows = []
    with mp.Pool(6) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs)
            if i % 10 == 0:
                el = (time.time()-t0)/60
                print(f'{i}/{len(jobs)} {el:.1f}min eta {el/i*(len(jobs)-i):.0f}min', flush=True)
                pd.DataFrame(rows).to_csv('/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/graph_sweep_fixed.csv', index=False)
    d = pd.DataFrame(rows)
    d.to_csv('/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/graph_sweep_fixed.csv', index=False)
    print('done %.1f min\n' % ((time.time()-t0)/60))
    order = ['occ'] + [n for n, _ in ARMS]
    print('=== all arms ===')
    print(d.groupby('arm').agg(n=('allin','size'), allin=('allin','mean'),
        one_child=('one_child','mean'), valid=('valid','mean'), info=('info','mean'),
        info_eff=('info_eff','mean'), marg=('marginalised','mean'),
        rel_med=('rel_err','median'), rel_iso_med=('rel_err_iso','median'),
        k_used=('k_used','mean')).reindex(order).round(4).to_string())
    occ_i = d[d.arm=='occ'].set_index(['p','density','seed']).info_eff
    print('\n=== alpha-hybrid at 0.05, under BOTH weightings ===')
    print('  arm                weighted            isotropic')
    print('                  acc   info_eff      acc   info_eff')
    for arm in order[1:]:
        g = d[d.arm==arm].set_index(['p','density','seed'])
        base = occ_i.reindex(g.index)
        line = '  %-16s' % arm
        for col in ('rel_err', 'rel_err_iso'):
            ok = (g[col] <= 0.05) & (g.info_eff < base)
            hyb = np.where(ok, g.info_eff, base)
            line += ' %.2f   %.3f    ' % (ok.mean(), np.nanmean(hyb))
        print(line + '   (OCC %.3f)' % occ_i.mean())
    d['exact'] = (d.rel_err < 1e-12)
    print('\n=== does a VALID abstraction EXIST? (rel_err == 0 exactly) ===')
    print('  prior: P(valid | conditions fail) = 0.267 over ARBITRARY failing partitions')
    print(d.groupby('arm').agg(n=('exact','size'), allin=('allin','mean'),
        one_child=('one_child','mean'), both=('valid','mean'),
        exact=('exact','mean')).reindex(order).round(3).to_string())
    print('\n  among arms with one_child == 0 (no guarantee), exactness rate:')
    nog = d[d.one_child == 0]
    if len(nog):
        print(nog.groupby('arm').agg(n=('exact','size'), allin=('allin','mean'),
            exact=('exact','mean')).round(3).to_string())

    print('\n=== graph_only_l10 by density ===')
    g = d[d.arm=='graph_only_l10']
    print(g.groupby(['p','density']).agg(n=('allin','size'), allin=('allin','mean'),
        info_eff=('info_eff','mean'), marg=('marginalised','mean'),
        rel_med=('rel_err','median')).round(3).to_string())
