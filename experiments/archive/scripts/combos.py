"""Coarsest pass: which combinations of {graph, recon, info} Pareto-dominate?

lambda_one_child = 100 throughout (fixed, load-bearing). Each of the other three is simply
ON (10) or OFF (0) -- 8 combinations, including one_child alone as the floor and the current
recommendation (graph=0, recon=0, info=10) as the incumbent.

Frontier axes:  acceptance rate  x  hybrid info_eff (OCC fallback, so all arms comparable).
Reported alongside but not on the frontier: All-In, one-child, exactness, runtime.

p=50, 3 densities, 4 seeds -- deliberately small. Iterate from whatever survives.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, itertools, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary
from src.utils.generate_data import sample_low_level_graph, sample_by_noise_adjustment_any_order
from src.utils.helper import process_output
from occ import occ, conditions
OUT = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/combos.csv'

COMBOS = [(g, r, i) for g, r, i in itertools.product((0., 10.), (0., 10.), (0., 10.))]


def name(g, r, i):
    on = [n for n, v in (('g', g), ('r', r), ('i', i)) if v]
    return 'oc+' + ('+'.join(on) if on else 'only')


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

    def ev(T, nm, secs):
        gs = [x for x in (np.where(T[:, j] == 1)[0] for j in range(T.shape[1])) if len(x)]
        ai, oc_, dag = conditions(A, gs)
        co = T @ T.T; np.fill_diagonal(co, 0)
        row = dict(arm=nm, allin=int(ai), one_child=int(oc_), valid=int(ai and oc_ and dag),
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
    out = [ev(T0, 'OCC', 0.0)]
    for lg, lr, li in COMBOS:
        t0 = time.time()
        gen = torch.Generator().manual_seed(31337+seed)
        base = np.zeros((p, k)); base[:, :min(k, T0.shape[1])] = T0[:, :k]
        init = torch.from_numpy(base).float()*5.0 + torch.randn(p, k, generator=gen)*0.1
        m = HailMary(n_low=p, n_high=k, A_low=torch.from_numpy(A).float(), logits_init=init)
        m.train(num_epochs=6000, lambda_graph=lg, lambda_recon=lr, lambda_info=li,
                lambda_one_child=100., lambda_surj=0., verbose=False)
        out.append(ev(m.get_partition()[0].detach().numpy(), name(lg, lr, li), time.time()-t0))
    for r in out:
        r.update(p=p, density=dens, seed=seed, K=K, k_offered=k)
    return out


if __name__ == '__main__':
    jobs = [(50, d, s) for d in (0.1, 0.5, 0.9) for s in range(4)]
    print(f'{len(jobs)} instances x {len(COMBOS)+1} arms', flush=True)
    t0 = time.time()
    with mp.Pool(6) as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    d = pd.DataFrame(rows); d.to_csv(OUT, index=False)
    d['exact'] = d.rel_err < 1e-12
    print('done %.1f min\n' % ((time.time()-t0)/60))
    order = ['OCC'] + [name(*c) for c in COMBOS]
    print('=== all combinations (lambda_one_child=100 throughout) ===')
    print(d.groupby('arm').agg(n=('allin','size'), allin=('allin','mean'),
        one_child=('one_child','mean'), valid=('valid','mean'), exact=('exact','mean'),
        info=('info','mean'), info_eff=('info_eff','mean'), marg=('marginalised','mean'),
        k_used=('k_used','mean'), secs=('secs','mean')).reindex(order).round(3).to_string())

    occ_i = d[d.arm == 'OCC'].set_index(['p','density','seed']).info_eff
    print('\n=== FRONTIER: acceptance x hybrid info_eff (alpha=0.05) ===')
    pts = []
    for arm in order[1:]:
        g = d[d.arm == arm].set_index(['p','density','seed'])
        base = occ_i.reindex(g.index)
        ok = (g.rel_err <= 0.05) & (g.info_eff < base)
        hyb = float(np.nanmean(np.where(ok, g.info_eff, base)))
        pts.append((arm, ok.mean(), hyb, g.secs.mean()))
    pts.append(('OCC', 1.0, occ_i.mean(), 0.0))
    for arm, acc, hyb, sec in sorted(pts, key=lambda t: t[2]):
        dominated = any(a2 >= acc and h2 <= hyb and (a2 > acc or h2 < hyb)
                        for n2, a2, h2, _ in pts if n2 != arm)
        print('  %-10s acceptance %.2f   hybrid info_eff %.3f   %5.1fs   %s'
              % (arm, acc, hyb, sec, 'dominated' if dominated else '** PARETO **'))
