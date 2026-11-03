"""THE FRONTIER: one loss family, one dial (lambda_graph), traced against tolerance alpha.

    L = lambda_graph * All-In  +  100 * one-child  +  lambda_info * intra_mass
                                                   +  lambda_recon * recon

THE DIAL IS THE UTILITY-TO-ONE-CHILD RATIO, WITHIN EACH ARM.

lambda_one_child stays at 100 and lambda_util sweeps 1 -> 100, separately for each utility
loss. Pushing utility harder relative to one-child should trade validity for retention --
that is the trade-off worth mapping.

NOT lambda_graph. Measured: lambda_graph has NO effect on
All-In satisfaction (intra_mass 9/16 vs 10/16, recon 15/16 vs 14/16, Fisher p=1.000 both) --
All-In is a property of the utility loss. recon achieves ~90% All-In whether or not you push
on it; intra_mass ~60%. So the continuous knob is

    t in [0,1]:   lambda_recon = 10(1-t),  lambda_info = 10t

t=0 is pure recon (near-exact, lower retention), t=1 is pure intra_mass (approximate, high
retention), and t interpolates. One loss family, one hyperparameter.
OCC is the alpha = 0 endpoint (exactly valid by construction, no training).

Utility mixes tried at every setting of the dial, so the choice of utility term is a
secondary axis rather than a change of method:
    info  (10, 0)      recon (0, 10)      both  (5, 5)

ERROR METRIC -- corrected. The relative observation error is
    tr(D^T Sigma_u D) / tr((E_low T)^T Sigma_u (E_low T)),   D = S E_high - E_low T
i.e. the fraction of high-level VARIANCE that is abstraction error (~ 1 - R^2). Sigma_u is
the per-node noise covariance, which this generator makes far from isotropic; the unweighted
version used earlier overstates the error by up to 19x.
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
B = '/Users/issy/Projects/linear-causal-abstraction'

LAM_UTIL = [1.0, 3.0, 10.0, 30.0, 100.0]     # the dial, against lambda_one_child = 100
ARMS = ['info', 'recon']
LAM_GRAPH = [0.0]                             # fixed: measured to have no effect on All-In


def score(A, T, E_low, su, absA, tot, **extra):
    gs = [x for x in (np.where(T[:, j] == 1)[0] for j in range(T.shape[1])) if len(x)]
    ai, oc, dag = conditions(A, gs)
    co = T @ T.T; np.fill_diagonal(co, 0)
    row = dict(allin=int(ai), one_child=int(oc), valid=int(ai and oc and dag), k=len(gs),
               info=float((absA * co).sum() / tot),
               largest=int(max(len(x) for x in gs)), **extra)
    try:
        S, Tp, Ah = process_output(T_hard=T, A_low=A, E_low=E_low, n_restarts=3)
        Eh = np.linalg.inv(np.eye(Ah.shape[0]) - Ah)
        D = S @ Eh - E_low @ Tp
        R = E_low @ Tp
        rel_w = float(np.sum(su[:, None] * D ** 2) / max(np.sum(su[:, None] * R ** 2), 1e-30))
        rel_i = float(np.mean(D ** 2) / max(np.mean(R ** 2), 1e-30))
        marg = (np.abs(Tp).sum(1) == 0)
        lost = co.astype(bool); lost[marg, :] = True; lost[:, marg] = True
        np.fill_diagonal(lost, False)
        row.update(rel_err=rel_w, rel_err_iso=rel_i, marginalised=int(marg.sum()),
                   info_eff=float((absA * lost).sum() / tot))
    except Exception:
        row.update(rel_err=np.nan, rel_err_iso=np.nan, marginalised=np.nan, info_eff=np.nan)
    return row


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
    X = sample_by_noise_adjustment_any_order(A, n_samples=2000, gennorm_param=1.8, seed=seed + 991)
    su = ((np.eye(p) - A.T) @ X.T).var(axis=1)      # per-node noise variance
    g, T0, ci = occ(A, seed=seed)
    K = ci['k']
    out = [score(A, T0, E_low, su, absA, tot, method='OCC', lam_graph=np.nan, arm='-', lam_util=np.nan, ratio=np.nan)]
    for arm in ARMS:
        for lu in LAM_UTIL:
            li_, lr_ = (lu, 0.) if arm == 'info' else (0., lu)
            lg = LAM_GRAPH[0]
            gen = torch.Generator().manual_seed(31337 + seed)
            init = torch.from_numpy(T0).float() * 5.0 + torch.randn(p, K, generator=gen) * 0.1
            m = HailMary(n_low=p, n_high=K, A_low=torch.from_numpy(A).float(), logits_init=init)
            m.train(num_epochs=6000, lambda_graph=lg, lambda_one_child=100.,
                    lambda_info=li_, lambda_recon=lr_, lambda_surj=0., verbose=False)
            T = m.get_partition()[0].detach().numpy()
            out.append(score(A, T, E_low, su, absA, tot, method=f'{arm}_u{lu}',
                             lam_graph=lg, arm=arm, lam_util=lu, ratio=lu/100.))
    for r in out:
        r.update(p=p, density=dens, seed=seed, K=K)
    return out


if __name__ == '__main__':
    jobs = [(p, d, s) for p in (50, 100) for d in (0.1, 0.5, 0.9) for s in range(3)]
    print(f'{len(jobs)} instances x 11 arms (OCC + 2 losses x 5 ratios)', flush=True)
    t0 = time.time(); rows = []
    with mp.Pool(6) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs)
            if i % 3 == 0:
                el = (time.time()-t0)/60
                print(f'{i}/{len(jobs)}  {el:.1f}min  eta {el/i*(len(jobs)-i):.0f}min', flush=True)
                pd.DataFrame(rows).to_csv(f'{B}/experiments/abstraction_study/data/frontier.csv', index=False)
    d = pd.DataFrame(rows)
    d.to_csv(f'{B}/experiments/abstraction_study/data/frontier.csv', index=False)
    print('done %.1f min\n' % ((time.time()-t0)/60))
    occ_ref = d[d.method == 'OCC'].groupby(['p','density']).info_eff.mean()
    print('=== weighted vs isotropic error (the metric fix) ===')
    h = d[d.method != 'OCC']
    print('  median rel_err  weighted %.5f   isotropic %.5f'
          % (h.rel_err.median(), h.rel_err_iso.median()))
    print('  p90    rel_err  weighted %.5f   isotropic %.5f'
          % (h.rel_err.quantile(.9), h.rel_err_iso.quantile(.9)))
    print('\n=== THE FRONTIER: best info_eff achievable at each tolerance alpha ===')
    print('   (OCC = alpha 0 endpoint.  "share" = fraction of instances meeting alpha)')
    for alpha in (0.0, 0.01, 0.05, 0.10, 0.25):
        print(f'\n  alpha = {alpha}')
        best = []
        for (u, lg), g in h.groupby(['arm', 'lam_util']):
            ok = g[g.rel_err <= alpha] if alpha > 0 else g[g.valid == 1]
            if len(ok) < 0.5 * len(g):
                continue
            best.append((u, lg, len(ok)/len(g), ok.info_eff.mean(), ok.allin.mean(), ok.k.mean()))
        best.sort(key=lambda t: t[3])
        for u, lg, share, ie, ai, k in best[:4]:
            print('    %-6s lg=%-5s share %.2f  info_eff %.3f  All-In %.2f  k %.1f'
                  % (u, lg, share, ie, ai, k))
        print('    OCC                          share 1.00  info_eff %.3f' % d[d.method=='OCC'].info_eff.mean())
    print('\n=== THE DIAL: lambda_util / lambda_one_child, per arm ===')
    print(h.groupby(['arm','lam_util']).agg(n=('info_eff','size'), allin=('allin','mean'),
        one_child=('one_child','mean'), valid=('valid','mean'),
        rel_err_med=('rel_err','median'), rel_err_p90=('rel_err',lambda s:s.quantile(.9)),
        info=('info','mean'), info_eff=('info_eff','mean'), k=('k','mean')).round(4).to_string())
    print('\n=== by density, per t ===')
    sel = h
    print(sel.groupby(['density','arm','lam_util']).agg(n=('info_eff','size'), allin=('allin','mean'),
        rel_err_med=('rel_err','median'), rel_err_p90=('rel_err',lambda s:s.quantile(.9)),
        info_eff=('info_eff','mean')).round(4).to_string())
    print('\nOCC info_eff by cell:'); print(occ_ref.round(4).to_string())
