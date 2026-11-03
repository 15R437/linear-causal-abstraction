"""k = K? Re-test on the CURRENT code with the CURRENT lambdas.

init_k_sweep.csv concluded k = K (OCC's own cluster count) is optimal, but it predates:
  * the edge_weight fix in graph_loss (2026-09-15)
  * process_output switching to fit_abstraction_alternative, now at n_restarts=5,
    maxiter=500, seed=0
and it used the as_is / recon arms, not lambda_recon=1 / graph=10 / one_child=100 / info=0
with init_type='occ_ac', which is what hail_mary.py's __main__ actually runs.

Same grid as init_k_sweep (p=60, density 0.5 and 0.9) so the numbers are comparable.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary, check_conditions
from src.utils.generate_data import sample_low_level_graph
from src.utils.helper import process_output, cluster_indices
from src.utils.losses import observation_loss, intervention_loss
from occ import occ

B = '/Users/issy/Projects/linear-causal-abstraction'
KMULTS = [0.5, 1.0, 1.5, 2.0]
LAM = dict(lambda_recon=1., lambda_graph=10., lambda_one_child=100., lambda_info=0.)
P, EPOCHS = 60, 3000


def score(A, T, E_low, absA, tot, **extra):
    idx = [x for x in cluster_indices(T) if len(x)]
    ai, oc, dag = check_conditions(A, cluster_indices(T))
    co = T @ T.T; np.fill_diagonal(co, 0)
    row = dict(allin=int(ai), one_child=int(oc), valid=int(ai and oc and dag),
               k_used=len(idx), largest=int(max(len(x) for x in idx)),
               intra=float((absA * co).sum() / tot), **extra)
    try:
        S, Tp, Ah = process_output(T_hard=T, A_low=A, E_low=E_low)   # current defaults
        Eh = np.linalg.inv(np.eye(Ah.shape[0]) - Ah)
        obs = float(observation_loss(E_low=E_low, E_high=Eh, S=S, T=Tp))
        intv = float(intervention_loss(E_low=E_low, E_high=Eh, S=S, T=Tp, T_hard=T))
        rowmax = np.abs(Tp).max(1)
        out = {}
        for tag, tol in (('', 0.0), ('_t2', 1e-2)):
            marg = rowmax <= tol
            lost = co.astype(bool).copy(); lost[marg, :] = True; lost[:, marg] = True
            np.fill_diagonal(lost, False)
            out['marg' + tag] = int(marg.sum())
            out['info_eff' + tag] = float((absA * lost).sum() / tot)
        row.update(obs=obs, intv=intv, exact=int(obs < 1e-8 and intv < 1e-8), **out)
    except Exception as e:
        row.update(obs=np.nan, intv=np.nan, exact=0, info_eff=np.nan,
                   info_eff_t2=np.nan, marg=np.nan, marg_t2=np.nan)
    return row


def job(a):
    torch.set_num_threads(1)
    dens, seed = a
    A, *_ = sample_low_level_graph(n_high=10, n_low=P, density=dens, shuffle=True,
        blockiness=0.5, heterogeneity=1.0, internal_density=0.0, graph_type='random', seed=seed)
    absA = np.abs(A); tot = absA.sum(); E_low = np.linalg.inv(np.eye(P) - A)
    g0, T0, ci = occ(A, seed=seed); K = ci['k']
    out = [score(A, T0, E_low, absA, tot, kmult=np.nan, k=K, arm='occ', secs=0.)]
    for km in KMULTS:
        k = int(np.clip(round(km * K), 2, P))
        t0 = time.time()
        torch.manual_seed(seed)
        m = HailMary(n_low=P, n_high=k, A_low=torch.from_numpy(A).float(), init_type='occ_ac')
        m.train(num_epochs=EPOCHS, lambda_surj=0, verbose=False, **LAM)
        T = m.get_partition()[0].cpu().detach().numpy()
        out.append(score(A, T, E_low, absA, tot, kmult=km, k=k, arm='hm', secs=time.time()-t0))
    for r in out:
        r.update(density=dens, seed=seed, K=K, p=P)
    return out


if __name__ == '__main__':
    jobs = [(d, s) for d in (0.5, 0.9) for s in range(6)]
    print(f'{len(jobs)} instances x {len(KMULTS)} kmults, p={P}, {EPOCHS} epochs', flush=True)
    t0 = time.time(); rows = []
    with mp.Pool(6) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs)
            print(f'{i}/{len(jobs)} {(time.time()-t0)/60:.1f}min', flush=True)
    d = pd.DataFrame(rows); d.to_csv(f'{B}/experiments/abstraction_study/data/ksweep_now.csv', index=False)
    pd.set_option('display.width', 240)
    occ_r = d[d.arm == 'occ']; hm = d[d.arm == 'hm']
    print('\ndone %.1f min' % ((time.time()-t0)/60))
    print('\nOCC reference: valid %.2f  exact %.2f  info_eff %.3f  info_eff_t2 %.3f  K %.1f (n=%d)'
          % (occ_r.valid.mean(), occ_r.exact.mean(), occ_r.info_eff.mean(),
             occ_r.info_eff_t2.mean(), occ_r.K.mean(), len(occ_r)))
    print('\nlambda_recon=1, graph=10, one_child=100, info=0, occ_ac init, CURRENT fitter:')
    print(hm.groupby('kmult').agg(n=('valid','size'), k=('k','mean'), k_used=('k_used','mean'),
        allin=('allin','mean'), one_child=('one_child','mean'), valid=('valid','mean'),
        exact=('exact','mean'), intra=('intra','mean'), info_eff=('info_eff','mean'),
        info_eff_t2=('info_eff_t2','mean'), marg_t2=('marg_t2','mean'),
        secs=('secs','mean')).round(3).to_string())
    print('\nHYBRID (use the run if exact, else OCC), per retention metric:')
    for ie in ('info_eff', 'info_eff_t2'):
        base = occ_r.set_index(['density','seed'])[ie]
        print(f'  {ie}:')
        for km, s in hm.groupby('kmult'):
            b = base.reindex(pd.MultiIndex.from_frame(s[['density','seed']])).values
            hyb = np.where(s.exact.values == 1, s[ie].values, b)
            print('    kmult %.1f : exact %.2f  hybrid %.3f   (OCC %.3f)'
                  % (km, s.exact.mean(), np.nanmean(hyb), np.nanmean(b)))
