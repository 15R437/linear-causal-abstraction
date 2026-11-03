"""Quick rerun of the key HailMary experiments with the FIXED graph_loss.

The fix (already in src/): graph_loss no longer multiplies the pairwise CE by
edge_weight = |A_low| @ T_soft. That term made the loss scale with edge MAGNITUDE (All-In is
structural, so it should be invariant) and destroyed its correlation with the violation count
in the soft regime where training happens (+0.002 weighted vs +0.460 unweighted).

Every "lambda_graph does nothing" conclusion was measured with the broken version. This asks
whether the fix changes them.

Arms (all lambda_one_child=100 unless stated):
  occ            reference, no training
  graph_only     lambda_graph=10, everything else 0   -- does graph_loss ALONE work now?
  oc_info        lambda_info=10, lambda_graph=0       -- the current recommendation
  oc_info_g1     lambda_info=10, lambda_graph=1
  oc_info_g10    lambda_info=10, lambda_graph=10      -- does adding graph help now?
  oc_info_g100   lambda_info=10, lambda_graph=100
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

ARMS = [('graph_only',   dict(lambda_graph=10.,  lambda_one_child=0.,   lambda_info=0.)),
        ('oc_info',      dict(lambda_graph=0.,   lambda_one_child=100., lambda_info=10.)),
        ('oc_info_g1',   dict(lambda_graph=1.,   lambda_one_child=100., lambda_info=10.)),
        ('oc_info_g10',  dict(lambda_graph=10.,  lambda_one_child=100., lambda_info=10.)),
        ('oc_info_g100', dict(lambda_graph=100., lambda_one_child=100., lambda_info=10.))]


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
                       info_eff=float((absA*lost).sum()/tot))
        except Exception:
            row.update(rel_err=np.nan, info_eff=row['info'])
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
    jobs = [(50, d, s) for d in (0.1, 0.3, 0.5, 0.7, 0.9) for s in range(4)]
    print(f'{len(jobs)} instances x 6 arms (p=50)', flush=True)
    t0 = time.time()
    with mp.Pool(6) as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    d = pd.DataFrame(rows)
    d.to_csv('/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/rerun_fixed.csv', index=False)
    print('done %.1f min\n' % ((time.time()-t0)/60))
    order = ['occ','graph_only','oc_info','oc_info_g1','oc_info_g10','oc_info_g100']
    print('=== WITH THE FIXED graph_loss ===')
    print(d.groupby('arm').agg(n=('allin','size'), allin=('allin','mean'),
        one_child=('one_child','mean'), valid=('valid','mean'),
        info=('info','mean'), info_eff=('info_eff','mean'),
        rel_err_med=('rel_err','median'), k_used=('k_used','mean'),
        secs=('secs','mean')).reindex(order).round(4).to_string())
    occ_i = d[d.arm=='occ'].set_index(['p','density','seed']).info_eff
    print('\n=== alpha-hybrid at 0.05 (accept if rel_err<=0.05 AND better, else OCC) ===')
    for arm in order[1:]:
        g = d[d.arm==arm].set_index(['p','density','seed'])
        base = occ_i.reindex(g.index)
        ok = (g.rel_err <= 0.05) & (g.info_eff < base)
        hyb = np.where(ok, g.info_eff, base)
        print('  %-13s accepted %.2f   hybrid info_eff %.3f   (OCC %.3f, %+.0f%%)'
              % (arm, ok.mean(), np.nanmean(hyb), occ_i.mean(),
                 100*(np.nanmean(hyb)-occ_i.mean())/occ_i.mean()))
