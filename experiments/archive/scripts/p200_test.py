"""The final pipeline at p=200, varying density and ground-truth n_high.

    OCC-AR -> HailMary (init = OCC partition, k = 2K, one_child=100, info=10, graph=0)
    accept iff  rel_err <= alpha  AND  info_eff improves,  else keep OCC's.

Reports the acceptance rate -- the number that decides whether the refinement is worth
running at this scale -- alongside retention.
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
B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'


def job(a):
    torch.set_num_threads(1)
    p, dens, nh, seed = a
    t_all = time.time()
    try:
        A, *_ = sample_low_level_graph(n_high=nh, n_low=p, density=dens, shuffle=True,
                                       blockiness=0.5, heterogeneity=1., internal_density=0.,
                                       graph_type='random', seed=seed, connected=True,
                                       max_attempts=300)
    except Exception:
        return None
    absA = np.abs(A); tot = absA.sum(); E_low = np.linalg.inv(np.eye(p) - A)
    X = sample_by_noise_adjustment_any_order(A, n_samples=2000, gennorm_param=1.8, seed=seed+991)
    su = ((np.eye(p) - A.T) @ X.T).var(axis=1)

    def ev(T):
        co = T @ T.T; np.fill_diagonal(co, 0)
        try:
            S, Tp, Ah = process_output(T_hard=T, A_low=A, E_low=E_low, n_restarts=3)
            Eh = np.linalg.inv(np.eye(Ah.shape[0]) - Ah)
            Dm = S @ Eh - E_low @ Tp; R = E_low @ Tp
            marg = (np.abs(Tp).sum(1) == 0)
            lost = co.astype(bool); lost[marg, :] = True; lost[:, marg] = True
            np.fill_diagonal(lost, False)
            return (float(np.sum(su[:,None]*Dm**2)/max(np.sum(su[:,None]*R**2),1e-30)),
                    float((absA*lost).sum()/tot), int(marg.sum()))
        except Exception:
            return np.nan, float((absA*co).sum()/tot), -1

    t0 = time.time(); g0, T0, ci = occ(A, seed=seed); t_occ = time.time()-t0
    K = ci['k']; k = int(min(2*K, p))
    rel0, ie0, m0 = ev(T0)

    t0 = time.time()
    gen = torch.Generator().manual_seed(31337+seed)
    base = np.zeros((p, k)); base[:, :min(k, T0.shape[1])] = T0[:, :k]
    init = torch.from_numpy(base).float()*5.0 + torch.randn(p, k, generator=gen)*0.1
    m = HailMary(n_low=p, n_high=k, A_low=torch.from_numpy(A).float(), logits_init=init)
    m.train(num_epochs=6000, lambda_graph=0., lambda_one_child=100., lambda_info=10.,
            lambda_recon=0., lambda_surj=0., verbose=False)
    T1 = m.get_partition()[0].detach().numpy()
    t_hm = time.time()-t0
    g1 = [x for x in (np.where(T1[:, j]==1)[0] for j in range(k)) if len(x)]
    ai, oc, dag = conditions(A, g1)
    rel1, ie1, m1 = ev(T1)
    return dict(p=p, density=dens, true_nhigh=nh, seed=seed, K=K, k_offered=k,
                occ_k=len(g0), occ_largest=max(len(x) for x in g0), occ_info=ie0,
                hm_k=len(g1), hm_largest=max(len(x) for x in g1), hm_info=ie1,
                hm_allin=int(ai), hm_one_child=int(oc), hm_rel=rel1, hm_marg=m1,
                improves=int(ie1 < ie0), t_occ=t_occ, t_hm=t_hm, t_all=time.time()-t_all)


if __name__ == '__main__':
    jobs = [(200, d, nh, s) for d in (0.1, 0.3, 0.5, 0.7, 0.9)
            for nh in (10, 20, 40) for s in range(3)]
    print(f'{len(jobs)} instances at p=200', flush=True)
    t0 = time.time(); rows = []
    with mp.Pool(6) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            if r: rows.append(r)
            if i % 5 == 0:
                el=(time.time()-t0)/60
                print(f'{i}/{len(jobs)} {el:.1f}min eta {el/i*(len(jobs)-i):.0f}min', flush=True)
                pd.DataFrame(rows).to_csv(f'{B}/data/p200_test.csv', index=False)
    d = pd.DataFrame(rows); d.to_csv(f'{B}/data/p200_test.csv', index=False)
    print('\n%d instances, %.1f min\n' % (len(d), (time.time()-t0)/60))
    for alpha in (0.01, 0.05, 0.10):
        d[f'acc{alpha}'] = ((d.hm_rel <= alpha) & (d.improves == 1)).astype(int)
    print('=== ACCEPTANCE RATE at p=200 ===')
    print(d.groupby(['density','true_nhigh']).agg(n=('improves','size'),
        improves=('improves','mean'), rel_med=('hm_rel','median'),
        acc_01=('acc0.01','mean'), acc_05=('acc0.05','mean'), acc_10=('acc0.1','mean')
        ).round(3).to_string())
    print('\n=== OVERALL ===')
    for alpha in (0.01,0.05,0.10):
        acc=d[f'acc{alpha}']
        hyb=np.where(acc==1, d.hm_info, d.occ_info)
        print('  alpha=%.2f  accepted %.2f   hybrid info_eff %.3f  (OCC %.3f, %+.0f%%)'%(
            alpha, acc.mean(), np.nanmean(hyb), d.occ_info.mean(),
            100*(np.nanmean(hyb)-d.occ_info.mean())/d.occ_info.mean()))
    print('\n=== structure, accepted runs (alpha=0.05) ===')
    a = d[d['acc0.05']==1]
    if len(a):
        print(a.groupby('density').agg(n=('hm_k','size'), occ_k=('occ_k','mean'), hm_k=('hm_k','mean'),
            occ_largest=('occ_largest','mean'), hm_largest=('hm_largest','mean'),
            occ_info=('occ_info','mean'), hm_info=('hm_info','mean')).round(2).to_string())
    print('\ntiming: OCC %.2fs   HailMary %.0fs   total %.0fs per instance'%(
        d.t_occ.mean(), d.t_hm.mean(), d.t_all.mean()))
