"""Does oc+r (one_child + recon) behave differently from a RANDOM init?

recon_loss never carried the edge_weight term, so it is unaffected by that fix. With OCC
init it was unremarkable (info_eff 0.564 vs OCC's own 0.563). This checks random init.

Every result in graph_sweep_fixed.csv initialised HailMary from OCC-AR's partition
(margin 5.0, noise 0.1). With the edge_weight term removed, graph_loss may now be able to
find good partitions unaided -- which would mean the pipeline no longer depends on OCC.
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

ARMS = [('oc+r',    dict(lambda_graph=0., lambda_one_child=100., lambda_info=0.,  lambda_recon=10.)),
        ('oc+r+i',  dict(lambda_graph=0., lambda_one_child=100., lambda_info=10., lambda_recon=10.)),
        ('oc+i',    dict(lambda_graph=0., lambda_one_child=100., lambda_info=10., lambda_recon=0.))]


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

    def ev(T, nm, init):
        gs = [x for x in (np.where(T[:, j] == 1)[0] for j in range(T.shape[1])) if len(x)]
        ai, oc_, dag = conditions(A, gs)
        co = T @ T.T; np.fill_diagonal(co, 0)
        row = dict(arm=nm, init=init, allin=int(ai), one_child=int(oc_),
                   valid=int(ai and oc_ and dag), k_used=len(gs),
                   info=float((absA*co).sum()/tot))
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
    out = [ev(T0, 'OCC', '-')]
    for nm, kw in ARMS:
        for init in ('occ', 'random'):
            gen = torch.Generator().manual_seed(31337+seed)
            if init == 'occ':
                base = np.zeros((p, k)); base[:, :min(k, T0.shape[1])] = T0[:, :k]
                li = torch.from_numpy(base).float()*5.0 + torch.randn(p, k, generator=gen)*0.1
            else:
                li = torch.randn(p, k, generator=gen)
            m = HailMary(n_low=p, n_high=k, A_low=torch.from_numpy(A).float(), logits_init=li)
            m.train(num_epochs=6000, lambda_surj=0., verbose=False, **kw)
            out.append(ev(m.get_partition()[0].detach().numpy(), nm, init))
    for r in out:
        r.update(p=p, density=dens, seed=seed)
    return out


if __name__ == '__main__':
    jobs = [(50, d, s) for d in (0.1, 0.5, 0.9) for s in range(4)]
    print(f'{len(jobs)} instances x 7 arms (p=50)', flush=True)
    t0 = time.time()
    with mp.Pool(6) as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    d = pd.DataFrame(rows)
    d.to_csv('/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/init_check_recon.csv', index=False)
    d['exact'] = d.rel_err < 1e-12
    print('done %.1f min\n' % ((time.time()-t0)/60))
    print('=== does the OCC-AR init still matter with the fixed graph_loss? ===')
    print(d[d.arm != 'OCC'].groupby(['arm','init']).agg(
        n=('allin','size'), allin=('allin','mean'), one_child=('one_child','mean'),
        valid=('valid','mean'), exact=('exact','mean'),
        info=('info','mean'), info_eff=('info_eff','mean'),
        k_used=('k_used','mean')).round(3).to_string())
    o = d[d.arm == 'OCC']
    print('\nOCC-AR reference: exact 1.000  info_eff %.3f  k %.1f' % (o.info_eff.mean(), o.k_used.mean()))
