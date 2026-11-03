"""Does extra k help when graph_loss is OFF?

slack.py and init_k_sweep.py both concluded k = K (OCC's own count) is optimal -- but both
ran with lambda_graph on. With All-In unenforced there is no pressure to consolidate, so
extra columns may now pay. Both utility arms, lambda_one_child = 100, lambda_graph = 0.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, argparse, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary
from src.utils.generate_data import sample_low_level_graph, sample_by_noise_adjustment_any_order
from src.utils.helper import process_output
from occ import occ, conditions
B = '/Users/issy/Projects/linear-causal-abstraction'
SLACK = [1.0, 1.5, 2.0, 3.0]
ARMS = ['info', 'recon']


def score(A, T, E_low, su, absA, tot, **extra):
    gs = [x for x in (np.where(T[:, j] == 1)[0] for j in range(T.shape[1])) if len(x)]
    ai, oc, dag = conditions(A, gs)
    co = T @ T.T; np.fill_diagonal(co, 0)
    row = dict(allin=int(ai), one_child=int(oc), valid=int(ai and oc and dag), k_used=len(gs),
               info=float((absA * co).sum() / tot), largest=int(max(len(x) for x in gs)), **extra)
    try:
        S, Tp, Ah = process_output(T_hard=T, A_low=A, E_low=E_low, n_restarts=3)
        Eh = np.linalg.inv(np.eye(Ah.shape[0]) - Ah)
        D = S @ Eh - E_low @ Tp; R = E_low @ Tp
        marg = (np.abs(Tp).sum(1) == 0)
        lost = co.astype(bool); lost[marg, :] = True; lost[:, marg] = True
        np.fill_diagonal(lost, False)
        row.update(rel_err=float(np.sum(su[:, None]*D**2)/max(np.sum(su[:, None]*R**2), 1e-30)),
                   marginalised=int(marg.sum()),
                   info_eff=float((absA * lost).sum() / tot))
    except Exception:
        row.update(rel_err=np.nan, marginalised=np.nan, info_eff=np.nan)
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
    X = sample_by_noise_adjustment_any_order(A, n_samples=2000, gennorm_param=1.8, seed=seed+991)
    su = ((np.eye(p) - A.T) @ X.T).var(axis=1)
    g, T0, ci = occ(A, seed=seed)
    K = ci['k']
    out = [score(A, T0, E_low, su, absA, tot, arm='OCC', slack=np.nan, K=K, k_offered=K)]
    for arm in ARMS:
        li_, lr_ = (10., 0.) if arm == 'info' else (0., 10.)
        for sl in SLACK:
            k = int(np.clip(round(sl * K), 2, p))
            gen = torch.Generator().manual_seed(31337 + seed)
            base = np.zeros((p, k)); base[:, :min(k, T0.shape[1])] = T0[:, :k]
            init = torch.from_numpy(base).float() * 5.0 + torch.randn(p, k, generator=gen) * 0.1
            m = HailMary(n_low=p, n_high=k, A_low=torch.from_numpy(A).float(), logits_init=init)
            m.train(num_epochs=6000, lambda_graph=0., lambda_one_child=100.,
                    lambda_info=li_, lambda_recon=lr_, lambda_surj=0., verbose=False)
            T = m.get_partition()[0].detach().numpy()
            out.append(score(A, T, E_low, su, absA, tot, arm=arm, slack=sl, K=K, k_offered=k))
    for r in out:
        r.update(p=p, density=dens, seed=seed)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--robust', action='store_true')
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    if a.robust:
        jobs = [(p, d, s) for p in (50, 100) for d in (0.1, 0.3, 0.5, 0.7, 0.9) for s in range(5)]
        out = 'slack_nograph_robust.csv'
    else:
        jobs = [(50, d, s) for d in (0.1, 0.5, 0.9) for s in range(4)]
        out = 'slack_nograph.csv'
    print(f'{len(jobs)} instances x 9 arms', flush=True)
    t0 = time.time(); rows = []
    with mp.Pool(a.workers) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs)
            if i % 3 == 0:
                pd.DataFrame(rows).to_csv(f'{B}/experiments/abstraction_study/data/{out}', index=False)
                print(f'{i}/{len(jobs)} {(time.time()-t0)/60:.1f}min', flush=True)
    d = pd.DataFrame(rows); d.to_csv(f'{B}/experiments/abstraction_study/data/{out}', index=False)
    print('done %.1f min\n' % ((time.time()-t0)/60))
    o = d[d.arm == 'OCC']
    print('OCC: valid %.2f  info %.3f  info_eff %.3f  k %.1f' % (
        o.valid.mean(), o['info'].mean(), o.info_eff.mean(), o.k_used.mean()))
    print('\nDOES EXTRA k HELP WITH graph_loss OFF?')
    print(d[d.arm != 'OCC'].groupby(['arm', 'slack']).agg(
        n=('info_eff', 'size'), k_offered=('k_offered', 'mean'), k_used=('k_used', 'mean'),
        allin=('allin', 'mean'), one_child=('one_child', 'mean'),
        rel_err_med=('rel_err', 'median'), rel_err_p90=('rel_err', lambda s: s.quantile(.9)),
        info=('info', 'mean'), info_eff=('info_eff', 'mean'),
        largest=('largest', 'mean')).round(4).to_string())
