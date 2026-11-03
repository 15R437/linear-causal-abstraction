"""Quick sweep: does HailMary do better with a different init, or a different k?

The marginalisation comparison pinned HailMary to closure's init at k = K. This varies both.
Scored on info_effective (intra-cluster mass + mass incident to marginalised nodes) so a
partition cannot win by having fit_abstraction silently drop half the graph.

p=60 for speed.  Losses: as_is (intra_mass) and recon.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, importlib.util, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary, regularised_surjective_init
from src.utils.generate_data import sample_low_level_graph
from src.utils.losses import observation_loss, intervention_loss
from src.utils.helper import process_output
from occ import occ, conditions
B = '/Users/issy/Projects/linear-causal-abstraction'

LOSSES = {'as_is': dict(lambda_graph=10., lambda_one_child=100., lambda_info=10., lambda_recon=0.),
          'recon': dict(lambda_graph=10., lambda_one_child=100., lambda_info=0., lambda_recon=10.)}


def make_init(kind, T0, p, k, seed):
    gen = torch.Generator().manual_seed(31337 + seed)
    if kind == 'closure':
        base = np.zeros((p, k)); base[:, :T0.shape[1]] = T0[:, :k]
        return torch.from_numpy(base).float() * 5.0 + torch.randn(p, k, generator=gen) * 0.1
    if kind == 'reg_surj':
        return regularised_surjective_init(n_low=p, n_high=k, noise_std=0.1, seed=31337 + seed)
    return torch.randn(p, k, generator=gen)          # 'random'


def evaluate(A, T, E_low, absA, tot):
    p = A.shape[0]
    groups = [g for g in (np.where(T[:, j] == 1)[0] for j in range(T.shape[1])) if len(g)]
    ai, oc, dag = conditions(A, groups)
    co = T @ T.T; np.fill_diagonal(co, 0)
    out = dict(allin=int(ai), one_child=int(oc), valid=int(ai and oc and dag), k_used=len(groups))
    try:
        S, T_pred, A_high = process_output(T_hard=T, A_low=A, E_low=E_low, n_restarts=3)
        E_high = np.linalg.inv(np.eye(A_high.shape[0]) - A_high)
        obs = float(observation_loss(E_low=E_low, E_high=E_high, S=S, T=T_pred))
        intv = float(intervention_loss(E_low=E_low, E_high=E_high, S=S, T=T_pred))
        marg = (np.abs(T_pred).sum(1) == 0)
        lost = co.astype(bool); lost[marg, :] = True; lost[:, marg] = True
        np.fill_diagonal(lost, False)
        out.update(exact=int(obs < 1e-8 and intv < 1e-8), marginalised=int(marg.sum()),
                   info_eff=float((absA * lost).sum() / tot),
                   info_intra=float((absA * co).sum() / tot))
    except Exception:
        out.update(exact=0, marginalised=np.nan, info_eff=np.nan, info_intra=np.nan)
    return out


def job(a):
    torch.set_num_threads(1)
    dens, seed = a
    p = 60
    try:
        A, *_ = sample_low_level_graph(n_high=10, n_low=p, density=dens, shuffle=True,
                                       blockiness=0.5, heterogeneity=1., internal_density=0.,
                                       graph_type='random', seed=seed, connected=True,
                                       max_attempts=300)
    except Exception:
        return []
    absA = np.abs(A); tot = absA.sum(); E_low = np.linalg.inv(np.eye(p) - A)
    groups, T0, ci = occ(A, seed=seed)
    K = ci['k']
    rows = [dict(loss='-', init='closure(no training)', kmult=1.0, k=K,
                 **evaluate(A, T0, E_low, absA, tot))]
    for kmult in (0.5, 1.0, 1.5, 2.0):
        k = int(np.clip(round(kmult * K), 2, p))
        for init in ('closure', 'reg_surj', 'random'):
            for lname, kw in LOSSES.items():
                li = make_init(init, T0, p, k, seed)
                m = HailMary(n_low=p, n_high=k, A_low=torch.from_numpy(A).float(), logits_init=li)
                m.train(num_epochs=4000, lambda_surj=0., verbose=False, **kw)
                T = m.get_partition()[0].detach().numpy()
                rows.append(dict(loss=lname, init=init, kmult=kmult, k=k,
                                 **evaluate(A, T, E_low, absA, tot)))
    for r in rows:
        r.update(density=dens, seed=seed, K=K)
    return rows


if __name__ == '__main__':
    jobs = [(d, s) for d in (0.5, 0.9) for s in range(5)]
    print(f'{len(jobs)} instances x 25 configs', flush=True)
    t0 = time.time()
    with mp.Pool(6) as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    d = pd.DataFrame(rows)
    d.to_csv(f'{B}/experiments/abstraction_study/data/init_k_sweep.csv', index=False)
    print('done %.1f min, %d rows\n' % ((time.time()-t0)/60, len(d)))
    base = d[d.loss == '-']
    print('closure baseline: valid %.2f  exact %.2f  info_eff %.3f  k %.1f'
          % (base.valid.mean(), base.exact.mean(), base.info_eff.mean(), base.k.mean()))
    print()
    h = d[d.loss != '-']
    print('=== HailMary: valid / exact / info_eff by init x k ===')
    print(h.groupby(['loss', 'init', 'kmult']).agg(
        n=('valid', 'size'), valid=('valid', 'mean'), exact=('exact', 'mean'),
        info_eff=('info_eff', 'mean'), marg=('marginalised', 'mean'),
        k_used=('k_used', 'mean')).round(3).to_string())
    print('\n=== best configs by info_eff (with closure fallback when not exact) ===')
    b = base.set_index(['density', 'seed'])['info_eff']
    res = []
    for (l, i, km), g in h.groupby(['loss', 'init', 'kmult']):
        gg = g.set_index(['density', 'seed'])
        hyb = np.where(gg.exact == 1, gg.info_eff, b.reindex(gg.index))
        res.append((l, i, km, gg.exact.mean(), float(np.nanmean(hyb))))
    r = pd.DataFrame(res, columns=['loss', 'init', 'kmult', 'exact', 'hybrid_info_eff'])
    print(r.sort_values('hybrid_info_eff').head(12).round(3).to_string(index=False))
    print('\n(closure alone hybrid baseline = %.3f)' % base.info_eff.mean())
