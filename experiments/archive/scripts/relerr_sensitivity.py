"""How sensitive is rel_err -- and the ACCEPT/REJECT decision -- to the noise variances?

D = S E_high - E_low T depends only on A_low and the partition. Sigma_u enters ONLY as a
weighting. So we can hold the partitions fixed and re-score under several weightings:

  generator : the actual per-node noise variance from the SNR-based sampler
  isotropic : Sigma_u = I
  inverse   : Sigma_u proportional to 1/var  (worst-case adversarial reweighting)
  uniform-r : a random positive reweighting, 20 draws
  highSNR   : data regenerated with a different noise scale

What matters is not whether the NUMBER moves but whether the DECISION (rel_err <= alpha)
flips.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary
from src.utils.generate_data import sample_low_level_graph, sample_by_noise_adjustment_any_order
from src.utils.helper import process_output
from occ import occ
ALPHA = 0.05


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
    E_low = np.linalg.inv(np.eye(p) - A)
    X = sample_by_noise_adjustment_any_order(A, n_samples=4000, gennorm_param=1.8, seed=seed+991)
    su = ((np.eye(p) - A.T) @ X.T).var(axis=1)
    X2 = sample_by_noise_adjustment_any_order(A, n_samples=4000, gennorm_param=1.2, seed=seed+555)
    su2 = ((np.eye(p) - A.T) @ X2.T).var(axis=1)

    g0, T0, ci = occ(A, seed=seed); K = ci['k']; k = int(min(2*K, p))
    gen = torch.Generator().manual_seed(31337+seed)
    base = np.zeros((p, k)); base[:, :min(k, T0.shape[1])] = T0[:, :k]
    init = torch.from_numpy(base).float()*5.0 + torch.randn(p, k, generator=gen)*0.1
    m = HailMary(n_low=p, n_high=k, A_low=torch.from_numpy(A).float(), logits_init=init)
    m.train(num_epochs=6000, lambda_graph=0., lambda_one_child=100., lambda_info=10.,
            lambda_recon=0., lambda_surj=0., verbose=False)
    T1 = m.get_partition()[0].detach().numpy()
    try:
        S, Tp, Ah = process_output(T_hard=T1, A_low=A, E_low=E_low, n_restarts=3)
        Eh = np.linalg.inv(np.eye(Ah.shape[0]) - Ah)
        Dm = S @ Eh - E_low @ Tp; R = E_low @ Tp
    except Exception:
        return []

    def rel(w):
        return float(np.sum(w[:, None]*Dm**2) / max(np.sum(w[:, None]*R**2), 1e-30))

    rng = np.random.default_rng(seed)
    rows = [dict(weighting='generator', rel=rel(su)),
            dict(weighting='isotropic', rel=rel(np.ones(p))),
            dict(weighting='inverse',   rel=rel(1.0/np.clip(su, 1e-12, None))),
            dict(weighting='highSNR',   rel=rel(su2))]
    rr = [rel(rng.random(p) * su.max()) for _ in range(20)]
    rows.append(dict(weighting='random20_med', rel=float(np.median(rr))))
    rows.append(dict(weighting='random20_min', rel=float(np.min(rr))))
    rows.append(dict(weighting='random20_max', rel=float(np.max(rr))))
    for r in rows:
        r.update(p=p, density=dens, seed=seed)
    return rows


if __name__ == '__main__':
    jobs = [(100, d, s) for d in (0.1, 0.3, 0.5, 0.7, 0.9) for s in range(4)]
    with mp.Pool(6) as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    d = pd.DataFrame(rows)
    d.to_csv('/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/relerr_sensitivity.csv', index=False)
    piv = d.pivot_table(index=['density','seed'], columns='weighting', values='rel')
    print('rel_err under different noise weightings (%d instances)\n' % piv.shape[0])
    print(piv.round(4).to_string())
    print('\n=== does the ACCEPT decision (rel_err <= %.2f) flip? ===' % ALPHA)
    acc = (piv <= ALPHA)
    ref = acc['generator']
    for c in piv.columns:
        if c == 'generator':
            continue
        print('  %-14s agrees with generator on %2d/%2d  (accept rate %.2f vs %.2f)'
              % (c, int((acc[c] == ref).sum()), len(ref), acc[c].mean(), ref.mean()))
    print('\n  ALL weightings agree on %d/%d instances' % (int(acc.nunique(axis=1).eq(1).sum()), len(acc)))
    print('\nspread within instance (max/min over weightings):')
    sp = (piv.max(axis=1) / piv.min(axis=1).clip(lower=1e-12))
    print('  median %.1fx   p90 %.1fx   max %.1fx' % (sp.median(), sp.quantile(.9), sp.max()))
