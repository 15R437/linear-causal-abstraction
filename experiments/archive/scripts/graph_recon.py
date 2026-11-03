"""graph + recon, and graph + recon + one_child -- the cells lossfit.py left out.

lossfit.py carried recon only at lambda_graph in {0, 1} and always WITH one_child, so three
combinations were missing. They matter because of the mechanism lossfit exposed:

  graph_loss is multiplied by co_assign, so it charges only CO-ASSIGNED pairs with mismatched
  child profiles -- a separating pressure, ~0 at singletons AND ~0 at the blob. one_child_loss
  is the only merging term (69.5 at singletons against graph's 0.0013).

So graph+recon without one_child has no merging pressure at all, and the question is whether
recon supplies one: recon_loss is also co_assign-weighted, but its H profile is built from
A_low_binary.T (PARENT sets, not child sets), so it penalises co-assigning nodes with
different parents -- again separating. Prediction: no blob, high k. Both arms test it, and
`recon_only` isolates recon from graph entirely.

Scoring is imported from lossfit so the two CSVs are directly comparable, and the instance
grid is a subset of lossfit's (seeds 0-3) so every arm here pairs with oc_info / oc_recon /
oc_recon_g1 on the same (p, density, seed).
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary
from src.utils.generate_data import sample_low_level_graph, sample_by_noise_adjustment_any_order
from occ import occ
from lossfit import score           # identical scoring, both fitters, tolerance ladder

B = '/Users/issy/Projects/linear-causal-abstraction'

ARMS = [('recon_only',    dict(lambda_graph=0.,  lambda_one_child=0.,   lambda_recon=10., lambda_info=0.)),
        ('g10_recon',     dict(lambda_graph=10., lambda_one_child=0.,   lambda_recon=10., lambda_info=0.)),
        ('g10_recon_oc',  dict(lambda_graph=10., lambda_one_child=100., lambda_recon=10., lambda_info=0.)),
        ('g1_recon_oc',   dict(lambda_graph=1.,  lambda_one_child=100., lambda_recon=10., lambda_info=0.)),
        ('g10_rec_inf_oc',dict(lambda_graph=10., lambda_one_child=100., lambda_recon=10., lambda_info=10.))]


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

    g0, T0, ci = occ(A, seed=seed)
    K = ci['k']; k = int(min(2 * K, p))          # slack 2, as lossfit.py
    out = score(A, T0, E_low, su, absA, tot, arm='occ', secs=0.0)
    for name, kw in ARMS:
        t0 = time.time()
        gen = torch.Generator().manual_seed(31337 + seed)     # same init stream as lossfit
        base = np.zeros((p, k)); base[:, :min(k, T0.shape[1])] = T0[:, :k]
        init = torch.from_numpy(base).float() * 5.0 + torch.randn(p, k, generator=gen) * 0.1
        m = HailMary(n_low=p, n_high=k, A_low=torch.from_numpy(A).float(), logits_init=init)
        m.train(num_epochs=6000, lambda_surj=0., verbose=False, **kw)
        out += score(A, m.get_partition()[0].detach().numpy(), E_low, su, absA, tot,
                     arm=name, secs=time.time() - t0)
    for r in out:
        r.update(p=p, density=dens, seed=seed, K=K, k_offered=k)
    return out


if __name__ == '__main__':
    jobs = [(p, d, s) for p in (50, 100) for d in (0.1, 0.3, 0.5, 0.7, 0.9) for s in range(4)]
    path = f'{B}/experiments/abstraction_study/data/graph_recon.csv'
    print(f'{len(jobs)} instances x {len(ARMS)+1} arms x 2 fitters', flush=True)
    t0 = time.time(); rows = []
    with mp.Pool(6) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs)
            pd.DataFrame(rows).to_csv(path, index=False)
            if i % 3 == 0:
                print(f'{i}/{len(jobs)} {(time.time()-t0)/60:.1f}min', flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print('done %.1f min -> %s' % ((time.time()-t0)/60, path))
