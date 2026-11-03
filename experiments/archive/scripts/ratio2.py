"""lambda_graph : lambda_recon RATIO -- the axis every previous sweep missed.

WHY. The loss terms are not on comparable scales. Measured at near-hard T_soft (p=50, d=0.5):

    partition      k    graph      recon     info    recon/graph
    singletons    50    0.00134    0.0014    0.0006        1
    occ           11    0.00077   32.0364    6.0951    41534
    one blob       2    0.00013    4.1291    2.0000    31227
    random 5-way   5    3.84565    5.7044    0.9793        1

graph_loss is ~4e4 times SMALLER than recon_loss near a good partition (it only becomes large,
~3.8, at badly mismatched partitions). So `graph_recon.py`'s lambda_graph=10 / lambda_recon=10
arm was recon-dominated by ~4e4 near convergence -- and indeed `g10_recon` and `recon_only`
produced near-identical partitions (k_used 45.6 vs 44.3 at d=0.1; 14.4 vs 15.1 at d=0.9) and
All-In 0.077 vs 0.000. It was never a "graph + recon" arm.

To balance the two near a good partition you need lambda_graph/lambda_recon ~ 1e4. This sweeps
that ratio, including the 1000:1 setting reported anecdotally to satisfy All-In often.

It also tests whether the report's lambda_graph range (0, 1, 10, 100 against lambda_info=10) is
simply too low: info_loss is ~8e3 times larger than graph_loss at the occ partition too.

COLUMN DEFINITIONS for the analysis output are printed by the script.
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
from lossfit import score

B = '/Users/issy/Projects/linear-causal-abstraction'

#        name            graph   one_child  recon  info     domination (utility/graph at occ)
ARMS = [('g100_i10',     dict(lambda_graph=100.,   lambda_one_child=0.,   lambda_recon=0., lambda_info=10.)),   # 792x  <- reported underperforming
        ('g500_i10',     dict(lambda_graph=500.,   lambda_one_child=0.,   lambda_recon=0., lambda_info=10.)),   # 158x
        ('g2000_i10',    dict(lambda_graph=2000.,  lambda_one_child=0.,   lambda_recon=0., lambda_info=10.)),   # 40x   <- matched to graph+recon 1000:1
        ('g4000_i10',    dict(lambda_graph=4000.,  lambda_one_child=0.,   lambda_recon=0., lambda_info=10.)),   # 20x
        ('g100_i10_oc',  dict(lambda_graph=100.,   lambda_one_child=100., lambda_recon=0., lambda_info=10.)),   # same, one-child ON
        ('g1000_r1_ref', dict(lambda_graph=1000.,  lambda_one_child=0.,   lambda_recon=1., lambda_info=0.)),   # 20x, the graph+recon reference
        ('graph_only',   dict(lambda_graph=10.,    lambda_one_child=0.,   lambda_recon=0., lambda_info=0.)),    # graph alone
        ('oc_info',      dict(lambda_graph=0.,     lambda_one_child=100., lambda_recon=0., lambda_info=10.))]   # current recommendation

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
    K = ci['k']; k = int(min(2 * K, p))
    out = score(A, T0, E_low, su, absA, tot, X=X, arm='occ', secs=0.0)
    for name, kw in ARMS:
        t0 = time.time()
        gen = torch.Generator().manual_seed(31337 + seed)   # same init stream as graph_recon
        base = np.zeros((p, k)); base[:, :min(k, T0.shape[1])] = T0[:, :k]
        init = torch.from_numpy(base).float() * 5.0 + torch.randn(p, k, generator=gen) * 0.1
        m = HailMary(n_low=p, n_high=k, A_low=torch.from_numpy(A).float(), logits_init=init)
        m.train(num_epochs=6000, lambda_surj=0., verbose=False, **kw)
        T_hard = m.get_partition()[0].detach().numpy()
        # record the UNWEIGHTED loss terms at the partition reached, so domination is visible
        with torch.no_grad():
            rec, gr, oc_, su_, en, inf = m.forward(1., True, True, True)
        out += score(A, T_hard, E_low, su, absA, tot, X=X, arm=name, secs=time.time() - t0,
                     term_graph=float(gr), term_recon=float(rec), term_oc=float(oc_),
                     term_info=float(inf))
    for r in out:
        r.update(p=p, density=dens, seed=seed, K=K, k_offered=k)
    return out


if __name__ == '__main__':
    jobs = [(p, d, s) for p in (50, 100) for d in (0.1, 0.3, 0.5, 0.7, 0.9) for s in range(4)]
    path = f'{B}/experiments/abstraction_study/data/ratio2.csv'
    print(f'{len(jobs)} instances x {len(ARMS)+1} arms (graph:info balance sweep) x 2 fitters', flush=True)
    t0 = time.time(); rows = []
    with mp.Pool(6) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs)
            pd.DataFrame(rows).to_csv(path, index=False)
            if i % 4 == 0:
                print(f'{i}/{len(jobs)} {(time.time()-t0)/60:.1f}min', flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print('done %.1f min -> %s' % ((time.time()-t0)/60, path))
