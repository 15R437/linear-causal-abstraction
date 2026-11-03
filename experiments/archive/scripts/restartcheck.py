"""Does n_restarts=5 reintroduce the spurious all-zero A_high optimum?

fitcmp.csv saw fit_abstraction_alternative return an all-zero A_high on 3/120 runs at
n_restarts=10 while the structural mask had edges. The restart loop guards against this by
keeping every candidate and tie-breaking toward the larger ||A_high||, so fewer restarts =
fewer candidates to tie-break among. This checks the edge-recovery failure modes directly.

Scope: OCC partitions only (no training needed, so this is cheap and every instance has a
known-good answer -- the structural mask). The failure mode was originally seen on trained
partitions, so this bounds rather than settles it.
"""
import os, sys, warnings
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[v] = '1'
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, multiprocessing as mp
from src.utils.generate_data import sample_low_level_graph
from src.utils.helper import fit_abstraction_alternative, high_level_mask, get_S
from src.utils.losses import observation_loss, intervention_loss
from occ import occ

SETTINGS = [(3, 500), (5, 500), (10, 500)]


def job(a):
    p, dens, seed = a
    try:
        A, *_ = sample_low_level_graph(n_high=10, n_low=p, density=dens, shuffle=True,
                                       blockiness=0.5, heterogeneity=1., internal_density=0.,
                                       graph_type='random', seed=seed, connected=True,
                                       max_attempts=300)
    except Exception:
        return []
    E_low = np.linalg.inv(np.eye(p) - A)
    g, T0, ci = occ(A, seed=seed)
    mask = int(high_level_mask(A_low=A, T_hard=T0).sum())
    rows = []
    for nr, mi in SETTINGS:
        Ah, T = fit_abstraction_alternative(A_low=A, E_low=E_low, T_hard=T0,
                                            n_restarts=nr, maxiter=mi, seed=0)[:2]
        k = Ah.shape[0]
        E_high = np.linalg.inv(np.eye(k) - Ah)
        S = get_S(T=T, E_low=E_low, T_hard=T0)
        obs = float(observation_loss(E_low=E_low, E_high=E_high, S=S, T=T))
        intv = float(intervention_loss(E_low=E_low, E_high=E_high, S=S, T=T, T_hard=T0))
        edges = int((np.abs(Ah) > 1e-9).sum())
        rows.append(dict(p=p, density=dens, seed=seed, n_restarts=nr, maxiter=mi,
                         mask_edges=mask, edges=edges, edges_vs_mask=edges - mask,
                         lost_all_edges=int(mask > 0 and edges == 0),
                         lost_some=int(edges < mask), over=int(edges > mask),
                         faith=float(np.linalg.norm(Ah)),
                         obs=obs, intv=intv, exact=int(obs < 1e-8 and intv < 1e-8)))
    return rows


if __name__ == '__main__':
    jobs = [(p, d, s) for p in (50, 100) for d in (0.1, 0.3, 0.5, 0.7, 0.9) for s in range(4)]
    with mp.Pool(6) as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    d = pd.DataFrame(rows)
    d.to_csv('/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/restartcheck.csv', index=False)
    pd.set_option('display.width', 220)
    print('OCC partitions, %d instances, maxiter=500, seed=0\n' % d[['p','density','seed']].drop_duplicates().shape[0])
    print('COLUMNS: lost_all = mask had edges but A_high has none (the spurious optimum).')
    print('  lost_some/over = A_high has fewer/more edges than the structural mask.')
    print('  faith = ||A_high||_F.  exact = obs<1e-8 and intv<1e-8.\n')
    print(d.groupby('n_restarts').agg(
        n=('edges', 'size'), mask_edges=('mask_edges', 'mean'), edges=('edges', 'mean'),
        lost_all=('lost_all_edges', 'sum'), lost_some=('lost_some', 'sum'),
        over=('over', 'sum'), faith=('faith', 'mean'), exact=('exact', 'mean')).round(3).to_string())
