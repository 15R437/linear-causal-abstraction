"""Three candidate information-loss measures, compared on the same valid partitions.

  info_raw   intra-cluster share of  |A_ij|                (current)
  info_var   intra-cluster share of  w_ij^2 Var(Xi)/Var(Xj) (standardised path coefficient)
  anc_lost   1 - (fraction of low-level ancestral pairs i~>j whose clusters still
             satisfy cluster(i) ~> cluster(j))             (weight-free, nothing optimises it)

If info_raw is a sound definition, it should track anc_lost. If it does not, the weighting
is doing the work and should be justified or replaced.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, importlib.util, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, multiprocessing as mp
from src.utils.generate_data import sample_low_level_graph, sample_by_noise_adjustment_any_order
import metric as M
SP = '/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad'


def reach(Ab):
    """Transitive closure by repeated boolean matmul (p is small enough)."""
    R = Ab.copy()
    for _ in range(int(np.ceil(np.log2(max(Ab.shape[0], 2)))) + 1):
        R = R | (R @ R)
    return R


def job(a):
    p, nh, seed = a
    A, _, _, _ = sample_low_level_graph(
        n_high=nh, n_low=p, density=0.5, shuffle=True, blockiness=0.5, heterogeneity=1.,
        internal_density=0., graph_type='random', seed=seed, connected=True, max_attempts=400)
    Ab = (A != 0)
    X = sample_by_noise_adjustment_any_order(A, n_samples=4000, gennorm_param=1.8, seed=seed)
    v = X.var(0); v[v < 1e-12] = 1e-12
    # standardised path coefficient: share of Var(Xj) explained by parent i
    W = (A ** 2) * v[:, None] / v[None, :]
    R = reach(Ab)
    np.fill_diagonal(R, False)
    rng = np.random.default_rng(seed)
    out = []
    for g in M.candidate_partitions(A, rng):
        T = np.zeros((p, len(g)), bool)
        for j, m in enumerate(g):
            T[m, j] = True
        same = T @ T.T
        np.fill_diagonal(same, False)
        Mh = np.zeros((len(g), len(g)), bool)
        for x in range(len(g)):
            for y in range(len(g)):
                if x != y and Ab[np.ix_(g[x], g[y])].any():
                    Mh[x, y] = True
        Rh = reach(Mh); np.fill_diagonal(Rh, False)
        cl = np.zeros(p, int)
        for j, m in enumerate(g):
            cl[m] = j
        cross = R & ~same                       # ancestral pairs spanning clusters
        kept = cross & Rh[np.ix_(cl, cl)]       # whose clusters still reach
        out.append(dict(p=p, true_nhigh=nh, seed=seed, k=len(g),
                        info_raw=(np.abs(A) * same).sum() / np.abs(A).sum(),
                        info_var=(W * same).sum() / W.sum(),
                        anc_lost=1 - kept.sum() / max(R.sum(), 1),
                        anc_intra=(R & same).sum() / max(R.sum(), 1)))
    return out


if __name__ == '__main__':
    jobs = [(p, nh, s) for p in (50, 100) for nh in (10, 20) for s in range(8) if nh <= p // 5]
    with mp.Pool(6) as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    d = pd.DataFrame(rows); d.to_csv(f'{SP}/infometric.csv', index=False)
    from scipy import stats
    print('%d partitions over %d instances\n' % (len(d), d.groupby(['p', 'true_nhigh', 'seed']).ngroups))
    print('WITHIN-INSTANCE Spearman (positive = the measures agree):')
    res = []
    for kk, g in d.groupby(['p', 'true_nhigh', 'seed']):
        if g.info_raw.nunique() < 4:
            continue
        res.append(dict(rv=stats.spearmanr(g.info_raw, g.info_var).correlation,
                        ra=stats.spearmanr(g.info_raw, g.anc_lost).correlation,
                        va=stats.spearmanr(g.info_var, g.anc_lost).correlation))
    r = pd.DataFrame(res)
    print('  usable instances: %d' % len(r))
    for c, lab in [('rv', 'info_raw vs info_var'), ('ra', 'info_raw vs anc_lost'),
                   ('va', 'info_var vs anc_lost')]:
        x = r[c].dropna()
        print('  %-22s mean %+.3f  median %+.3f  positive %2d/%2d'
              % (lab, x.mean(), x.median(), (x > 0).sum(), len(x)))
    print()
    print('levels (mean over all partitions):')
    print(d[['info_raw', 'info_var', 'anc_lost', 'anc_intra']].mean().round(3).to_string())
