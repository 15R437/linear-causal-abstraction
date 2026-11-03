"""Why do some random DAGs admit NO All-In-satisfying one-child closure, under any order?

For each instance from estscale, recompute structural features and compare the 44 that
returned nothing against the 216 that returned something. Leading hypothesis from the
earlier closure work: All-In breaks when a cluster with an outgoing edge contains a node
that has no edge into the child cluster -- classically a SINK absorbed into a non-terminal
cluster.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, importlib.util, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, multiprocessing as mp
from src.utils.generate_data import sample_low_level_graph
_s = importlib.util.spec_from_file_location(
    'mk', '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts/max_k_estimate.py')
mk = importlib.util.module_from_spec(_s); _s.loader.exec_module(mk)
SP = '/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad'


def job(a):
    p, nh, seed = a
    A, _, _, _ = sample_low_level_graph(
        n_high=nh, n_low=p, density=0.5, shuffle=True, blockiness=0.5, heterogeneity=1.,
        internal_density=0., graph_type='random', seed=seed, connected=True, max_attempts=400)
    Ab = (A != 0)
    outdeg, indeg = Ab.sum(1), Ab.sum(0)
    rng = np.random.default_rng(seed)

    # diagnose ONE closure draw: where exactly does All-In break?
    g = mk.closure(A, rng)
    K = len(g)
    M = np.zeros((K, K), bool)
    for x in range(K):
        for y in range(K):
            if x != y and Ab[np.ix_(g[x], g[y])].any():
                M[x, y] = True
    viol_nodes = 0          # nodes violating All-In
    viol_sinks = 0          # of those, ones that are global sinks
    parent_clusters = 0     # clusters with a child
    for x in range(K):
        kids = np.where(M[x])[0]
        if len(kids) == 0:
            continue
        parent_clusters += 1
        for y in kids:
            for i in g[x]:
                if not Ab[i, g[y]].any():
                    viol_nodes += 1
                    if outdeg[i] == 0:
                        viol_sinks += 1
    return dict(p=p, true_nhigh=nh, seed=seed,
                n_sinks=int((outdeg == 0).sum()), sink_frac=float((outdeg == 0).mean()),
                n_roots=int((indeg == 0).sum()),
                max_outdeg=int(outdeg.max()), mean_outdeg=float(outdeg.mean()),
                density=float(Ab.mean()), K_closure=K,
                largest_cluster=max(len(x) for x in g),
                parent_clusters=parent_clusters,
                viol_nodes=viol_nodes, viol_sinks=viol_sinks,
                allin_ok=int(mk.conditions(A, g)[0]))


if __name__ == '__main__':
    d = pd.read_csv(f'{SP}/estscale.csv')
    jobs = list(d[['p', 'true_nhigh', 'seed']].itertuples(index=False, name=None))
    with mp.Pool(6) as pool:
        rows = pool.map(job, jobs)
    f = pd.DataFrame(rows).merge(d[['p', 'true_nhigh', 'seed', 'returned', 'draws_ok']],
                                 on=['p', 'true_nhigh', 'seed'])
    f.to_csv(f'{SP}/bimodal.csv', index=False)
    print('%d instances\n' % len(f))
    print('=== feature means: instances that returned NOTHING vs SOMETHING ===')
    cols = ['n_sinks', 'sink_frac', 'n_roots', 'max_outdeg', 'mean_outdeg', 'density',
            'K_closure', 'largest_cluster', 'parent_clusters', 'viol_nodes', 'viol_sinks']
    print(f.groupby('returned')[cols].mean().round(3).T.to_string())
    print()
    print('=== correlation of each feature with draws_ok (0-12) ===')
    print(f[cols].corrwith(f.draws_ok, method='spearman').round(3).sort_values().to_string())
    print()
    print('among violating nodes, what fraction are global sinks?')
    v = f[f.viol_nodes > 0]
    print('  %.3f  (n=%d instances with >=1 violation on the diagnosed draw)'
          % (v.viol_sinks.sum() / v.viol_nodes.sum(), len(v)))
