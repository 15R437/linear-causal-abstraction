"""(a) How BAD are HailMary's All-In failures? (b) Is its info edge just the closure init?

(a) counts actual All-In violations -- pairs (node i, high-level edge X->Y) where i is in X
    but has no edge into Y. Those nodes are what fit_abstraction marginalises, so the count
    matters, not just the binary pass/fail. losses.graph_loss is a pairwise profile distance,
    NOT a violation count, so it can't answer this.
(b) runs HailMary from a RANDOM init on the same graphs at the same k. If the info advantage
    survives, it is the optimiser's; if it collapses, it was the initialisation's.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, importlib.util, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary, closure_logits
from src.utils.generate_data import sample_low_level_graph
_s = importlib.util.spec_from_file_location(
    'mk', '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts/max_k_estimate.py')
mk = importlib.util.module_from_spec(_s); _s.loader.exec_module(mk)
SP = '/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad'


def violations(A, T):
    """(#violating (node,edge) pairs, #distinct violating nodes, #high-level edges)."""
    Ab = (A != 0)
    groups = [np.where(T[:, j] == 1)[0] for j in range(T.shape[1])]
    groups = [g for g in groups if len(g)]
    K = len(groups)
    M = np.zeros((K, K), bool)
    for x in range(K):
        for y in range(K):
            if x != y and Ab[np.ix_(groups[x], groups[y])].any():
                M[x, y] = True
    pairs, nodes = 0, set()
    for x in range(K):
        for y in np.where(M[x])[0]:
            for i in groups[x]:
                if not Ab[i, groups[y]].any():
                    pairs += 1
                    nodes.add(int(i))
    return pairs, len(nodes), int(M.sum()), K


def run(A, p, k, init, seed):
    gen = torch.Generator().manual_seed(31337 + seed)
    if init == 'closure':
        li = closure_logits(A, k, rng=np.random.default_rng(31337 + seed), generator=gen)
    else:
        li = torch.randn(p, k, generator=gen) * 1.0
    m = HailMary(n_low=p, n_high=k, A_low=torch.from_numpy(A).float(), logits_init=li)
    m.train(num_epochs=6000, lambda_recon=0., lambda_graph=10., lambda_one_child=100.,
            lambda_surj=0., lambda_info=10., verbose=False)
    T = m.get_partition()[0].detach().numpy()
    vp, vn, he, K = violations(A, T)
    M = (T.T @ (A != 0).astype(float) @ T) > 0
    np.fill_diagonal(M, False)
    co = T @ T.T; np.fill_diagonal(co, 0)
    return dict(init=init, allin=int(vp == 0), one_child=int(M.sum(1).max() <= 1),
                viol_pairs=vp, viol_nodes=vn, viol_node_frac=vn / p, high_edges=he,
                used=K, info=(np.abs(A) * co).sum() / np.abs(A).sum())


def job(a):
    torch.set_num_threads(1)
    gt, dens, p, seed = a
    A, _, _, _ = sample_low_level_graph(
        n_high=10, n_low=p, density=dens, shuffle=True, blockiness=0.5, heterogeneity=1.,
        internal_density=0., graph_type=gt, seed=seed, connected=True, max_attempts=400)
    rng = np.random.default_rng(seed)
    best = 0
    for _ in range(12):
        g = mk.closure(A, rng)
        if all(mk.conditions(A, g)) and len(g) > best:
            best = len(g)
    k = max(2, best if best else 10)
    out = []
    for init in ('closure', 'random'):
        r = run(A, p, k, init, seed)
        r.update(graph_type=gt, p=p, density=dens, seed=seed, k=k)
        out.append(r)
    return out


if __name__ == '__main__':
    jobs = [(gt, d, 100, s) for gt, d in [('random', 0.5), ('one_child', 1.0)]
            for s in range(10)]
    print(f'{len(jobs)} instances x 2 inits', flush=True)
    with mp.Pool(6) as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    d = pd.DataFrame(rows); d.to_csv(f'{SP}/violinit.csv', index=False)
    print('\n=== (b) closure init vs random init ===')
    print(d.groupby(['graph_type', 'init']).agg(
        n=('allin', 'size'), allin=('allin', 'mean'), one_child=('one_child', 'mean'),
        info=('info', 'mean'), used=('used', 'mean')).round(3).to_string())
    print('\n=== (a) severity of All-In failures (failing runs only) ===')
    f = d[d.allin == 0]
    print(f.groupby(['graph_type', 'init']).agg(
        n=('viol_pairs', 'size'), viol_pairs=('viol_pairs', 'mean'),
        viol_nodes=('viol_nodes', 'mean'), frac_of_p=('viol_node_frac', 'mean'),
        high_edges=('high_edges', 'mean')).round(3).to_string())
