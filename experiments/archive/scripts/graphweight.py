"""With one_child OFF, does scaling lambda_graph up make All-In bite?

The earlier no-one-child run used lambda_graph=10 vs lambda_info=10 -- equal weights, and
info is easy to drive to ~0, so it won every time (All-In satisfied 0/114). This sweeps
lambda_graph over two orders of magnitude, with a lambda_one_child=100 control.
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
from src.utils.losses import graph_loss
_s = importlib.util.spec_from_file_location(
    'mk', '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts/max_k_estimate.py')
mk = importlib.util.module_from_spec(_s); _s.loader.exec_module(mk)
SP = '/private/tmp/claude-501/-Users-issy-Projects-linear-causal-abstraction/66cd7cf1-4427-4b0d-befb-2e39ce18ba28/scratchpad'


def job(a):
    torch.set_num_threads(1)
    gt, dens, p, lg, loc, seed = a
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
    gen = torch.Generator().manual_seed(31337 + seed)
    li = closure_logits(A, k, rng=np.random.default_rng(31337 + seed), generator=gen)
    m = HailMary(n_low=p, n_high=k, A_low=torch.from_numpy(A).float(), logits_init=li)
    m.train(num_epochs=6000, lambda_recon=0., lambda_graph=lg, lambda_one_child=loc,
            lambda_surj=0., lambda_info=10., verbose=False)
    T = m.get_partition()[0].detach().numpy()
    M = (T.T @ (A != 0).astype(float) @ T) > 0
    np.fill_diagonal(M, False)
    co = T @ T.T; np.fill_diagonal(co, 0)
    gl = graph_loss(A_low=A, T_hard=T)
    oc = int(M.sum(1).max() <= 1)
    return dict(graph_type=gt, p=p, lambda_graph=lg, lambda_one_child=loc, seed=seed, k=k,
                allin=int(gl < 1e-8), one_child=oc, both=int(gl < 1e-8 and oc),
                graph_loss=float(gl), info=(np.abs(A) * co).sum() / np.abs(A).sum(),
                used=int((T.sum(0) > 0).sum()), max_clust=int(T.sum(0).max()),
                high_edges=int(M.sum()))


if __name__ == '__main__':
    jobs = []
    for gt, dens in [('random', 0.5), ('one_child', 1.0)]:
        for lg in (10., 50., 200., 1000.):
            for s in range(8):
                jobs.append((gt, dens, 100, lg, 0., s))        # one_child loss OFF
        for s in range(8):
            jobs.append((gt, dens, 100, 10., 100., s))          # control: standard config
    print(f'{len(jobs)} runs', flush=True)
    t0 = time.time(); rows = []
    with mp.Pool(6) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.append(r)
            if i % 20 == 0:
                print(f'{i}/{len(jobs)} {(time.time()-t0)/60:.1f}min', flush=True)
    d = pd.DataFrame(rows); d.to_csv(f'{SP}/graphweight.csv', index=False)
    print('\n%d runs\n' % len(d))
    d['arm'] = np.where(d.lambda_one_child > 0, 'CONTROL lg=10 oc=100',
                        'oc=0  lg=' + d.lambda_graph.astype(int).astype(str))
    print(d.groupby(['graph_type', 'arm']).agg(
        n=('allin', 'size'), allin=('allin', 'mean'), one_child=('one_child', 'mean'),
        both=('both', 'mean'), info=('info', 'mean'), used=('used', 'mean'),
        maxc=('max_clust', 'mean'), hedges=('high_edges', 'mean')).round(3).to_string())
