"""HailMary vs closure across random-DAG density -- the regime where closure is weak.

Both get the same k (the size of the best closure partition, defined whether or not that
partition is valid), so this isolates the search, not the budget.
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


def score(A, T):
    M = (T.T @ (A != 0).astype(float) @ T) > 0
    np.fill_diagonal(M, False)
    co = T @ T.T; np.fill_diagonal(co, 0)
    gl = graph_loss(A_low=A, T_hard=T)
    oc = int(M.sum(1).max() <= 1)
    return int(gl < 1e-8 and oc), int(gl < 1e-8), oc, \
        (np.abs(A) * co).sum() / np.abs(A).sum(), int((T.sum(0) > 0).sum())


def job(a):
    torch.set_num_threads(1)
    p, nh, dens, seed = a
    try:
        A, _, _, _ = sample_low_level_graph(
            n_high=nh, n_low=p, density=dens, shuffle=True, blockiness=0.5,
            heterogeneity=1., internal_density=0., graph_type='random', seed=seed,
            connected=True, max_attempts=400)
    except Exception:
        return None
    rng = np.random.default_rng(seed)
    best, bg = 0, None
    clo_ok = 0
    for _ in range(12):
        g = mk.closure(A, rng)
        valid = all(mk.conditions(A, g))
        clo_ok = max(clo_ok, int(valid))
        if len(g) > best and (valid or bg is None):
            best, bg = len(g), g
    k = max(2, best)
    Tc = np.zeros((p, len(bg)))
    for j, m in enumerate(bg):
        Tc[m, j] = 1
    c_s, c_a, c_o, c_i, _ = score(A, Tc)

    gen = torch.Generator().manual_seed(31337 + seed)
    li = closure_logits(A, k, rng=np.random.default_rng(31337 + seed), generator=gen)
    m = HailMary(n_low=p, n_high=k, A_low=torch.from_numpy(A).float(), logits_init=li)
    t0 = time.time()
    m.train(num_epochs=6000, lambda_recon=0., lambda_graph=10., lambda_one_child=100.,
            lambda_surj=0., lambda_info=10., verbose=False)
    T = m.get_partition()[0].detach().numpy()
    h_s, h_a, h_o, h_i, used = score(A, T)
    return dict(p=p, density=dens, seed=seed, k=k,
                clo_success=c_s, clo_allin=c_a, clo_onechild=c_o, clo_info=c_i,
                hm_success=h_s, hm_allin=h_a, hm_onechild=h_o, hm_info=h_i,
                hm_used=used, hm_secs=time.time() - t0)


if __name__ == '__main__':
    jobs = [(p, 10, d, s) for p in (50, 100) for d in (0.1, 0.2, 0.3, 0.5) for s in range(10)]
    print(f'{len(jobs)} instances', flush=True)
    with mp.Pool(6) as pool:
        rows = [r for r in pool.map(job, jobs) if r]
    d = pd.DataFrame(rows); d.to_csv(f'{SP}/lowdens.csv', index=False)
    print('%d instances\n' % len(d))
    print(d.groupby(['p', 'density']).agg(
        n=('clo_success', 'size'),
        clo_succ=('clo_success', 'mean'), hm_succ=('hm_success', 'mean'),
        clo_info=('clo_info', 'mean'), hm_info=('hm_info', 'mean'),
        k=('k', 'mean'), hm_used=('hm_used', 'mean'), secs=('hm_secs', 'mean')).round(3).to_string())
