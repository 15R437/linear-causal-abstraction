"""Does HailMary do better warm-started from occ_skiprepair than from occ?

The two partitions are IDENTICAL unless no closure draw satisfies All-in, which happens on
~11% of instances -- so a random sample would be almost all no-ops. This script first scans
seeds cheaply (closure is ~0.1s) to find instances where repair fires, then runs HailMary
from both warm starts on those instances only.

occ            : widest fully-valid draw; none exists -> repair(widest)   [merges to fix All-in]
occ_skiprepair : widest fully-valid draw; none exists -> widest unrepaired [fit marginalises]
"""
import os, sys, time, warnings
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary
from src.utils.generate_data import sample_low_level_graph
from occ import one_child_closure, conditions, repair
from frontier_fig import score, P

LAMS = [10., 1000.]
EPOCHS = 3000
N_TARGET = 6


def draws(A, seed, n_draws=12):
    rng = np.random.default_rng(seed)
    best, best_g, widest, widest_g = 0, None, 0, None
    for _ in range(n_draws):
        g = one_child_closure(A, rng)
        if len(g) > widest:
            widest, widest_g = len(g), g
        if all(conditions(A, g)) and len(g) > best:
            best, best_g = len(g), g
    return best_g, widest_g


def gen(dens, seed):
    return sample_low_level_graph(n_high=10, n_low=P, density=dens, shuffle=True,
        blockiness=0.5, heterogeneity=1., internal_density=0., graph_type='random',
        seed=seed, connected=True, max_attempts=300)[0]


def scan(a):
    """Cheap: does this instance need repair?"""
    dens, seed = a
    try:
        A = gen(dens, seed)
    except Exception:
        return None
    best_g, widest_g = draws(A, seed)
    return (dens, seed) if best_g is None else None


def T_of(g, k):
    T = np.zeros((P, len(g)))
    for j, m in enumerate(g):
        T[np.asarray(m), j] = 1.0
    return T


def job(a):
    torch.set_num_threads(1)
    dens, seed = a
    A = gen(dens, seed)
    absA = np.abs(A); tot = absA.sum()
    E_low = np.linalg.inv(np.eye(P) - A); su = np.ones(P)
    best_g, widest_g = draws(A, seed)
    assert best_g is None, 'only instances needing repair'
    warm = {'occ': repair(A, widest_g)[0], 'occ_skiprepair': widest_g}
    rows = []
    for wname, g in warm.items():
        T0 = T_of(g, len(g)); K = T0.shape[1]
        rows.append(score(A, T0, E_low, su, absA, tot, warm=wname, method='warmstart_only',
                          lam_oc=np.nan, k=K, seed=seed, density=dens, secs=0.))
        for lam in LAMS:
            t0 = time.time()
            gen_t = torch.Generator().manual_seed(31337 + seed)
            base = np.zeros((P, K)); base[:, :K] = T0
            init = torch.from_numpy(base).float() * 5.0 + torch.randn(P, K, generator=gen_t) * 0.1
            m = HailMary(n_low=P, n_high=K, A_low=torch.from_numpy(A).float(), logits_init=init)
            m.train(num_epochs=EPOCHS, lambda_recon=1., lambda_graph=100.,
                    lambda_one_child=lam, lambda_info=0., lambda_surj=0, verbose=False)
            rows.append(score(A, m.get_partition()[0].detach().numpy(), E_low, su, absA, tot,
                              warm=wname, method='HailMary', lam_oc=lam, k=K, seed=seed,
                              density=dens, secs=time.time()-t0))
    return rows


if __name__ == '__main__':
    cand = [(d, s) for d in (0.1, 0.3, 0.5) for s in range(60)]
    print('scanning %d instances for ones where repair fires...' % len(cand), flush=True)
    with mp.Pool(6) as pool:
        hits = [r for r in pool.map(scan, cand) if r is not None]
    print('  found %d; using the first %d\n' % (len(hits), min(len(hits), N_TARGET)), flush=True)
    jobs = hits[:N_TARGET]
    t0 = time.time()
    with mp.Pool(6) as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    d = pd.DataFrame(rows)
    d.to_csv('/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/warmstart.csv', index=False)
    pd.set_option('display.width', 220)
    print('done %.1f min   (%d instances where the two warm starts differ)\n'
          % ((time.time()-t0)/60, len(jobs)))
    print('the warm starts themselves:')
    print(d[d.method=='warmstart_only'].groupby('warm').agg(
        n=('iem','size'), k=('k','mean'), iem=('iem','mean'), rae=('rae','median'),
        marg=('marg','mean'), allin=('allin','mean')).round(4).to_string())
    print('\nHailMary trained from each:')
    print(d[d.method=='HailMary'].groupby(['lam_oc','warm']).agg(
        n=('iem','size'), iem=('iem','mean'), rae_med=('rae','median'),
        marg=('marg','mean'), k_used=('k_used','mean'), allin=('allin','mean')).round(4).to_string())
    print('\npaired per instance (HailMary iem, skiprepair minus occ; negative = skiprepair better):')
    h = d[d.method=='HailMary']
    for lam in LAMS:
        s = h[h.lam_oc==lam].pivot_table(index=['density','seed'], columns='warm', values='iem')
        if {'occ','occ_skiprepair'} <= set(s.columns):
            diff = (s.occ_skiprepair - s.occ)
            print('  lam=%-6g mean diff %+.4f   skiprepair better on %d/%d'
                  % (lam, diff.mean(), int((diff<0).sum()), len(diff)))
