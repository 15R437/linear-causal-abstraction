"""Do INDEPENDENT restarts buy anything on a single instance?

restarts.csv found 5 restarts ~ 1 restart, but those restarts initialised from
closure_logits -- and frontier_fig.py likewise uses 5.0*onehot(pi_0) + 0.1*randn, a margin
of 5 against noise of 0.1. Those are jitter around one point, not independent tries, so the
earlier null result may measure the initialisation rather than the landscape.

This repeats the question with init_type='random' (regularised_surjective_init, distinct
seed per run), which on the Leontief graph produced genuinely varied partitions. Reports
best-of-n against n=1 for both RAE and information loss, so the value of restarts can be
read off directly.
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
from src.utils.generate_data import sample_low_level_graph
from occ import occ
from frontier_fig import score, P

R = 10                      # independent restarts per (instance, lambda)
LAMS = [0., 1000.]
EPOCHS = 3000


def job(a):
    torch.set_num_threads(1)
    dens, inst, lam, r = a
    try:
        A, *_ = sample_low_level_graph(n_high=10, n_low=P, density=dens, shuffle=True,
            blockiness=0.5, heterogeneity=1., internal_density=0., graph_type='random',
            seed=inst, connected=True, max_attempts=300)
    except Exception:
        return []
    absA = np.abs(A); tot = absA.sum()
    E_low = np.linalg.inv(np.eye(P) - A); su = np.ones(P)
    g0, T0, ci = occ(A, seed=inst); K = max(ci['k'], 2)
    out = []
    if lam == LAMS[0] and r == 0:      # OCC reference once per instance
        out.append(score(A, T0, E_low, su, absA, tot, method='OCC', lam_oc=np.nan,
                         k=K, seed=inst, density=dens, restart=-1, secs=0.))
    t0 = time.time()
    torch.manual_seed(10_000 * inst + 97 * r + int(lam))
    m = HailMary(n_low=P, n_high=K, A_low=torch.from_numpy(A).float(), init_type='random')
    m.train(num_epochs=EPOCHS, lambda_recon=1., lambda_graph=100., lambda_one_child=lam,
            lambda_info=0., lambda_surj=0, verbose=False)
    out.append(score(A, m.get_partition()[0].detach().numpy(), E_low, su, absA, tot,
                     method='HailMary', lam_oc=lam, k=K, seed=inst, density=dens,
                     restart=r, secs=time.time()-t0))
    return out


if __name__ == '__main__':
    jobs = [(d, i, l, r) for d in (0.1, 0.5, 0.9) for i in range(4)
            for l in LAMS for r in range(R)]
    print('%d runs: 3 densities x 4 instances x %d lambdas x %d independent restarts'
          % (len(jobs), len(LAMS), R), flush=True)
    t0 = time.time(); rows = []
    path = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/restarts_randinit.csv'
    with mp.Pool(6) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs)
            if i % 20 == 0:
                pd.DataFrame(rows).to_csv(path, index=False)
                print('%d/%d %.1fmin' % (i, len(jobs), (time.time()-t0)/60), flush=True)
    d = pd.DataFrame(rows); d.to_csv(path, index=False)
    pd.set_option('display.width', 220)
    print('\ndone %.1f min\n' % ((time.time()-t0)/60))
    h = d[d.method == 'HailMary']
    print('DOES best-of-%d BEAT a single run?  (median over the 12 instances)' % R)
    for lam in LAMS:
        print('\n  lambda_oc = %g' % lam)
        for dens, g in h[h.lam_oc == lam].groupby('density'):
            one, best_r, best_i = [], [], []
            for inst, gi in g.groupby('seed'):
                one.append(gi[gi.restart == 0].rae.iloc[0])
                best_r.append(gi.rae.min())
                best_i.append(gi.iem.min())
            first_i = [g[(g.seed == s) & (g.restart == 0)].iem.iloc[0] for s in g.seed.unique()]
            print('    d=%.1f  RAE: single %.4f -> best-of-%d %.4f   |  IEM: single %.3f -> best-of-%d %.3f'
                  % (dens, np.median(one), R, np.median(best_r),
                     np.median(first_i), R, np.median(best_i)))
