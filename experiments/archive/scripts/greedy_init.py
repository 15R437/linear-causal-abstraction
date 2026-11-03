"""HailMary initialised from the all-in greedy partition instead of AOC Closure.

Closure init starts in the exact-but-uninformative corner (intra ~ 0.56) and the restart
study showed HailMary gets trapped near it. The all-in greedy partition is the
complementary corner: All-In already satisfied, intra ~ 0.000, one-child badly violated.
Starting there leaves HailMary only one job -- fix one-child -- which is what lambda_oc
drives. Same logit construction as frontier_fig (5*onehot + 0.1*randn).
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
from partition_baselines import allin_agglom, violations
from frontier_fig import score, P, EPOCHS

DENS = [0.1, 0.3, 0.5, 0.7, 0.9]
LAMS = [0., 10., 100., 1000.]


def logits_from(T, K, seed):
    gen = torch.Generator().manual_seed(31337 + seed)
    base = np.zeros((P, K)); base[:, :min(K, T.shape[1])] = T[:, :K]
    return torch.from_numpy(base).float() * 5.0 + torch.randn(P, K, generator=gen) * 0.1


def job(a):
    torch.set_num_threads(1)
    dens, seed = a
    try:
        A, *_ = sample_low_level_graph(n_high=10, n_low=P, density=dens, shuffle=True,
            blockiness=0.5, heterogeneity=1., internal_density=0., graph_type='random',
            seed=seed, connected=True, max_attempts=300)
    except Exception:
        return []
    absA = np.abs(A); tot = absA.sum()
    E_low = np.linalg.inv(np.eye(P) - A); su = np.ones(P)
    g0, T0, ci = occ(A, seed=seed); K = max(ci['k'], 2)
    com = dict(k=K, seed=seed, density=dens)
    out = [score(A, T0, E_low, su, absA, tot, method='OCC', lam_oc=np.nan, secs=0., **com)]
    Tg = allin_agglom(A, K)
    r = score(A, Tg, E_low, su, absA, tot, method='all-in greedy', lam_oc=np.nan,
              secs=0., **com); r.update(violations(A, Tg)); out.append(r)
    # random init is the control: neither corner, so it isolates how much of any
    # greedy-init effect is the STARTING PARTITION rather than just 'not closure'.
    for tag, T_init in (('greedy-init', Tg), ('closure-init', T0), ('random-init', None)):
        for lam in LAMS:
            t0 = time.time()
            torch.manual_seed(31337 + seed)
            m = (HailMary(n_low=P, n_high=K, A_low=torch.from_numpy(A).float(),
                          init_type='random') if T_init is None else
                 HailMary(n_low=P, n_high=K, A_low=torch.from_numpy(A).float(),
                          logits_init=logits_from(T_init, K, seed)))
            m.train(num_epochs=EPOCHS, lambda_recon=1., lambda_graph=100.,
                    lambda_one_child=lam, lambda_info=0., lambda_surj=0, verbose=False)
            Th = m.get_partition()[0].detach().numpy()
            r = score(A, Th, E_low, su, absA, tot, method='HM ' + tag, lam_oc=lam,
                      secs=time.time() - t0, **com)
            r.update(violations(A, Th)); out.append(r)
    return out


if __name__ == '__main__':
    jobs = [(d, i) for d in DENS for i in range(10)]
    print('%d instances x 3 inits x %d lambdas (+refs), %d epochs'
          % (len(jobs), len(LAMS), EPOCHS), flush=True)
    t0 = time.time(); rows = []
    path = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/greedy_init.csv'
    with mp.Pool(6) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs); pd.DataFrame(rows).to_csv(path, index=False)
            if i % 5 == 0:
                print('%d/%d %.1fmin' % (i, len(jobs), (time.time()-t0)/60), flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print('done %.1f min' % ((time.time()-t0)/60), flush=True)
