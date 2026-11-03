"""Does one_child_loss track one-child satisfaction?

Same design as graphloss_tracks.py. Perturb OCC's partition by reassigning a fraction of
nodes, giving a controlled spread of "excess children" (sum over clusters of max(children-1,0)),
then ask whether the SOFT one_child_loss (computed on T_soft) tracks it.

Note the soft loss is evaluated at several softmax temperatures, because a partition that is
hard (margin large) and one that is soft can have very different loss values for the SAME
hard assignment -- which is the suspected failure mode.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
warnings.filterwarnings('ignore')
import numpy as np, torch, pandas as pd, multiprocessing as mp
from src.models.hail_mary import HailMary
from src.utils.generate_data import sample_low_level_graph
from occ import occ, high_level_graph

FRACS = np.linspace(0, 1, 11)
MARGINS = [2.0, 12.0]


def excess_children(Ab, T):
    groups = [g for g in (np.where(T[:, j] == 1)[0] for j in range(T.shape[1])) if len(g)]
    if not groups:
        return 0, 0
    M = high_level_graph(Ab, groups)
    ch = M.sum(1)
    return int(np.maximum(ch - 1, 0).sum()), int(ch.max())


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
    Ab = (A != 0)
    g0, T0, ci = occ(A, seed=seed)
    K = T0.shape[1]
    At = torch.from_numpy(A).float()
    rng = np.random.default_rng(seed)
    rows = []
    for f in FRACS:
        for rep in range(3):
            T = T0.copy()
            n_move = int(round(f * p))
            if n_move:
                who = rng.choice(p, n_move, replace=False)
                T[who] = 0
                T[who, rng.integers(0, K, n_move)] = 1
            keep = T.sum(0) > 0
            if keep.sum() < 2:
                continue
            Tk = T[:, keep]
            exc, mx = excess_children(Ab, Tk)
            for mg in MARGINS:
                m = HailMary(n_low=p, n_high=Tk.shape[1], A_low=At,
                             logits_init=torch.from_numpy(Tk).float() * mg)
                with torch.no_grad():
                    _, _, oc, _, _, _ = m.forward(1.0, compute_recon=False, compute_graph=False)
                rows.append(dict(p=p, density=dens, seed=seed, frac=float(f), rep=rep,
                                 margin=mg, excess=exc, max_children=mx,
                                 satisfied=int(mx <= 1), oc_loss=float(oc)))
    return rows


if __name__ == '__main__':
    jobs = [(p, d, s) for p in (50, 100) for d in (0.1, 0.5, 0.9) for s in range(3)]
    with mp.Pool(6) as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    d = pd.DataFrame(rows)
    d.to_csv('/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/onechild_tracks.csv', index=False)
    from scipy import stats
    print('%d partitions, %d instances\n' % (len(d), d.groupby(['p','density','seed']).ngroups))
    for mg, g in d.groupby('margin'):
        print('=== margin %.0f %s ===' % (mg, '(soft)' if mg < 5 else '(near-hard)'))
        sat = g[g.satisfied == 1]; vio = g[g.satisfied == 0]
        print('  satisfied   (n=%4d): oc_loss  mean %.4f  median %.4f  max %.4f'
              % (len(sat), sat.oc_loss.mean(), sat.oc_loss.median(), sat.oc_loss.max()))
        print('  VIOLATING   (n=%4d): oc_loss  mean %.4f  median %.4f  min %.4f'
              % (len(vio), vio.oc_loss.mean(), vio.oc_loss.median(), vio.oc_loss.min()))
        overlap = (sat.oc_loss.max() > vio.oc_loss.min()) if len(sat) and len(vio) else None
        print('  separable by magnitude? %s' % ('NO (ranges overlap)' if overlap else 'yes'))
        sp = stats.spearmanr(g.oc_loss, g.excess)
        print('  Spearman(oc_loss, excess children) = %+.3f  (p=%.1e)' % (sp.correlation, sp.pvalue))
        r = [stats.spearmanr(x.oc_loss, x.excess).correlation
             for _, x in g.groupby(['p','density','seed']) if x.excess.nunique() > 1]
        r = np.array([v for v in r if not np.isnan(v)])
        print('  within-instance: mean %+.3f  positive %d/%d\n' % (r.mean(), int((r>0).sum()), len(r)))
