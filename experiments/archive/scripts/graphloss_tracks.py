"""Does graph_loss actually track All-In violations?

Two quantities share the name and only one is a training signal:
  HARD  losses.graph_loss(A_low, T_hard) -- the validity check ("graph_loss == 0")
  SOFT  the graph_loss term in HailMary.forward, computed on T_soft -- the gradient signal

For each we ask two things:
  (i)  is it ZERO exactly when there are no All-In violations?   (a correctness check)
  (ii) does its MAGNITUDE track the number of violating nodes?   (is it a useful signal?)

Partitions are generated with a controlled spread of violations by taking OCC's valid
partition and randomly reassigning a fraction f of nodes, f = 0 .. 1.
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
from src.utils.losses import graph_loss as hard_graph_loss
from occ import occ, high_level_graph

FRACS = np.linspace(0, 1, 11)


def violations(Ab, T):
    groups = [g for g in (np.where(T[:, j] == 1)[0] for j in range(T.shape[1])) if len(g)]
    M = high_level_graph(Ab, groups)
    bad = set()
    for x in range(len(groups)):
        for y in np.where(M[x])[0]:
            for i in groups[x]:
                if not Ab[i, groups[y]].any():
                    bad.add(int(i))
    return len(bad), len(groups), int(M.sum(1).max() <= 1) if len(groups) else 1


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
            v, k_used, oc = violations(Ab, Tk)
            hard = float(hard_graph_loss(A_low=A, T_hard=Tk))
            m = HailMary(n_low=p, n_high=Tk.shape[1], A_low=At,
                         logits_init=torch.from_numpy(Tk).float() * 12.0)
            with torch.no_grad():
                _, soft, _, _, _, _ = m.forward(1.0, compute_recon=False)
            rows.append(dict(p=p, density=dens, seed=seed, frac=float(f), rep=rep,
                             viol=v, k_used=k_used, one_child=oc,
                             hard_gl=hard, soft_gl=float(soft)))
    return rows


if __name__ == '__main__':
    jobs = [(p, d, s) for p in (50, 100) for d in (0.1, 0.5, 0.9) for s in range(3)]
    with mp.Pool(6) as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    d = pd.DataFrame(rows)
    d.to_csv('/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/graphloss_tracks.csv', index=False)
    from scipy import stats
    print('%d partitions across %d instances\n' % (len(d), d.groupby(['p','density','seed']).ngroups))

    print('=== (i) is graph_loss ZERO exactly when there are no violations? ===')
    for name in ('hard_gl', 'soft_gl'):
        z = d[name] <= 1e-9
        nv = d.viol == 0
        tp = int((z & nv).sum()); fp = int((z & ~nv).sum())
        fn = int((~z & nv).sum()); tn = int((~z & ~nv).sum())
        print('  %-8s zero&no-viol %4d | zero&VIOL %4d | nonzero&no-viol %4d | nonzero&viol %4d'
              % (name, tp, fp, fn, tn))
        if fp or fn:
            print('           -> NOT an exact indicator (%d disagreements of %d)' % (fp + fn, len(d)))
        else:
            print('           -> exact indicator')

    print('\n=== (ii) does the MAGNITUDE track the violation count? ===')
    for name in ('hard_gl', 'soft_gl'):
        sp = stats.spearmanr(d[name], d.viol)
        pe = stats.pearsonr(d[name], d.viol)
        print('  %-8s pooled: Spearman %+.3f (p=%.1e)   Pearson %+.3f' % (name, sp.correlation, sp.pvalue, pe[0]))
    print()
    res = {'hard_gl': [], 'soft_gl': []}
    for kk, g in d.groupby(['p', 'density', 'seed']):
        for name in res:
            if g[name].nunique() > 1 and g.viol.nunique() > 1:
                res[name].append(stats.spearmanr(g[name], g.viol).correlation)
    for name, v in res.items():
        v = np.array(v)
        print('  %-8s within-instance Spearman: mean %+.3f  median %+.3f  positive %d/%d'
              % (name, v.mean(), np.median(v), int((v > 0).sum()), len(v)))

    print('\n=== among partitions with NO violations, what does graph_loss look like? ===')
    nv = d[d.viol == 0]
    print(nv[['hard_gl','soft_gl']].describe().loc[['count','mean','50%','max']].round(4).to_string())
    print('\n=== among partitions WITH violations ===')
    wv = d[d.viol > 0]
    print(wv[['hard_gl','soft_gl','viol']].describe().loc[['count','mean','50%','min']].round(4).to_string())
