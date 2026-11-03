"""Should graph_loss keep the `* edge_weight` term?

Current:  H profiles come from A_low_binary (structural) but the pairwise penalty is weighted
by  edge_weight = |A_low| @ T_soft  (magnitudes). All-In is a structural condition, so the
magnitude dependence is arguably wrong -- but edge_weight may also be acting as a relevance
MASK, suppressing disagreement on clusters a node does not point into.

Three variants, scored on how well each tracks the All-In violation count:
  weighted : (current)            edge_weight = |A_low| @ T_soft
  binary   : mask without magnitude   edge_weight = A_low_binary @ T_soft
  none     : no weighting at all      edge_weight = 1
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
warnings.filterwarnings('ignore')
import numpy as np, torch, torch.nn.functional as F, pandas as pd, multiprocessing as mp
from src.utils.generate_data import sample_low_level_graph
from occ import occ, high_level_graph

FRACS = np.linspace(0, 1, 11)


def graph_loss_variant(A, T_soft, mode):
    p = A.shape[0]
    At = torch.from_numpy(A).float()
    Ab = (At != 0).float()
    T = T_soft
    co = T @ T.T
    dm = 1.0 - torch.eye(p)
    H = torch.max(Ab[:, None, :] * T.T[None, :, :], dim=-1)[0] * (1 - T)   # (p,k)
    Hs = torch.clamp(H, 1e-6, 1 - 1e-6)
    Hl = torch.logit(Hs)
    tgt = Hs.detach()[:, None, :]
    lgt = Hl[None, :, :]
    ce = tgt * F.softplus(-lgt) + (1 - tgt) * F.softplus(lgt)              # (p,p,k)
    if mode == 'weighted':
        w = (At.abs() @ T)[None, :, :]
    elif mode == 'binary':
        w = (Ab @ T)[None, :, :]
    else:
        w = torch.ones(1, p, T.shape[1])
    return float(((co * dm) * (ce * w).sum(-1)).mean())


def violations(Ab, T):
    groups = [g for g in (np.where(T[:, j] == 1)[0] for j in range(T.shape[1])) if len(g)]
    M = high_level_graph(Ab, groups)
    bad = set()
    for x in range(len(groups)):
        for y in np.where(M[x])[0]:
            for i in groups[x]:
                if not Ab[i, groups[y]].any():
                    bad.add(int(i))
    return len(bad)


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
    rng = np.random.default_rng(seed)
    rows = []
    for f in FRACS:
        for rep in range(3):
            T = T0.copy()
            n = int(round(f * p))
            if n:
                who = rng.choice(p, n, replace=False)
                T[who] = 0
                T[who, rng.integers(0, K, n)] = 1
            keep = T.sum(0) > 0
            if keep.sum() < 2:
                continue
            Tk = T[:, keep]
            v = violations(Ab, Tk)
            Ts = torch.softmax(torch.from_numpy(Tk).float() * 4.0, dim=-1)
            rows.append(dict(p=p, density=dens, seed=seed, viol=v,
                             weighted=graph_loss_variant(A, Ts, 'weighted'),
                             binary=graph_loss_variant(A, Ts, 'binary'),
                             none=graph_loss_variant(A, Ts, 'none')))
    return rows


if __name__ == '__main__':
    jobs = [(p, d, s) for p in (50, 100) for d in (0.1, 0.5, 0.9) for s in range(4)]
    with mp.Pool(6) as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    d = pd.DataFrame(rows)
    d.to_csv('/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/edgeweight_ablation.csv', index=False)
    from scipy import stats
    print('%d partitions, %d instances\n' % (len(d), d.groupby(['p','density','seed']).ngroups))
    print('how well does each variant track the All-In violation count?')
    print('  variant    pooled Spearman   within-instance mean   positive   separates 0-viol?')
    for m in ('weighted', 'binary', 'none'):
        sp = stats.spearmanr(d[m], d.viol).correlation
        r = [stats.spearmanr(g[m], g.viol).correlation
             for _, g in d.groupby(['p','density','seed']) if g.viol.nunique() > 1]
        r = np.array([v for v in r if not np.isnan(v)])
        nv, wv = d[d.viol == 0][m], d[d.viol > 0][m]
        sep = 'yes' if (len(nv) and len(wv) and nv.max() < wv.min()) else 'no (overlap)'
        print('  %-9s  %+.3f            %+.3f                %2d/%2d      %s'
              % (m, sp, r.mean(), int((r > 0).sum()), len(r), sep))
    print('\nmagnitude sensitivity: rescale ALL edge weights by 10x, does the loss change?')
    print('  (All-In is structural, so a correct loss should be INVARIANT)')
    A, *_ = sample_low_level_graph(n_high=10, n_low=50, density=0.5, shuffle=True, blockiness=0.5,
                                   heterogeneity=1., internal_density=0., graph_type='random',
                                   seed=0, connected=True, max_attempts=300)
    g0, T0, ci = occ(A, seed=0)
    Ts = torch.softmax(torch.from_numpy(T0).float() * 4.0, dim=-1)
    for m in ('weighted', 'binary', 'none'):
        a1 = graph_loss_variant(A, Ts, m)
        a2 = graph_loss_variant(A * 10.0, Ts, m)
        print('  %-9s  x1: %.6f   x10: %.6f   ratio %.2f' % (m, a1, a2, a2 / max(a1, 1e-12)))
