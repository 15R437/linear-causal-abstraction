"""Is AOC Closure better WITHOUT the All-in repair step?

Repair fixes All-in violations by MERGING a violating cluster with its child, which costs
intra-cluster edge mass -- permanently. But fit_abstraction_alternative can instead drive the
offending nodes' T-weight to ~0, paying only the mass incident to those nodes. If the nodes
are few and the merges are large, skipping repair should win.

Four arms on identical instances, differing ONLY in which draw they return:
  occ             widest fully-valid draw; if none, repair(widest)      (current method)
  occ_skiprepair  widest fully-valid draw; if none, widest UNREPAIRED   (isolates repair)
  occ_norepair    widest draw, All-in never checked                     (ignores All-in)
  occ_valid_only  widest fully-valid draw; if none, nothing             (coverage reference)

occ vs occ_skiprepair isolates the repair step alone. occ vs occ_norepair is the stronger
change of abandoning All-in when selecting among draws.

Scored exactly as the frontier figure: RAE (max over regimes, Sigma_u = I) and information
loss at the 1e-2 marginalisation tolerance.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/scripts')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, multiprocessing as mp
from src.utils.generate_data import sample_low_level_graph
from occ import one_child_closure, conditions, repair
from frontier_fig import score, P


def partitions(A, seed, n_draws=12):
    """Reproduce occ()'s draw loop, but expose the pre-repair options separately."""
    rng = np.random.default_rng(seed)
    best, best_g, widest, widest_g, n_ok = 0, None, 0, None, 0
    for _ in range(n_draws):
        g = one_child_closure(A, rng)
        if len(g) > widest:
            widest, widest_g = len(g), g
        if all(conditions(A, g)):
            n_ok += 1
            if len(g) > best:
                best, best_g = len(g), g
    out = {}
    if best_g is not None:
        out['occ'] = best_g
        out['occ_valid_only'] = best_g
        out['occ_skiprepair'] = best_g               # identical to occ when a valid draw exists
    else:
        out['occ'] = repair(A, widest_g)[0]          # the current fallback
        out['occ_skiprepair'] = widest_g             # ... skipped: let the fit marginalise
    out['occ_norepair'] = widest_g                   # widest draw, All-in never checked
    return out, dict(valid_draws=n_ok, repaired=best_g is None)


def job(a):
    dens, seed = a
    try:
        A, *_ = sample_low_level_graph(n_high=10, n_low=P, density=dens, shuffle=True,
            blockiness=0.5, heterogeneity=1., internal_density=0., graph_type='random',
            seed=seed, connected=True, max_attempts=300)
    except Exception:
        return []
    absA = np.abs(A); tot = absA.sum()
    E_low = np.linalg.inv(np.eye(P) - A); su = np.ones(P)
    parts, info = partitions(A, seed)
    rows = []
    for nm, g in parts.items():
        T = np.zeros((P, len(g)))
        for j, m in enumerate(g):
            T[np.asarray(m), j] = 1.0
        t0 = time.time()
        rows.append(score(A, T, E_low, su, absA, tot, method=nm, lam_oc=np.nan,
                          k=len(g), seed=seed, density=dens, secs=time.time()-t0,
                          repaired=int(info['repaired']), valid_draws=info['valid_draws']))
    return rows


if __name__ == '__main__':
    jobs = [(d, s) for d in (0.1, 0.3, 0.5, 0.7, 0.9) for s in range(6)]
    print(f'{len(jobs)} instances x 3 arms, p={P}', flush=True)
    t0 = time.time(); rows = []
    path = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/norepair.csv'
    with mp.Pool(6) as pool:
        for i, rs in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.extend(rs); pd.DataFrame(rows).to_csv(path, index=False)
            print(f'{i}/{len(jobs)} {(time.time()-t0)/60:.1f}min', flush=True)
    d = pd.DataFrame(rows); d.to_csv(path, index=False)
    pd.set_option('display.width', 220)
    print('\ndone %.1f min\n' % ((time.time()-t0)/60))
    print('repaired instances: %d of %d\n' % (d[d.method=='occ'].repaired.sum(),
                                              d.method.eq('occ').sum()))
    print(d.groupby('method').agg(n=('iem','size'), k=('k','mean'), largest=('largest','mean'),
        allin=('allin','mean'), one_child=('one_child','mean'), valid=('valid','mean'),
        rae_med=('rae','median'), iem=('iem','mean'), marg=('marg','mean'),
        exact=('rae', lambda s: float((s < 1e-8).mean()))).round(4).to_string())
    for other in ('occ_skiprepair', 'occ_norepair'):
        print('\nPAIRED, occ vs %s (same instances):' % other)
        a = d[d.method=='occ'].set_index(['density','seed'])
        b = d[d.method==other].set_index(['density','seed'])
        j = a.join(b, rsuffix='_x', how='inner')
        print(j.groupby(level=0).apply(lambda g: pd.Series({
            'n': len(g), 'iem_occ': g.iem.mean(), 'iem_other': g.iem_x.mean(),
            'differs': float((~np.isclose(g.iem, g.iem_x)).mean()),
            'other_wins': float((g.iem_x < g.iem).mean()),
            'marg_occ': g['marg'].mean(), 'marg_other': g.marg_x.mean()})).round(4).to_string())
    print('\n(old block below)')
    print('PAIRED, occ vs occ_norepair:')
    a = d[d.method=='occ'].set_index(['density','seed'])
    b = d[d.method=='occ_norepair'].set_index(['density','seed'])
    j = a.join(b, rsuffix='_nr', how='inner')
    print(j.groupby(level=0).apply(lambda g: pd.Series({
        'n': len(g), 'iem_occ': g.iem.mean(), 'iem_norepair': g.iem_nr.mean(),
        'rae_occ': g.rae.median(), 'rae_norepair': g.rae_nr.median(),
        'norepair_wins_on_iem': float((g.iem_nr < g.iem).mean()),
        'marg_occ': g['marg'].mean(), 'marg_norepair': g.marg_nr.mean()})).round(4).to_string())
