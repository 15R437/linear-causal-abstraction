"""Two selection questions, done correctly.

(A) OC CLOSURE. Every draw is one-child by construction, so ALL draws are eligible --
    no validity filter (that was my error: filtering to All-In-valid draws answers the
    AOC question instead). widest-of-12 vs min-intra-of-12.

(B) AOC CLOSURE. Current design: keep the widest ALL-IN-VALID draw; repair one draw only
    if none is valid. Alternative: repair every invalid draw so all 12 candidates are
    valid, then select the one with least information loss. Costs more repair calls but
    gives the selection something to choose between -- which matters most at low density,
    where repair fires 71% of the time and costs ~4.9x the information loss.
"""
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, multiprocessing as mp
from src.utils.generate_data import sample_low_level_graph
from occ import one_child_closure, conditions, repair

P, N_DRAWS = 100, 12


def intra_of(absA, tot, groups):
    s = 0.0
    for g in groups:
        if len(g) > 1:
            sub = absA[np.ix_(g, g)]
            s += sub.sum() - np.trace(sub)
    return s / tot


def job(a):
    dens, seed = a
    try:
        A, *_ = sample_low_level_graph(n_high=10, n_low=P, density=dens, shuffle=True,
            blockiness=0.5, heterogeneity=1., internal_density=0., graph_type='random',
            seed=seed, connected=True, max_attempts=300)
    except Exception:
        return []
    absA = np.abs(A); tot = absA.sum()
    rng = np.random.default_rng(seed)
    t0 = time.time()
    draws = [one_child_closure(A, rng) for _ in range(N_DRAWS)]
    t_draws = time.time() - t0
    info = [(g, len(g), intra_of(absA, tot, g), all(conditions(A, g))) for g in draws]
    valid = [d for d in info if d[3]]

    # (A) OC Closure: all draws eligible
    oc_wide = max(info, key=lambda d: d[1])
    oc_mini = min(info, key=lambda d: d[2])

    # (B1) AOC as implemented: widest valid, else repair the widest draw
    t0 = time.time()
    if valid:
        aoc_cur = max(valid, key=lambda d: d[1]); n_rep_cur = 0
    else:
        g, nm = repair(A, max(info, key=lambda d: d[1])[0])
        aoc_cur = (g, len(g), intra_of(absA, tot, g), True); n_rep_cur = 1
    t_cur = time.time() - t0

    # (B2) repair EVERY invalid draw, then select least information loss
    t0 = time.time(); cands = []; n_rep_all = 0
    for g, k, iv, ok in info:
        if ok:
            cands.append((g, k, iv))
        else:
            gr, _ = repair(A, g); n_rep_all += 1
            cands.append((gr, len(gr), intra_of(absA, tot, gr)))
    aoc_alt = min(cands, key=lambda c: c[2])
    t_alt = time.time() - t0

    return [dict(density=dens, seed=seed, n_valid=len(valid),
                 oc_k_wide=oc_wide[1], oc_intra_wide=oc_wide[2],
                 oc_k_mini=oc_mini[1], oc_intra_mini=oc_mini[2],
                 aoc_k_cur=aoc_cur[1], aoc_intra_cur=aoc_cur[2],
                 aoc_k_alt=aoc_alt[1], aoc_intra_alt=aoc_alt[2],
                 n_rep_cur=n_rep_cur, n_rep_all=n_rep_all,
                 t_draws=t_draws, t_cur=t_cur, t_alt=t_alt)]


if __name__ == '__main__':
    jobs = [(d, i) for d in (0.1, 0.3, 0.5, 0.7, 0.9) for i in range(10)]
    rows = []
    with mp.Pool(6) as pool:
        for rs in pool.imap_unordered(job, jobs):
            rows.extend(rs)
    d = pd.DataFrame(rows)
    d.to_csv('/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study/data/occ_select2.csv', index=False)
    pd.set_option('display.width', 200)
    print('instances %d\n' % len(d))
    print('=== (A) OC CLOSURE: widest-of-12 vs min-intra-of-12 (ALL draws eligible) ===')
    t = d.groupby('density').agg(k_wide=('oc_k_wide','median'), intra_wide=('oc_intra_wide','median'),
        k_mini=('oc_k_mini','median'), intra_mini=('oc_intra_mini','median')).round(3)
    t['gain'] = (t.intra_wide - t.intra_mini).round(3)
    print(t.to_string())
    g = d.oc_intra_wide - d.oc_intra_mini
    print('pooled: %.3f -> %.3f | median gain %+.4f | better on %d/%d\n'
          % (d.oc_intra_wide.median(), d.oc_intra_mini.median(), np.median(g), (g > 1e-9).sum(), len(d)))

    print('=== (B) AOC: current (select-then-repair) vs repair-all-then-select ===')
    t = d.groupby('density').agg(n_valid=('n_valid','median'),
        k_cur=('aoc_k_cur','median'), intra_cur=('aoc_intra_cur','median'),
        k_alt=('aoc_k_alt','median'), intra_alt=('aoc_intra_alt','median'),
        reps_cur=('n_rep_cur','mean'), reps_alt=('n_rep_all','mean'),
        t_cur=('t_cur','mean'), t_alt=('t_alt','mean')).round(3)
    t['gain'] = (t.intra_cur - t.intra_alt).round(3)
    print(t.to_string())
    g = d.aoc_intra_cur - d.aoc_intra_alt
    print('pooled: %.3f -> %.3f | median gain %+.4f | better on %d/%d, worse on %d'
          % (d.aoc_intra_cur.median(), d.aoc_intra_alt.median(), np.median(g),
             (g > 1e-9).sum(), len(d), (g < -1e-9).sum()))
    print('COST: repairs %.1f -> %.1f per instance | selection time %.3fs -> %.3fs (draws alone %.3fs)'
          % (d.n_rep_cur.mean(), d.n_rep_all.mean(), d.t_cur.mean(), d.t_alt.mean(), d.t_draws.mean()))
