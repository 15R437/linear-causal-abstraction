"""Run closure on a toy graph and LOOK at what it produces.

    venv/bin/python experiments/abstraction_study/scripts/explore.py
    venv/bin/python .../explore.py --p 12 --nhigh 4 --density 0.5 --seed 3
    venv/bin/python .../explore.py --p 10 --seed 0 --brute      # also enumerate every
                                                                # valid partition
Prints the low-level edges, the clusters closure returns, the induced high-level graph,
and the size distribution. With --brute (p <= 10) it also shows the most balanced valid
partition that EXISTS, so you can see what closure is or is not leaving behind.
Writes a picture via the project's own plotting machinery unless --noplot.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, argparse, importlib.util, warnings
sys.path.insert(0, '/Users/issy/Projects/linear-causal-abstraction')
warnings.filterwarnings('ignore')
import numpy as np
from src.utils.generate_data import sample_low_level_graph
B = '/Users/issy/Projects/linear-causal-abstraction'
_s = importlib.util.spec_from_file_location('mk', f'{B}/experiments/abstraction_study/scripts/max_k_estimate.py')
mk = importlib.util.module_from_spec(_s); _s.loader.exec_module(mk)
_f = importlib.util.spec_from_file_location('fb', f'{B}/experiments/abstraction_study/scripts/fallback.py')
fb = importlib.util.module_from_spec(_f); _f.loader.exec_module(fb)

ap = argparse.ArgumentParser()
ap.add_argument('--p', type=int, default=12)
ap.add_argument('--nhigh', type=int, default=4)
ap.add_argument('--density', type=float, default=0.5)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--graph', default='random', choices=['random', 'one_child', 'chain'])
ap.add_argument('--draws', type=int, default=12)
ap.add_argument('--brute', action='store_true')
ap.add_argument('--noplot', action='store_true')
a = ap.parse_args()

A, _, _, _ = sample_low_level_graph(
    n_high=a.nhigh, n_low=a.p, density=a.density, shuffle=True, blockiness=0.5,
    heterogeneity=1., internal_density=0., graph_type=a.graph, seed=a.seed,
    connected=True, max_attempts=400)
Ab = (A != 0); absA = np.abs(A)
print(f'\nLOW-LEVEL GRAPH  p={a.p}  {a.graph}  density={a.density}  seed={a.seed}')
print(f'  {Ab.sum()} edges,  out-degree min {Ab.sum(1).min()} / mean {Ab.sum(1).mean():.1f} / max {Ab.sum(1).max()}')
for i in range(a.p):
    ch = np.where(Ab[i])[0]
    if len(ch):
        print(f'    {i:2d} -> {", ".join(str(c) for c in ch)}')

rng = np.random.default_rng(a.seed)
best, bg, bany, ba = 0, None, 0, None
ok_draws = 0
for _ in range(a.draws):
    g = mk.closure(A, rng)
    if all(mk.conditions(A, g)):
        ok_draws += 1
        if len(g) > best:
            best, bg = len(g), g
    if len(g) > bany:
        bany, ba = len(g), g
used_fb = bg is None
if used_fb:
    bg, n_merges = fb.targeted_merge(Ab, ba)
    print(f'\n  !! no valid closure draw in {a.draws}; targeted-merge fallback used '
          f'({n_merges} merges)')
print(f'\nCLOSURE  ({ok_draws}/{a.draws} draws valid)')


def show(groups, label):
    K = len(groups)
    M = np.zeros((K, K), bool)
    for x in range(K):
        for y in range(K):
            if x != y and Ab[np.ix_(groups[x], groups[y])].any():
                M[x, y] = True
    T = np.zeros((a.p, K))
    for j, m in enumerate(groups):
        T[m, j] = 1
    co = T @ T.T; np.fill_diagonal(co, 0)
    info = (absA * co).sum() / absA.sum()
    sizes = sorted((len(g) for g in groups), reverse=True)
    q = np.array(sizes, float); q = q / q.sum()
    bal = float(-(q * np.log(q)).sum() / np.log(len(q))) if len(q) > 1 else 0.
    ai, oc, dag = mk.conditions(A, groups)
    print(f'  {label}: k={K}   All-In={ai}  one-child={oc}  acyclic={dag}')
    for j, m in enumerate(groups):
        kids = np.where(M[j])[0]
        print(f'    Y{j:<2d} = {{{", ".join(str(int(x)) for x in sorted(m))}}}'
              + (f'   -> Y{kids[0]}' if len(kids) else '   (sink)'))
    print(f'    sizes {sizes}   largest={sizes[0]}/{a.p} ({sizes[0]/a.p:.0%})  '
          f'singletons={sum(1 for s in sizes if s == 1)}  balance={bal:.3f}')
    print(f'    information loss = {info:.4f}')
    return info, bal, T


info_c, bal_c, T_c = show(bg, 'closure')

if a.brute and a.p <= 10:
    def parts(c):
        c = list(c)
        if len(c) == 1:
            yield [c]; return
        first = c[0]
        for sm in parts(c[1:]):
            for n, sub in enumerate(sm):
                yield sm[:n] + [[first] + sub] + sm[n+1:]
            yield [[first]] + sm
    allv = []
    for pt in parts(range(a.p)):
        gs = [np.array(x) for x in pt]
        if all(mk.conditions(A, gs)):
            sz = sorted(len(x) for x in pt)
            q = np.array(sz, float); q = q / q.sum()
            bl = float(-(q*np.log(q)).sum()/np.log(len(q))) if len(q) > 1 else 0.
            allv.append((len(pt), bl, gs))
    print(f'\nBRUTE FORCE: {len(allv)} valid partitions exist')
    same = [v for v in allv if v[0] == len(bg)]
    if same:
        bb = max(same, key=lambda v: v[1])
        print(f'  most balanced valid partition AT THE SAME k={len(bg)}:')
        show(bb[2], '  best-balanced')
        print(f'  -> closure balance {bal_c:.3f} vs best available {bb[1]:.3f}')
    mx = max(v[0] for v in allv)
    print(f'  true max k = {mx}   (closure found {len(bg)})')

if not a.noplot:
    try:
        from src.utils.plots import visualize_causal_graph
        out = f'{B}/experiments/abstraction_study/figures/toy_p{a.p}_{a.graph}_d{a.density}_s{a.seed}.png'
        visualize_causal_graph(A_low=A, T_matrix=T_c, save_path=out)
        print(f'\nfigure -> {out}')
    except Exception as e:
        print(f'\n(plot skipped: {type(e).__name__}: {e})')
