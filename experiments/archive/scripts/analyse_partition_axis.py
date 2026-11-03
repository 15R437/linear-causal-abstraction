"""Which axis of variation -- density or blockiness -- better separates the partition learners?

An axis is useful for comparing methods if (a) it MOVES the metric, and more importantly
(b) it changes the SEPARATION between methods or their ORDERING. An axis that shifts every
method by the same amount tells you about the generator, not about the methods.

Reports per metric, for each candidate axis:
  move      var of axis-level means / total var of cell means      (does the axis move it)
  sep_lo/hi between-method spread at the lowest / highest axis level
  inter     var of (method effect) across axis levels / total var  (interaction: does the
            axis change the RANKING or the GAPS -- the thing that matters here)
  flips     # method pairs whose median order reverses between the axis extremes
"""
import sys
import numpy as np, pandas as pd

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
TAG = sys.argv[1] if len(sys.argv) > 1 else 'partition_quality'
d = pd.read_csv(f'{B}/data/{TAG}.csv')
d['arm'] = np.where(d.method == 'GradCAP',
                    'GradCAP $\\lambda$=' + d.lam_oc.fillna(0).astype(int).astype(str),
                    d.method)
LEARNERS = ['AOC Closure', 'OC Closure', 'Greedy All-In', 'GradCAP $\\lambda$=0',
            'GradCAP $\\lambda$=100']
METRICS = [('allin_nodes', 'All-In violating nodes'), ('allin_pairs', 'All-In violation pairs'),
           ('allin_frac', 'All-In violations / opportunities'),
           ('oc_excess', 'one-child excess edges'), ('intra', 'intra-cluster edge mass'),
           ('k_used', 'clusters used')]

print(f'n rows {len(d)}, seeds {d.seed.nunique()}, '
      f'grid {sorted(d.blk.unique())} x {sorted(d.density.unique())}\n')


def analyse(metric):
    s = d[d.arm.isin(LEARNERS)]
    cell = s.groupby(['arm', 'blk', 'density'])[metric].median().reset_index()
    out = {}
    for axis, other in (('blk', 'density'), ('density', 'blk')):
        # average over the OTHER axis so the two are compared on equal terms
        ca = cell.groupby(['arm', axis])[metric].mean().reset_index()
        piv = ca.pivot(index=axis, columns='arm', values=metric)[LEARNERS]
        grand = piv.values.mean()
        tot = ((piv.values - grand) ** 2).mean()
        axis_means = piv.mean(axis=1)
        move = ((axis_means - grand) ** 2).mean() / tot if tot > 0 else 0.0
        # interaction = residual after removing additive axis and method main effects
        meth_means = piv.mean(axis=0)
        add = axis_means.values[:, None] + meth_means.values[None, :] - grand
        inter = ((piv.values - add) ** 2).mean() / tot if tot > 0 else 0.0
        lo, hi = piv.iloc[0], piv.iloc[-1]
        flips = sum(1 for i in range(len(LEARNERS)) for j in range(i + 1, len(LEARNERS))
                    if np.sign(lo.iloc[i] - lo.iloc[j]) != np.sign(hi.iloc[i] - hi.iloc[j]))
        out[axis] = dict(move=move, inter=inter, sep_lo=lo.max() - lo.min(),
                         sep_hi=hi.max() - hi.min(), flips=flips, piv=piv)
    return out


rows = []
for m, label in METRICS:
    a = analyse(m)
    for axis in ('blk', 'density'):
        r = a[axis]
        rows.append(dict(metric=label, axis={'blk': 'blockiness', 'density': 'density'}[axis],
                         move=r['move'], interaction=r['inter'], sep_lo=r['sep_lo'],
                         sep_hi=r['sep_hi'], flips=r['flips']))
res = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print('=== does the axis move the metric, and does it change the method comparison? ===')
print(res.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

print('\n=== verdict per metric (higher interaction = more informative axis) ===')
for m, label in METRICS:
    sub = res[res.metric == label].set_index('axis')
    win = sub.interaction.idxmax()
    print(f'  {label:36s} interaction  blockiness {sub.loc["blockiness","interaction"]:.3f}  '
          f'density {sub.loc["density","interaction"]:.3f}   -> {win}')

print('\n=== method medians along each axis (averaged over the other) ===')
for m, label in METRICS[:5]:
    a = analyse(m)
    for axis in ('blk', 'density'):
        print(f'\n{label}  vs  {axis}')
        print(a[axis]['piv'].to_string(float_format=lambda v: f'{v:.3f}'))
