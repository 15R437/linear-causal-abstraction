"""Partition learners on the three structural metrics, faceted by density.

All three metrics are put on a common [0, 1] "fraction of the ideal" scale where HIGHER IS
BETTER, which is what lets them share one axis:

    All-In     1 - allin_pairs / max_pairs      max_pairs = sum over quotient edges x->y
                                                of |X_x|, i.e. every chance to violate
    one-child  1 - oc_excess  / (n_qedges - 1)  n_qedges-1 = the most excess a quotient with
                                                that many edges can have
    info kept  1 - intra_edges                  intra_edges = share of A's nonzero edges
                                                swallowed inside a cluster (unweighted count)

The first two are 1 exactly when the condition holds. The third is flipped from the loss so
it points the same way as the other two; it is the EDGE COUNT version, not the mass-weighted
one.

Density is the axis of variation: it beats blockiness on the interaction statistic for every
metric (analyse_partition_axis.py), and `info kept` in particular moves a long way along it.

THE k BUDGET IS LOAD-BEARING. Singletons satisfy All-In vacuously AND swallow no edges, so
the finest partition trivially scores 1.0 on two of the three bars. Every method runs at the
same budget K = AOC Closure's cluster count, shown per panel; `k_used` is printed below
because a method may spend fewer than it was given.
"""
import sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
TAG = sys.argv[1] if len(sys.argv) > 1 else 'partition_quality_n20'
INK, INK2, GRID = '#111111', '#555555', '#dcdcdc'
METRICS = [('allin_score', 'All-In', '#2a78d6'),
           ('oc_score', 'one-child', '#eb6834'),
           ('info_kept', 'info kept', '#118a6e')]
ORDER = ['AOC Closure', 'OC Closure', 'Greedy All-In', 'GradCAP l=0', 'GradCAP l=100',
         'GT partition']
SHORT = {'AOC Closure': 'AOC', 'OC Closure': 'OC', 'Greedy All-In': 'Greedy',
         'GradCAP l=0': r'GC $\lambda$0', 'GradCAP l=100': r'GC $\lambda$100',
         'GT partition': 'GT'}

mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'axes.linewidth': 0.6, 'axes.edgecolor': '#888888',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 0,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

d = pd.read_csv(f'{B}/data/{TAG}.csv')
d['arm'] = np.where(d.method == 'GradCAP',
                    'GradCAP l=' + d.lam_oc.fillna(0).astype(int).astype(str), d.method)
d = d[d.arm.isin(ORDER)].copy()
d['allin_score'] = 1.0 - d.allin_pairs / d.max_pairs.clip(lower=1)
d['oc_score'] = 1.0 - d.oc_excess / (d.n_qedges - 1).clip(lower=1)
d.loc[d.max_pairs <= 0, 'allin_score'] = 1.0
d.loc[d.n_qedges <= 1, 'oc_score'] = 1.0
d['info_kept'] = 1.0 - d.intra_edges
arms = [a for a in ORDER if a in set(d.arm)]
dens = sorted(d.density.unique())

nrow = 2 if len(dens) > 2 else 1
ncol = int(np.ceil(len(dens) / nrow))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.35 * ncol, 2.55 * nrow), squeeze=False,
                         sharey=True)
W = 0.26
xs = np.arange(len(arms))

for pi, dn in enumerate(dens):
    ax = axes[pi // ncol][pi % ncol]
    s = d[d.density == dn]
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(True, axis='y', color=GRID, lw=0.45, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8)
    for mi, (m, lab, col) in enumerate(METRICS):
        med = np.array([s[s.arm == a][m].median() for a in arms], float)
        q1 = np.array([s[s.arm == a][m].quantile(.25) for a in arms], float)
        q3 = np.array([s[s.arm == a][m].quantile(.75) for a in arms], float)
        off = (mi - 1) * W
        ax.bar(xs + off, med, width=W * 0.92, color=col, edgecolor='white', lw=0.6, zorder=3)
        ax.errorbar(xs + off, med, yerr=[np.clip(med - q1, 0, None),
                                         np.clip(q3 - med, 0, None)],
                    fmt='none', ecolor=INK2, lw=0.8, capsize=1.6, zorder=5)
    ax.set_ylim(0, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[a] for a in arms], fontsize=8, color=INK)
    ax.set_xlim(-0.55, len(arms) - 0.45)
    kmed = s.budget_k.median()
    nseed = s.seed.nunique()
    ax.set_title(r'density $=%g$   ($k=%d$, $n=%d$)' % (dn, kmed, nseed),
                 fontsize=9, color=INK, pad=4)
    if pi % ncol == 0:
        ax.set_ylabel('fraction of ideal', fontsize=9, color=INK, labelpad=3)

for pi in range(len(dens), nrow * ncol):
    axes[pi // ncol][pi % ncol].axis('off')

fig.legend(handles=[Patch(facecolor=c, label=l) for _, l, c in METRICS],
           loc='lower center', ncol=3, frameon=False, fontsize=8.5, labelcolor=INK,
           handlelength=1.5, handletextpad=0.6, columnspacing=2.0,
           bbox_to_anchor=(0.5, -0.035))
fig.tight_layout(rect=[0, 0.06, 1, 0.99], w_pad=1.4, h_pad=1.5)
fig.savefig(f'{B}/figures/partition_bars.pdf')
fig.savefig(f'{B}/figures/partition_bars.png', dpi=300)
plt.close(fig)
print('wrote figures/partition_bars.png (+pdf)')

pd.set_option('display.width', 220)
print('\n=== median per method, all densities pooled (1.0 = ideal on all three) ===')
print(d.groupby('arm')[['allin_score', 'oc_score', 'info_kept', 'k_used', 'budget_k']]
      .median().reindex(arms).to_string(float_format=lambda v: f'{v:.3f}'))
print('\n=== info kept by density (the metric that actually moves) ===')
print(d.pivot_table(index='arm', columns='density', values='info_kept', aggfunc='median')
      .reindex(arms).to_string(float_format=lambda v: f'{v:.3f}'))
print('\n=== k_used by density (budget k in the column header) ===')
kb = d.groupby('density').budget_k.median()
t = d.pivot_table(index='arm', columns='density', values='k_used', aggfunc='median').reindex(arms)
t.columns = [f'd={c} (k={kb[c]:.0f})' for c in t.columns]
print(t.to_string(float_format=lambda v: f'{v:.0f}'))
