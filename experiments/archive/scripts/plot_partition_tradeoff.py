"""The partition trade-off, as parallel coordinates.

Three vertical axes, one line per method. Every axis is "fraction of the ideal", higher is
better, so a TRADE-OFF shows up as a CROSSING: AOC Closure runs high-high-low, Greedy All-In
runs high-low-high, and their lines cross between the second and third axis.

    All-In     1 - allin_pairs / max_pairs      max_pairs = sum over quotient edges x->y of
                                                |X_x|, i.e. every chance to violate
    one-child  1 - oc_excess  / (n_qedges - 1)  n_qedges-1 = the most excess a quotient with
                                                that many edges can have
    info kept  1 - intra_edges                  intra_edges = share of A's nonzero EDGES
                                                swallowed inside a cluster (count, not mass)

The first two are exactly 1 when the condition holds; the third is flipped from the loss so
all three point the same way.

Pooled over density and blockiness; the band behind each line is the interquartile range on
each axis, so the spread is visible without a facet per density. Density is the axis that
moves `info kept` (0.76 -> 0.22 for AOC Closure as density goes 0.1 -> 0.9); that trend is
reported in the printed table rather than drawn.

THE k BUDGET IS LOAD-BEARING: singletons satisfy All-In vacuously AND swallow no edges, so
the finest partition scores 1.0 on two of three axes for free. Every method runs at the same
budget k = AOC Closure's cluster count; median k_used is printed next to each label.
"""
import sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
TAG = sys.argv[1] if len(sys.argv) > 1 else 'partition_quality_n20'
INK, INK2, GRID = '#111111', '#555555', '#dcdcdc'
AX = [('allin_score', 'All-In'), ('oc_score', 'one-child'), ('info_kept', 'info kept')]
COL = {'AOC Closure': '#eb6834', 'OC Closure': '#118a6e', 'Greedy All-In': '#b5651d',
       'GradCAP l=0': '#86b6ef', 'GradCAP l=100': '#2a78d6', 'GT partition': '#6a3d9a'}
MK = {'AOC Closure': 'o', 'OC Closure': 'v', 'Greedy All-In': 'D',
      'GradCAP l=0': 's', 'GradCAP l=100': '^', 'GT partition': '*'}
LBL = {'AOC Closure': 'AOC Closure', 'OC Closure': 'OC Closure',
       'Greedy All-In': 'Greedy All-In', 'GradCAP l=0': r'GradCAP $\lambda_{oc}{=}0$',
       'GradCAP l=100': r'GradCAP $\lambda_{oc}{=}100$', 'GT partition': 'ground truth'}
ORDER = ['AOC Closure', 'OC Closure', 'Greedy All-In', 'GradCAP l=0', 'GradCAP l=100',
         'GT partition']

mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'axes.linewidth': 0.6, 'axes.edgecolor': '#888888',
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

fig, ax = plt.subplots(figsize=(7.4, 3.9))
X = np.arange(len(AX))
for sp in ('top', 'right', 'bottom'):
    ax.spines[sp].set_visible(False)
ax.spines['left'].set_visible(False)

# one vertical axis per metric
for x in X:
    ax.axvline(x, color='#aaaaaa', lw=0.9, zorder=2)
    for yt in (0.0, 0.25, 0.5, 0.75, 1.0):
        ax.plot([x - 0.022, x + 0.022], [yt, yt], color='#aaaaaa', lw=0.8, zorder=2)
for yt in (0.0, 0.25, 0.5, 0.75, 1.0):
    ax.axhline(yt, color=GRID, lw=0.45, zorder=0)
    ax.text(-0.10, yt, f'{yt:.2f}', fontsize=7.5, color=INK2, ha='right', va='center')

ends = {}
for a in arms:
    t = d[d.arm == a]
    med = np.array([t[m].median() for m, _ in AX])
    q1 = np.array([t[m].quantile(.25) for m, _ in AX])
    q3 = np.array([t[m].quantile(.75) for m, _ in AX])
    ax.fill_between(X, q1, q3, color=COL[a], alpha=0.07, lw=0, zorder=3)
    ax.plot(X, med, color=COL[a], lw=1.9, zorder=5, solid_capstyle='round')
    ax.plot(X, med, MK[a], color=COL[a], ms=11 if a == 'GT partition' else 6.5,
            mec='white', mew=1.0, zorder=6, ls='none')
    ends[a] = (med[-1], t.k_used.median())

# Several methods land on exactly 1.0 on the last axis, so the right-hand labels have to be
# pushed apart or they print on top of each other. Nudge down from the top, keeping order.
GAP = 0.058
order = sorted(ends, key=lambda a: -ends[a][0])
ypos, prev = {}, np.inf
for a in order:
    y = min(ends[a][0], prev - GAP)
    ypos[a] = y
    prev = y
for a in order:
    yv, k = ends[a]
    ax.annotate(f'{LBL[a]}  ($k{{=}}{k:.0f}$)', xy=(X[-1] + 0.03, yv),
                xytext=(X[-1] + 0.10, ypos[a]), fontsize=8, color=COL[a],
                va='center', ha='left', annotation_clip=False,
                arrowprops=dict(arrowstyle='-', color=COL[a], lw=0.6,
                                shrinkA=0, shrinkB=2, alpha=0.65))

ax.set_xlim(-0.16, len(AX) - 1 + 0.06)
ax.set_ylim(-0.05, 1.06)
ax.set_xticks(X)
ax.set_xticklabels([l for _, l in AX], fontsize=10, color=INK)
ax.tick_params(axis='x', length=0, pad=6)
ax.set_yticks([])
ax.set_ylabel('fraction of the ideal   (1.0 = best)', fontsize=9, color=INK, labelpad=22)
n_inst = len(d) // max(len(arms), 1)
ax.set_title('crossing lines are trade-offs   '
             r'($p{=}100$, pooled over density and blockiness, $n{\approx}%d$ instances)'
             % n_inst, fontsize=9, color=INK, pad=8)
fig.tight_layout(rect=[0, 0, 0.74, 1])
fig.savefig(f'{B}/figures/partition_tradeoff.pdf')
fig.savefig(f'{B}/figures/partition_tradeoff.png', dpi=300)
plt.close(fig)
print('wrote figures/partition_tradeoff.png (+pdf)')

pd.set_option('display.width', 220)
print('\n=== the six numbers the figure draws (median) ===')
print(d.groupby('arm')[['allin_score', 'oc_score', 'info_kept', 'k_used']]
      .median().reindex(arms).to_string(float_format=lambda v: f'{v:.3f}'))
print('\n=== info kept by density (pooled away in the figure) ===')
print(d.pivot_table(index='arm', columns='density', values='info_kept', aggfunc='median')
      .reindex(arms).to_string(float_format=lambda v: f'{v:.3f}'))
