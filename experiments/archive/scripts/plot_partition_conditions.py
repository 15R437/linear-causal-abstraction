"""Partition learners compared on the structural conditions. Two fine-grained scatters.

  (i)  partition_cond_quadrant   All-In satisfaction  vs  one-child satisfaction
  (ii) partition_cond_intra      All-In satisfaction  vs  intra-cluster EDGE fraction

Both axes in (i) are SATISFACTION fractions, i.e. 1 - (violations / chances to violate), so
1.0 means the condition holds exactly and the valid corner is TOP-RIGHT. They are the
violation fractions flipped; the flip is only so both plots point the same way.

    allin_score = 1 - allin_pairs / max_pairs
                  max_pairs = sum over quotient edges x->y of |X_x|, every chance to violate
    oc_score    = 1 - oc_excess  / (n_qedges - 1)
                  n_qedges - 1 = the most excess a quotient with that many edges can have

In (ii) the y axis is an unweighted EDGE COUNT fraction -- how many of A's nonzero edges are
swallowed inside a cluster -- not the mass-weighted version. Lower is better there, so the
good corner is BOTTOM-RIGHT.

The k budget is load-bearing: singletons satisfy All-In vacuously AND swallow no edges, so
the finest partition trivially optimises both plots. Every method runs at the same budget
K = AOC Closure's cluster count; `k_used` is printed because a method may spend fewer.
"""
import sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
TAG = sys.argv[1] if len(sys.argv) > 1 else 'partition_quality'
INK, INK2, GRID = '#111111', '#555555', '#dcdcdc'
COL = {'AOC Closure': '#eb6834', 'OC Closure': '#118a6e', 'Greedy All-In': '#b5651d',
       'GradCAP l=0': '#86b6ef', 'GradCAP l=100': '#2a78d6', 'GT partition': '#6a3d9a'}
MK = {'AOC Closure': 'o', 'OC Closure': 'v', 'Greedy All-In': 'D',
      'GradCAP l=0': 's', 'GradCAP l=100': '^', 'GT partition': '*'}
ALL = ['AOC Closure', 'OC Closure', 'Greedy All-In', 'GradCAP l=0', 'GradCAP l=100',
       'GT partition']

mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'axes.linewidth': 0.6, 'axes.edgecolor': '#888888',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

d = pd.read_csv(f'{B}/data/{TAG}.csv')
d['arm'] = np.where(d.method == 'GradCAP',
                    'GradCAP l=' + d.lam_oc.fillna(0).astype(int).astype(str), d.method)
d = d[d.arm.isin(ALL)].copy()
d['allin_score'] = 1.0 - d.allin_pairs / d.max_pairs.clip(lower=1)
d['oc_score'] = 1.0 - d.oc_excess / (d.n_qedges - 1).clip(lower=1)
d.loc[d.max_pairs <= 0, 'allin_score'] = 1.0
d.loc[d.n_qedges <= 1, 'oc_score'] = 1.0
arms = [a for a in ALL if a in set(d.arm)]
rng = np.random.default_rng(0)


def style(ax):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, lw=0.45, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8)


def cloud(ax, s, xcol, ycol, jit=0.008):
    for a in arms:
        t = s[s.arm == a]
        if not len(t):
            continue
        ax.scatter(t[xcol] + rng.normal(0, jit, len(t)),
                   t[ycol] + rng.normal(0, jit, len(t)),
                   s=5, color=COL[a], alpha=0.13, lw=0, zorder=3)
    DODGE = {'AOC Closure': (0, 0), 'OC Closure': (0, 0), 'Greedy All-In': (0.012, -0.012),
             'GradCAP l=0': (-0.030, 0.012), 'GradCAP l=100': (0, 0),
             'GT partition': (0.012, 0.022)}
    for a in arms:
        t = s[s.arm == a]
        if not len(t):
            continue
        x, y = t[xcol].median(), t[ycol].median()
        dx, dy = DODGE.get(a, (0, 0))
        yspan = 1.0 if ycol == 'oc_score' else 1.0
        ax.errorbar(x, y, xerr=[[x - t[xcol].quantile(.25)], [t[xcol].quantile(.75) - x]],
                    yerr=[[y - t[ycol].quantile(.25)], [t[ycol].quantile(.75) - y]],
                    color=COL[a], lw=1.0, capsize=0, alpha=0.8, zorder=5)
        ax.plot(x + dx, y + dy, MK[a], color=COL[a], ms=13 if a == 'GT partition' else 9,
                mec='white', mew=1.1, zorder=6)


HAND = [Line2D([], [], color=COL[a], marker=MK[a], ms=6 if a != 'GT partition' else 9,
               lw=0, mec='white', mew=0.8, label=a) for a in ALL]


def figure(ycol, ylab, fname, title, good, facet=('density', 'density')):
    """Pooled panel + one panel per level of the faceting axis."""
    axis, xlab = facet
    lv = sorted(d[axis].unique())
    fig, axes = plt.subplots(1, len(lv) + 1, figsize=(2.05 * (len(lv) + 1) + 0.4, 2.5),
                             sharex=True, sharey=True, squeeze=False)
    for ci, panel in enumerate([None] + lv):
        ax = axes[0][ci]; style(ax)
        ax.axvline(1.0, color='#bbbbbb', lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.axhline(1.0 if ycol == 'oc_score' else 0.0, color='#bbbbbb', lw=0.8,
                   ls=(0, (3, 3)), zorder=1)
        cloud(ax, d if panel is None else d[d[axis] == panel], 'allin_score', ycol)
        ax.set_title('all instances' if panel is None else f'{xlab} = {panel:g}',
                     fontsize=9, color=INK, pad=3)
        ax.set_xlabel('All-In satisfied\n$1-$ violations / chances', fontsize=8, color=INK,
                      labelpad=1.5)
        if ci == 0:
            ax.set_ylabel(ylab, fontsize=8.5, color=INK, labelpad=2)
            ax.text(0.03, 0.97 if good == 'top' else 0.03, 'best', fontsize=7,
                    color='#2f6f4e', transform=ax.transAxes,
                    va='top' if good == 'top' else 'bottom')
    axes[0][0].set_xlim(-0.06, 1.08)
    fig.legend(handles=HAND, loc='lower center', ncol=6, frameon=False, fontsize=7.5,
               labelcolor=INK, handlelength=1.6, handletextpad=0.5, columnspacing=1.2,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(title, fontsize=10, color=INK, y=1.02)
    fig.tight_layout(rect=[0, 0.15, 1, 0.97], w_pad=1.1)
    fig.savefig(f'{B}/figures/{fname}.pdf'); fig.savefig(f'{B}/figures/{fname}.png', dpi=300)
    plt.close(fig); print(f'wrote figures/{fname}.png (+pdf)')


# density is the axis of variation: it beats blockiness on the interaction statistic for
# EVERY metric (analyse_partition_axis.py). Blockiness companions are kept to show the null.
figure('oc_score', 'one-child satisfied\n$1-$ excess / max', 'partition_cond_quadrant',
       'the two structural conditions against each other  (top-right = valid)', 'top')
figure('intra_edges', 'intra-cluster edge fraction', 'partition_cond_intra',
       'All-In against information thrown away  (bottom-right = best)', 'bottom')
figure('oc_score', 'one-child satisfied\n$1-$ excess / max', 'partition_cond_quadrant_blk',
       'the two structural conditions  --  blockiness barely moves them', 'top',
       facet=('blk', 'blockiness'))
figure('intra_edges', 'intra-cluster edge fraction', 'partition_cond_intra_blk',
       'All-In against information thrown away  --  blockiness barely moves them', 'bottom',
       facet=('blk', 'blockiness'))

pd.set_option('display.width', 230)
cols = ['allin_score', 'oc_score', 'intra_edges', 'intra', 'allin_nodes', 'allin_pairs',
        'oc_excess', 'k_used', 'budget_k']
cols += [c for c in ('rae', 'marg', 'norm_Ahigh') if c in d.columns]
print('\n=== median per method (all cells pooled) ===')
print(d.groupby('arm')[cols].median().reindex(arms).to_string(float_format=lambda v: f'{v:.3g}'))
print('\n=== fraction of instances satisfying each condition outright ===')
t = d.groupby('arm')[['allin_ok', 'oc_ok']].mean().reindex(arms)
t['both'] = d.groupby('arm').apply(
    lambda g: float(((g.allin_ok == 1) & (g.oc_ok == 1)).mean()), include_groups=False).reindex(arms)
print(t.to_string(float_format=lambda v: f'{v:.3f}'))
print('\n=== intra: EDGE COUNT fraction vs MASS fraction (they come apart) ===')
print(d.groupby('arm')[['intra_edges', 'intra']].median().reindex(arms)
      .assign(diff=lambda x: x.intra_edges - x.intra).to_string(float_format=lambda v: f'{v:.3f}'))
