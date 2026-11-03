"""Partition learners compared as partition learners. Three figures.

  1  partition_axis      justifies the axis of variation: every metric against BOTH
                         blockiness and density, so the choice is shown, not asserted.
  2  partition_tradeoff  the (violations, intra) trade-off the question asks for, with
                         one-child excess carried as marker size -- without it Greedy
                         All-In reads as a free win at (0, 0.00) when it is running
                         one-child excess up to ~70.
  3  partition_translate low violations -> good abstraction? Violations against downstream
                         reconstruction and against marginalised-node count.
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
LEARNERS = ['AOC Closure', 'OC Closure', 'Greedy All-In', 'GradCAP l=0', 'GradCAP l=100']
ALL = LEARNERS + ['GT partition']

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.6, 'axes.edgecolor': '#888888',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

d = pd.read_csv(f'{B}/data/{TAG}.csv')
d['arm'] = np.where(d.method == 'GradCAP',
                    'GradCAP l=' + d.lam_oc.fillna(0).astype(int).astype(str), d.method)
d = d[d.arm.isin(ALL)]
blks, dens = sorted(d.blk.unique()), sorted(d.density.unique())


def style(ax):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, lw=0.45, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8)


def legend(fig, arms=ALL, y=-0.02, ncol=6, extra=None):
    h = [Line2D([], [], color=COL[a], marker=MK[a], ms=5 if a != 'GT partition' else 8,
                lw=1.4, mec='white', mew=0.6, label=a) for a in arms]
    if extra:
        h += extra
    fig.legend(handles=h, loc='lower center', ncol=ncol, frameon=False, fontsize=7.5,
               labelcolor=INK, handlelength=1.7, handletextpad=0.5, columnspacing=1.2,
               bbox_to_anchor=(0.5, y))


# ============================================================ 1. axis of variation
MET = [('allin_nodes', 'All-In violating nodes'), ('oc_excess', 'one-child excess'),
       ('intra', 'intra-cluster edge mass'), ('k_used', 'clusters used')]
fig, axes = plt.subplots(2, len(MET), figsize=(8.6, 4.2), squeeze=False)
for ri, (axis, xlab) in enumerate((('blk', 'blockiness'), ('density', 'density'))):
    for ci, (m, lab) in enumerate(MET):
        ax = axes[ri][ci]; style(ax)
        for a in ALL:
            s = d[d.arm == a]
            g = s.groupby([axis, 'seed'])[m].median().groupby(axis)
            med, q1, q3 = g.median(), g.quantile(.25), g.quantile(.75)
            ax.fill_between(med.index, q1, q3, color=COL[a], alpha=0.10, lw=0, zorder=2)
            ax.plot(med.index, med.values, color=COL[a], lw=1.4, marker=MK[a],
                    ms=5 if a != 'GT partition' else 7.5, mec='white', mew=0.6, zorder=4)
        ax.set_xlabel(xlab, fontsize=9, color=INK, labelpad=1.5)
        if ri == 0:
            ax.set_title(lab, fontsize=9, color=INK, pad=4)
        if ci == 0:
            ax.set_ylabel('median over instances', fontsize=8.5, color=INK, labelpad=2)
legend(fig, y=-0.005)
fig.suptitle('the same metrics against each candidate axis (other axis averaged out)',
             fontsize=9.5, color=INK, y=1.005)
fig.tight_layout(rect=[0, 0.09, 1, 0.985], w_pad=1.4, h_pad=1.5)
fig.savefig(f'{B}/figures/partition_axis.pdf'); fig.savefig(f'{B}/figures/partition_axis.png', dpi=300)
plt.close(fig); print('wrote figures/partition_axis.png (+pdf)')


# ============================================================ 2. the trade-off scatter
def tradeoff(axis, levels, fname, xlab):
    fig, axes = plt.subplots(1, len(levels), figsize=(8.8, 2.35), squeeze=False)
    smax = max(d.oc_excess.max(), 1)
    for ci, lv in enumerate(levels):
        ax = axes[0][ci]; style(ax)
        s = d[d[axis] == lv]
        for a in ALL:
            t = s[s.arm == a]
            ax.scatter(t.allin_nodes, t.intra, s=7, color=COL[a], alpha=0.16,
                       lw=0, zorder=3)
        for a in ALL:
            t = s[s.arm == a]
            if not len(t):
                continue
            x, y, oc = t.allin_nodes.median(), t.intra.median(), t.oc_excess.median()
            ax.errorbar(x, y, xerr=[[x - t.allin_nodes.quantile(.25)],
                                    [t.allin_nodes.quantile(.75) - x]],
                        yerr=[[y - t.intra.quantile(.25)], [t.intra.quantile(.75) - y]],
                        color=COL[a], lw=1.0, capsize=0, alpha=0.75, zorder=5)
            ax.scatter([x], [y], s=40 + 320 * oc / smax, color=COL[a], marker=MK[a],
                       edgecolor='white', lw=0.9, zorder=6)
        ax.set_xlabel('All-In violating nodes', fontsize=8.5, color=INK, labelpad=1.5)
        ax.set_title(f'{xlab} = {lv:g}', fontsize=9, color=INK, pad=3)
        ax.set_xlim(-4, 104)
        if ci == 0:
            ax.set_ylabel('intra-cluster edge mass', fontsize=8.5, color=INK, labelpad=2)
    extra = [Line2D([], [], color='#999999', marker='o', ms=np.sqrt(40) * .7, lw=0,
                    mec='white', label='marker area $\\propto$ one-child excess')]
    legend(fig, y=-0.04, ncol=4, extra=extra)
    fig.suptitle('partition quality: the two conditions trade off, and one-child is the '
                 'hidden third axis', fontsize=9.5, color=INK, y=1.03)
    fig.tight_layout(rect=[0, 0.20, 1, 0.98], w_pad=1.3)
    fig.savefig(f'{B}/figures/{fname}.pdf'); fig.savefig(f'{B}/figures/{fname}.png', dpi=300)
    plt.close(fig); print(f'wrote figures/{fname}.png (+pdf)')


tradeoff('blk', blks, 'partition_tradeoff', 'blockiness')
tradeoff('density', dens, 'partition_tradeoff_density', 'density')


# ============================================================ 3. does it translate?
if 'recon' in d.columns:
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.5))
    for ax in axes:
        style(ax)
    for a in ALL:
        t = d[d.arm == a]
        axes[0].scatter(t.allin_nodes, t.recon, s=6, color=COL[a], alpha=0.14, lw=0)
        axes[1].scatter(t.allin_nodes, t.marg, s=6, color=COL[a], alpha=0.14, lw=0)
        axes[2].scatter(t.intra, t.recon, s=6, color=COL[a], alpha=0.14, lw=0)
    for a in ALL:
        t = d[d.arm == a]
        ms = 9 if a != 'GT partition' else 13
        axes[0].plot(t.allin_nodes.median(), t.recon.median(), MK[a], color=COL[a],
                     ms=ms, mec='white', mew=1.0, zorder=6)
        axes[1].plot(t.allin_nodes.median(), t.marg.median(), MK[a], color=COL[a],
                     ms=ms, mec='white', mew=1.0, zorder=6)
        axes[2].plot(t.intra.median(), t.recon.median(), MK[a], color=COL[a],
                     ms=ms, mec='white', mew=1.0, zorder=6)
    for ax, xl, yl in ((axes[0], 'All-In violating nodes', 'reconstruction error'),
                       (axes[1], 'All-In violating nodes', 'marginalised nodes'),
                       (axes[2], 'intra-cluster edge mass', 'reconstruction error')):
        ax.set_xlabel(xl, fontsize=8.5, color=INK, labelpad=1.5)
        ax.set_ylabel(yl, fontsize=8.5, color=INK, labelpad=2)
    axes[0].set_ylim(-0.05, 2.2); axes[2].set_ylim(-0.05, 2.2)
    legend(fig, y=-0.03, ncol=6)
    fig.suptitle('low violation count does not translate into a good abstraction',
                 fontsize=9.5, color=INK, y=1.02)
    fig.tight_layout(rect=[0, 0.16, 1, 0.97], w_pad=1.5)
    fig.savefig(f'{B}/figures/partition_translate.pdf')
    fig.savefig(f'{B}/figures/partition_translate.png', dpi=300)
    plt.close(fig); print('wrote figures/partition_translate.png (+pdf)')

# ---- numbers
pd.set_option('display.width', 220)
print('\n=== per-method summary (median over all cells) ===')
cols = ['allin_nodes', 'allin_pairs', 'allin_frac', 'oc_excess', 'intra', 'k_used', 'budget_k']
if 'recon' in d.columns:
    cols += ['recon', 'rae', 'marg']
print(d.groupby('arm')[cols].median().reindex(ALL).to_string(float_format=lambda v: f'{v:.3g}'))
print('\n=== fraction of instances fully satisfying each condition ===')
print(d.groupby('arm')[['allin_ok', 'oc_ok']].mean().reindex(ALL)
      .assign(both=lambda x: d.groupby('arm').apply(
          lambda g: float((g.allin_ok * g.oc_ok).mean())).reindex(ALL))
      .to_string(float_format=lambda v: f'{v:.2f}'))
