"""Figure: how the Leontief graph's structure varies with the coefficient threshold.

Four panels, one control variable (the threshold) on a shared log x-axis. Four quantities
on wildly different scales (largest SCC 1-136, density 0.01-0.6, mass 0.6-1.0, IEM
0.25-0.96) so they get small multiples rather than a dual axis.

The unthresholded values (thr=0) cannot sit on a log axis, so each panel marks them as a
dashed reference line instead of a data point.

The vertical band marks the percolation transition at thr ~ 0.014-0.016, where the giant
strongly connected component shatters from 46 nodes to 5 and the economy stops being one
interlocked block.
"""
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
INK, INK2, GRID = '#111111', '#555555', '#dcdcdc'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.6, 'axes.edgecolor': '#888888',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.01,
})

d = pd.read_csv(f'{B}/data/leontief_thresh_sweep.csv')
raw = d[d.thr == 0].iloc[0]          # unthresholded reference
s = d[d.thr > 0].sort_values('thr')
PERC = (0.014, 0.016)

panels = [
    ('max_scc',  'largest SCC (nodes)',        BLUE,   True,  raw.max_scc),
    ('density',  'graph density',              ORANGE, True,  raw.density),
    ('mass_kept','coefficient mass retained',  AQUA,   False, raw.mass_kept),
    ('occ_iem',  "AOC Closure information loss", INK,   False, raw.occ_iem),
]

fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.3), sharex=True)
for ax, (col, lab, c, logy, ref) in zip(axes.ravel(), panels):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, lw=0.45, zorder=0); ax.set_axisbelow(True)
    ax.axvspan(*PERC, color='#f0efec', zorder=1)
    ax.axhline(ref, color=c, lw=0.8, ls=(0, (3, 2)), alpha=0.65, zorder=2)
    ax.plot(s.thr, s[col], '-o', color=c, lw=1.5, ms=3.0, mec='white', mew=0.5, zorder=4)
    ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.set_ylabel(lab, fontsize=8, color=INK, labelpad=2)
    ax.tick_params(colors=INK2, labelsize=7.5)

axes[0, 0].annotate('unthresholded (%d)' % raw.max_scc, (0.10, raw.max_scc),
                    xytext=(0, 3), textcoords='offset points', fontsize=6.5, color=BLUE,
                    ha='right')
axes[0, 1].annotate('unthresholded (%.2f)' % raw.density, (0.10, raw.density),
                    xytext=(0, 3), textcoords='offset points', fontsize=6.5, color=ORANGE,
                    ha='right')
axes[0, 0].annotate('SCC shatters', (0.0165, 22), xytext=(7, 0), textcoords='offset points',
                    fontsize=6.5, color=INK2, va='center')
for ax in axes[1]:
    ax.set_xlabel(r'coefficient threshold $\tau$', fontsize=8.5, color=INK, labelpad=1.5)
fig.subplots_adjust(left=0.115, right=0.985, top=0.965, bottom=0.135, wspace=0.34, hspace=0.16)
fig.savefig(f'{B}/figures/leontief_threshold.pdf')
fig.savefig(f'{B}/figures/leontief_threshold.png', dpi=300)
print('wrote figures/leontief_threshold.pdf (+png)')
print('\nkey rows:')
print(d[d.thr.isin([0, 0.001, 0.01, 0.014, 0.016, 0.02, 0.036, 0.05])]
      [['thr','density','mass_kept','n_scc','max_scc','occ_k','occ_iem']]
      .to_string(index=False, float_format=lambda v: '%.4f' % v))
