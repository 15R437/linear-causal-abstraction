"""The dial, in within-instance rank space -- where the effect actually lives.

A pooled (RAE, loss) scatter mixes between-instance variation (large) with the
within-instance lambda effect (moderate), and the first swamps the second. Ranking each
instance's 9 runs against each other removes the instance offset, leaving only the
question the dial claim asks: within one graph, does raising lambda_oc move the run toward
exact-and-lossy?

The lambda grid is also uneven in effect: with lambda_graph = 100, the graph term
contributes ~230 at a random partition while one_child ~48, so lambda_oc must exceed ~5
before it can influence the objective at all. Values 0, 0.3, 1 and 3 are therefore
effectively one arm, and the panel marks the engaged range.
"""
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
INK, INK2, GRID = '#111111', '#555555', '#dcdcdc'
RAMP = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#0d366b']
mpl.rcParams.update({'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'axes.linewidth': 0.6, 'axes.edgecolor': '#888888',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.01})

d = pd.read_csv(f'{B}/data/frontier2.csv')
h = d[(d.method == 'HailMary') & d.iem.notna() & d.rae.notna()].copy()
h['r_rae'] = h.groupby(['density', 'seed']).rae.rank(method='average')
h['r_iem'] = h.groupby(['density', 'seed']).iem.rank(method='average')
lams = sorted(h.lam_oc.unique())
h['ci'] = h.lam_oc.map({l: i for i, l in enumerate(lams)})

fig, ax = plt.subplots(1, 1, figsize=(3.3, 2.15))
g = h.groupby('lam_oc')[['r_rae', 'r_iem']].mean()
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.grid(True, color=GRID, lw=0.45, zorder=0); ax.set_axisbelow(True)
ax.axvspan(5, 1600, color='#f2f1ee', zorder=1)
ax.plot(g.index.values, g.r_rae.values, '-o', color='#2a78d6', lw=1.6, ms=4.4,
        mec='white', mew=0.7, label=r'$\mathrm{RAE}$ rank', zorder=4)
ax.plot(g.index.values, g.r_iem.values, '-s', color='#eb6834', lw=1.6, ms=4.4,
        mec='white', mew=0.7, label='information-loss rank', zorder=4)
ax.annotate(r'$\lambda_{\mathrm{oc}}$ engaged', (1500, 8.55), fontsize=7.5, color=INK2,
            ha='right')
ax.set_xscale('symlog', linthresh=0.3)
ax.set_xlim(-0.05, 1600); ax.set_ylim(1, 9)
ax.set_xticks([0, 1, 10, 100, 1000])
ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: '%g' % v))
ax.set_yticks([2, 4, 6, 8])
ax.set_xlabel(r'$\lambda_{\mathrm{oc}}$  (one-child weight)', fontsize=9, color=INK, labelpad=1.5)
ax.set_ylabel('mean rank within instance', fontsize=9, color=INK, labelpad=2)
ax.tick_params(colors=INK2, labelsize=8)
ax.legend(frameon=False, fontsize=7.5, loc='center left', labelcolor=INK,
          handlelength=1.6, handletextpad=0.5, borderpad=0.1,
          bbox_to_anchor=(0.0, 0.42))
fig.subplots_adjust(left=0.165, right=0.98, top=0.97, bottom=0.215)
fig.savefig(f'{B}/figures/paper_dial_rank.pdf')
fig.savefig(f'{B}/figures/paper_dial_rank.png', dpi=300)
print('wrote figures/paper_dial_rank.pdf (+png)  [single panel]')
