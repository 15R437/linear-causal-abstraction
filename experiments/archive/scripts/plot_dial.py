"""The dial figure: every run is a dot, coloured by lambda_one-child.

Supports the claim that lambda_oc traverses an exactness-information trade-off. One dot per
(instance, lambda); pale = lambda 0, dark = lambda 1000. If the claim holds, colour should
grade from bottom-right (informative, inexact) to top-left (exact, lossy).
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
FLOOR = 1e-4
mpl.rcParams.update({'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'axes.linewidth': 0.6, 'axes.edgecolor': '#888888',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.01})

d = pd.read_csv(f'{B}/data/frontier2.csv')
h = d[(d.method == 'HailMary') & d.iem.notna() & d.rae.notna()].copy()
lams = sorted(h.lam_oc.unique())
h['ci'] = h.lam_oc.map({l: i for i, l in enumerate(lams)})
h['rae_p'] = h.rae.clip(lower=FLOOR)
occ = d[(d.method == 'OCC') & d.iem.notna()]

fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.05), sharey=True)
for ax, dn in zip(axes, (0.1, 0.5, 0.9)):
    s = h[h.density == dn]
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, lw=0.45, zorder=0); ax.set_axisbelow(True)
    # AOC Closure reference band for this density
    o = occ[occ.density == dn].iem
    ax.axhline(o.median(), color='#eb6834', lw=1.3, ls=(0, (4, 2)), zorder=2)
    ax.scatter(s.rae_p, s.iem, c=[RAMP[i] for i in s.ci], s=17, lw=0.4,
               edgecolor='white', zorder=4)
    ax.set_xscale('log'); ax.set_xlim(FLOOR * 0.6, 2.2); ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([1e-4, 1e-2, 1.0])
    ax.set_xlabel(r'$\mathrm{RAE}$', fontsize=9, color=INK, labelpad=1.5)
    ax.set_title(r'$d=%.1f$' % dn, fontsize=9, color=INK, pad=3)
    ax.tick_params(colors=INK2, labelsize=8)
axes[0].set_ylabel('information loss', fontsize=9, color=INK, labelpad=2)
axes[0].annotate('AOC Closure', (0.9, occ[occ.density == 0.1].iem.median()),
                 xytext=(0, 4), textcoords='offset points', fontsize=7, color='#eb6834',
                 ha='right')

cax = fig.add_axes([1.005, 0.19, 0.014, 0.66])
cb = fig.colorbar(ScalarMappable(norm=BoundaryNorm(np.arange(len(lams) + 1) - 0.5, len(lams)),
                                 cmap=ListedColormap(RAMP)), cax=cax,
                  ticks=np.arange(len(lams)))
cb.ax.set_yticklabels(['%g' % l for l in lams], fontsize=7, color=INK2)
cb.outline.set_visible(False); cb.ax.tick_params(length=0)
cb.set_label(r'$\lambda_{\mathrm{oc}}$', fontsize=9, color=INK, labelpad=3)
fig.subplots_adjust(left=0.098, right=0.965, top=0.86, bottom=0.235, wspace=0.12)
fig.savefig(f'{B}/figures/paper_dial.pdf'); fig.savefig(f'{B}/figures/paper_dial.png', dpi=300)
print('wrote figures/paper_dial.pdf (+png)')
