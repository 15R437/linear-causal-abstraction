"""Companion to paper_alpha_main: compression achieved, by density and method.

Information loss alone is uninterpretable, because on dense graphs intra-cluster mass
tracks 1/k -- a method can always lower IEM by using more clusters. This plots the
compression actually achieved (clusters used / low-level variables) on the same density
axis, so the two figures read together.

Lines are medians over the 20 instances per density; whiskers are interquartile ranges,
dodged horizontally so they do not composite into phantom extra series. The whiskers show
instance-to-instance spread, NOT uncertainty on the median: the four methods run on the
same instances at the same budget, so the inferential quantity is the paired difference,
which is reported in the caption and is significant at every density despite the overlap.
The cluster budget is k = |pi_0|, AOC Closure's own count, so the AOC Closure line IS the
budget: it always returns exactly its own partition. Every other method is working under
that ceiling, and a line below it means the method left clusters empty.
"""
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
INK, INK2, GRID = '#111111', '#555555', '#dcdcdc'
P = 100.0
SERIES = [
    ('OCC',       None,   'AOC Closure',                      '#eb6834', 's'),
    ('HailMary',  0.0,    r'HailMary $\lambda_{\rm oc}{=}0$',    '#86b6ef', 'o'),
    ('HailMary',  1000.0, r'HailMary $\lambda_{\rm oc}{=}1000$', '#0d366b', '^'),
    ('UCA',       None,   'UCA',                              '#1baf7a', 'v'),
]
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.6, 'axes.edgecolor': '#888888',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.01,
})

d = pd.read_csv(f'{B}/data/frontier2.csv')
d = d[d.iem.notna()].copy()
d['compress'] = d.k_used / P
dens = sorted(d.density.unique())

fig, ax = plt.subplots(figsize=(4.0, 2.5))
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.grid(True, color=GRID, lw=0.45, zorder=0); ax.set_axisbelow(True)

# IQR as DODGED whiskers, not translucent fills. Four alpha-composited bands produce up to
# 2^4 distinct tints where they overlap, so a reader counts more shades than there are
# series. Dodging removes the compositing entirely and keeps every interval readable.
OFF = np.linspace(-0.024, 0.024, len(SERIES))
for (meth, lam, lab, col, mk), dx in zip(SERIES, OFF):
    s = d[d.method == meth] if lam is None else d[(d.method == meth) & (d.lam_oc == lam)]
    if not len(s):
        continue
    g = s.groupby('density').compress
    med, q1, q3 = g.median(), g.quantile(.25), g.quantile(.75)
    x = med.index.values
    ax.errorbar(x + dx, med.values, yerr=[med.values - q1.values, q3.values - med.values],
                fmt='none', ecolor=col, elinewidth=0.9, capsize=1.6, capthick=0.9,
                alpha=0.85, zorder=3)
    ax.plot(x, med.values, '-', marker=mk, color=col, lw=1.5, ms=4.0,
            mec='white', mew=0.6, zorder=5)

ax.set_xlim(0.05, 0.95); ax.set_ylim(0, 0.64)
ax.set_xticks(dens)
ax.set_xlabel('graph density', fontsize=9, color=INK, labelpad=1.5)
ax.set_ylabel('compression  $k_{\\rm used}/p$', fontsize=9, color=INK, labelpad=2)
ax.tick_params(colors=INK2, labelsize=8)

handles = [Line2D([], [], color=c, lw=1.5, marker=m, ms=4.0, mec='white', mew=0.6, label=l)
           for _, _, l, c, m in SERIES]
ax.legend(handles=handles, frameon=False, fontsize=7.5, labelcolor=INK, loc='upper right',
          handlelength=1.7, handletextpad=0.5, borderpad=0.2, labelspacing=0.35)
fig.subplots_adjust(left=0.165, right=0.98, top=0.965, bottom=0.185)
fig.savefig(f'{B}/figures/paper_compression.pdf')
fig.savefig(f'{B}/figures/paper_compression.png', dpi=300)
print('wrote figures/paper_compression.pdf (+png)\n')
t = d.groupby(['density', 'method', 'lam_oc'], dropna=False).agg(
    n=('compress', 'size'), k_budget=('k', 'median'), k_used=('k_used', 'median'),
    compress=('compress', 'median'), iem=('iem', 'median')).round(3)
print(t.to_string())
