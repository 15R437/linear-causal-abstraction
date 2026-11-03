"""Partition learners in the tk_sweep format: RAE and ReconErr against p and against k.

Built from `eigen_bench.csv`, which already carries the right grid -- its five cells give a
p axis at FIXED p/k = 10 ((40,4), (100,10), (200,20)) and a k axis at FIXED p = 100
((100,4), (100,10), (100,20)), sharing the (100,10) cell exactly as tk_sweep's two panels do.
Three points per axis rather than six or seven, so read the trend, not the wiggle.

NOT the alpha study's data: that fixes p=100, k=10 and sweeps DENSITY, so it cannot make this
figure at all.

ONE FITTER (`--fitter`, default eigen) so the lines differ only by partition. `--regime`
picks acyclic or cyclic. The GT partition is drawn as the reference every learner is trying
to reach; `random` is the control.

SETTINGS differ from the alpha study and the difference matters: blockiness 1 here against
0.8 there, so A is exactly T A_high T^+ and ReconErr = 0 is attainable -- which is what makes
the GT partition's line meaningful as a floor rather than as another curve. Density 0.5,
n=20 seeds, median with an interquartile band (the GT reference line is drawn without
a band, so the four learned arms stay legible).
"""
import sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
_a = [a for a in sys.argv[1:] if not a.startswith('-')]
TAG = _a[0] if _a else 'eigen_bench'
_g = lambda f, d: ([x.split('=', 1)[1] for x in sys.argv[1:] if x.startswith(f)] or [d])[0]
FITTER, REGIME = _g('--fitter=', 'eigen'), _g('--regime=', 'acyclic')
TITLE = _g('--title=', r'Optimizing $\mathbf{T}$')
RATIO, P_FIXED = 10, 100
INK, INK2, GRID = '#111111', '#555555', '#dcdcdc'
ORDER = ['GT', 'Greedy All-In', 'OC Closure', 'AOC Closure', 'random']
COL = {'GT': '#444444', 'Greedy All-In': '#2ca02c', 'OC Closure': '#7a7a7a',
       'AOC Closure': '#eb6834', 'random': '#9467bd'}
MK = {'GT': 'o', 'Greedy All-In': 'D', 'OC Closure': 's', 'AOC Closure': '*', 'random': 'v'}
LS = {'GT': (0, (4, 2)), 'Greedy All-In': '-', 'OC Closure': '--',
      'AOC Closure': '-', 'random': ':'}
FLOOR = 1e-18
SPEC = {'rae': ('RAE', 'log'), 'recon': ('ReconErr', 'linear'),
        'edge_recall': ('edge recall', 'linear'), 'norm_Ahigh': (r'$\|\tilde{A}\|_F$', 'log')}
MET = (_g('--metrics=', 'rae,recon')).split(',')

mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'axes.linewidth': 0.6, 'axes.edgecolor': '#888888',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.01,
    'pdf.fonttype': 42, 'ps.fonttype': 42})

d = pd.read_csv(f'{B}/data/{TAG}.csv')
d = d[(d.fitter == FITTER) & (d.regime == REGIME) & d.rae.notna()].copy()
NSEED = int(d.groupby(['p', 'k', 'partition']).size().min())
arms = [a for a in ORDER if a in set(d.partition)]
SWEEPS = [('p', d[d.k == d.p // RATIO], r'$k = p/%d$' % RATIO),
          ('k', d[d.p == P_FIXED], r'$p = %d$' % P_FIXED)]


def style(ax):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, lw=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8)


def draw(ax, sub, xcol, col, scale):
    x = np.array(sorted(sub[xcol].unique()), dtype=float)
    for a in arms:
        s = sub[sub.partition == a]
        if not len(s):
            continue
        g = s.groupby(xcol)[col]
        med = g.median().reindex(x).values
        lo, hi = g.quantile(.25).reindex(x).values, g.quantile(.75).reindex(x).values
        if scale == 'log':
            med, lo, hi = (np.clip(v, FLOOR, None) for v in (med, lo, hi))
        if a != 'GT':
            ax.fill_between(x, lo, hi, color=COL[a], alpha=.10, lw=0, zorder=2)
        ax.plot(x, med, color=COL[a], marker=MK[a], ls=LS[a], lw=1.5,
                ms=8 if MK[a] == '*' else 5, mec='white', mew=.6, zorder=3)
    ax.set_yscale('log' if scale == 'log' else 'linear')
    if scale == 'linear':
        # Data-driven top: AOC Closure exceeds 1.0 when its partition is coarser than the
        # budget (ReconErr 1.4 at k=20), and a hard 1.12 ceiling silently clipped it.
        hi = 1.0
        for a in arms:
            s_ = sub[sub.partition == a]
            if len(s_):
                q = s_.groupby(xcol)[col].quantile(.75)
                if np.isfinite(q).any():
                    hi = max(hi, float(np.nanmax(q.values)))
        ax.set_ylim(-0.05, hi * 1.06 + 0.04)
    ax.set_xticks(x)
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))


fig, axes = plt.subplots(len(MET), 2, figsize=(7.2, 1.85 * len(MET)), squeeze=False)
for r, col in enumerate(MET):
    lab, scale = SPEC[col]
    for c, (xcol, sub, xlab) in enumerate(SWEEPS):
        ax = axes[r][c]
        style(ax)
        draw(ax, sub, xcol, col, scale)
        if c == 0:
            ax.set_ylabel(lab, fontsize=9, color=INK, labelpad=2)
        if r == 0:
            ax.set_title(xlab, fontsize=9, color=INK, pad=2)
        if r == len(MET) - 1:
            ax.set_xlabel(xcol, fontsize=9, color=INK2, labelpad=1)
hand = [Line2D([], [], color=COL[a], marker=MK[a], ls=LS[a], lw=1.5,
               ms=8 if MK[a] == '*' else 5, mec='white', mew=.6,
               label='GT partition' if a == 'GT' else a) for a in arms]
fig.tight_layout(pad=0.25, h_pad=0.55, w_pad=1.0)
top = max(a.get_position().y1 for a in axes.ravel())
bot = min(a.get_position().y0 for a in axes.ravel())
if TITLE:
    fig.suptitle(TITLE, fontsize=10, color=INK, y=top + 0.030, va='bottom')
fig.legend(handles=hand, loc='upper center', ncol=len(hand), frameon=False, fontsize=8.5,
           bbox_to_anchor=(.5, bot - 0.105))
for ext in ('png', 'pdf'):
    fig.savefig(f'{B}/figures/paper_partition_sweep_{REGIME}.{ext}', dpi=220)
print('wrote figures/paper_partition_sweep_%s.{png,pdf}  (fitter=%s, n=%d)'
      % (REGIME, FITTER, NSEED))
print(d.groupby(['p', 'k', 'partition'])[MET].median().to_string(float_format=lambda v: f'{v:.2e}'))
