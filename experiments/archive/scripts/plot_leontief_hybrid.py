"""Leontief: information loss against coefficient threshold, HailMary hybrid vs AOC Closure.

The hybrid rule is the one from the synthetic experiments, applied to a single dataset:
run HailMary at a FIXED lambda_one-child = 0 with 3 restarts, accept the run with lowest
information loss among those whose Relative Abstraction Error clears alpha AND which
improves on AOC Closure, and otherwise fall back to AOC Closure. (Sweeping lambda over
{0, 10, 100} and taking the best changes only two of the eleven thresholds -- 0.448 vs
0.622 at tau=0.005 and 0.542 vs 0.692 at tau=0.008 -- so the fixed configuration is
reported instead, being a single reproducible setting.) The composite is therefore
never worse than AOC Closure, and alpha -> 0 recovers it.

A random balanced partition at matched k gets the same treatment, as a check that the
learning rather than the granularity is doing the work.

The top axis reports the graph density corresponding to each threshold -- a monotone
reparameterisation of the same control variable, not a second measure.
"""
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
INK, INK2, GRID = '#111111', '#555555', '#dcdcdc'
BLUE, ORANGE, GREY = '#2a78d6', '#eb6834', '#8a8a8a'
ALPHA = 0.05
PERC = (0.014, 0.016)
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.6, 'axes.edgecolor': '#888888',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.01,
})

d = pd.read_csv(f'{B}/data/leontief_thresh_full.csv')
d = d[d.iem.notna()].copy()
sweep = pd.read_csv(f'{B}/data/leontief_thresh_sweep.csv')
thrs = sorted(d.thr.unique())

rows = []
for t in thrs:
    s = d[d.thr == t]
    occ = s[s.method == 'OCC'].iem.iloc[0]
    rec = dict(thr=t, occ=occ)
    for nm, meth in (('hm', 'HailMary'), ('rb', 'random_balanced')):
        sub = s[s.method == meth]
        if meth == 'HailMary':
            sub = sub[sub.lam_oc == 0.0]
        ok = sub[(sub.rae <= ALPHA) & (sub.iem < occ)]
        rec[nm] = ok.iem.min() if len(ok) else occ
        rec[nm + '_acc'] = len(ok)
        rec[nm + '_n'] = len(sub)
        if len(ok):
            rec[nm + '_lam'] = ok.loc[ok.iem.idxmin()].lam_oc
    rows.append(rec)
h = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(4.4, 2.6))
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.grid(True, color=GRID, lw=0.45, zorder=0); ax.set_axisbelow(True)
ax.plot(h.thr, h.occ, '-s', color=ORANGE, lw=1.6, ms=4.2, mec='white', mew=0.7, zorder=4)
ax.plot(h.thr, h.rb, '-^', color=GREY, lw=1.1, ms=3.6, mec='white', mew=0.6, zorder=3,
        alpha=0.85)
ax.plot(h.thr, h.hm, '-o', color=BLUE, lw=1.8, ms=4.6, mec='white', mew=0.7, zorder=5)
ax.set_xscale('log')
ax.set_xlim(8e-4, 0.13)
ax.set_ylim(0, 1.0)
ax.set_xlabel(r'coefficient threshold $\tau$', fontsize=9, color=INK, labelpad=1.5)
ax.set_ylabel('information loss', fontsize=9, color=INK, labelpad=2)
ax.tick_params(colors=INK2, labelsize=8)

# top axis: density at each threshold (same control variable, reparameterised)
ax2 = ax.twiny()
ax2.set_xscale('log'); ax2.set_xlim(ax.get_xlim())
ticks = [0.001, 0.01, 0.1]
ax2.set_xticks(ticks)
ax2.set_xticklabels(['%.2f' % np.interp(t, sweep.thr, sweep.density) for t in ticks],
                    fontsize=7.5, color=INK2)
ax2.set_xlabel('graph density', fontsize=8, color=INK2, labelpad=3)
for sp in ('right', 'left'):
    ax2.spines[sp].set_visible(False)
ax2.tick_params(colors=INK2, length=2.5)

handles = [Line2D([], [], color=BLUE, lw=1.8, marker='o', ms=4.6, mec='white', mew=0.7,
                  label='HailMary + fallback\n' + r'($\lambda_{\rm oc}{=}0$, $\alpha{=}%g$)' % ALPHA),
           Line2D([], [], color=ORANGE, lw=1.6, marker='s', ms=4.2, mec='white', mew=0.7,
                  label='AOC Closure'),
           Line2D([], [], color=GREY, lw=1.1, marker='^', ms=3.6, mec='white', mew=0.6,
                  label='random partition + fallback')]
ax.legend(handles=handles, frameon=False, fontsize=7.5, labelcolor=INK,
          loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=1.8,
          handletextpad=0.5, borderpad=0.2, labelspacing=0.9)
fig.set_size_inches(6.4, 2.5)
fig.subplots_adjust(left=0.095, right=0.655, top=0.82, bottom=0.18)
fig.savefig(f'{B}/figures/leontief_hybrid.pdf')
fig.savefig(f'{B}/figures/leontief_hybrid.png', dpi=300)
print('wrote figures/leontief_hybrid.pdf (+png)\n')
pd.set_option('display.width', 200)
print(h.round(4).to_string(index=False))
