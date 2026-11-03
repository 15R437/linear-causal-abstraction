"""Leontief: information loss vs coefficient threshold, 20 independent restarts.

The line is what the method DELIVERS -- best accepted run of 20 (accept when RAE <= alpha
and it improves on AOC Closure, else fall back), which is the hybrid rule from the synthetic
experiments applied to a single dataset. The band shows the interquartile range of the
individual accepted runs, so the run-to-run variability behind that best-of is visible
rather than hidden by the min.

Restarts are genuinely independent here: AOC Closure returns k=1 on this graph, so
initialisation is regularised_surjective_init with a distinct seed per run.
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
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.6, 'axes.edgecolor': '#888888',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.01,
})

d = pd.read_csv(f'{B}/data/leontief_thresh_r20.csv')
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
        ok = sub[(sub.rae <= ALPHA) & (sub.iem < occ)]
        rec[nm] = ok.iem.min() if len(ok) else occ
        rec[nm + '_q1'] = ok.iem.quantile(.25) if len(ok) else occ
        rec[nm + '_q3'] = ok.iem.quantile(.75) if len(ok) else occ
        rec[nm + '_acc'] = len(ok)
        rec[nm + '_n'] = len(sub)
    rows.append(rec)
h = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(6.4, 2.5))
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax.grid(True, color=GRID, lw=0.45, zorder=0); ax.set_axisbelow(True)
ax.fill_between(h.thr, h.hm_q1, h.hm_q3, color=BLUE, alpha=0.15, lw=0, zorder=2)
ax.plot(h.thr, h.occ, '-s', color=ORANGE, lw=1.6, ms=4.2, mec='white', mew=0.7, zorder=4)
ax.plot(h.thr, h.rb, '-^', color=GREY, lw=1.1, ms=3.6, mec='white', mew=0.6, zorder=3, alpha=0.85)
ax.plot(h.thr, h.hm, '-o', color=BLUE, lw=1.8, ms=4.6, mec='white', mew=0.7, zorder=5)
ax.set_xscale('log')
ax.set_xlim(8e-4, 0.13)
ax.set_ylim(0, 1.0)
ax.set_xlabel(r'coefficient threshold $\tau$', fontsize=9, color=INK, labelpad=1.5)
ax.set_ylabel('information loss', fontsize=9, color=INK, labelpad=2)
ax.tick_params(colors=INK2, labelsize=8)

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
           Line2D([], [], color=BLUE, lw=6, alpha=0.25, label='IQR of accepted runs'),
           Line2D([], [], color=ORANGE, lw=1.6, marker='s', ms=4.2, mec='white', mew=0.7,
                  label='AOC Closure'),
           Line2D([], [], color=GREY, lw=1.1, marker='^', ms=3.6, mec='white', mew=0.6,
                  label='random partition + fallback')]
ax.legend(handles=handles, frameon=False, fontsize=7.5, labelcolor=INK,
          loc='center left', bbox_to_anchor=(1.02, 0.5), handlelength=1.8,
          handletextpad=0.5, borderpad=0.2, labelspacing=0.85)
fig.subplots_adjust(left=0.095, right=0.655, top=0.82, bottom=0.18)
fig.savefig(f'{B}/figures/leontief_hybrid.pdf')
fig.savefig(f'{B}/figures/leontief_hybrid.png', dpi=300)
print('wrote figures/leontief_hybrid.pdf (+png)   [20 restarts]\n')
pd.set_option('display.width', 220)
print(h[['thr','occ','hm','hm_q1','hm_q3','hm_acc','hm_n','rb','rb_acc']].round(4).to_string(index=False))
