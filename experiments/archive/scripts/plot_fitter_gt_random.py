"""fit_abstraction_auto on four partition types, at both faithfulness settings.

Layout: 2 rows (faithful=True / False) x 3 columns (RAE, ReconErr, time), bars grouped by
partition type. Rows share column axes so the faithfulness cost is read vertically and the
partition separation horizontally.

Scales are chosen per metric, not uniformly: RAE spans ~25 orders of magnitude and time ~3,
so both are log; reconstruction is linear on [0, ~1.1] where 1.0 is the edgeless answer and
0 is exact, which is the range the separation lives in.

Also emits a bar version at the anchor cell (p=100, k=10, d=0.5) for when a single
comparison is wanted rather than a trend.
"""
import sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
_args = [a for a in sys.argv[1:] if not a.startswith('-')]
TAG = _args[0] if _args else 'fitter_gt_random'
INK, INK2, GRID = '#111111', '#555555', '#dcdcdc'
COL = {'GT acyclic': '#6a3d9a', 'GT cyclic': '#b07aa1', 'random': '#c44536', 'UCA': '#118a6e'}
MK = {'GT acyclic': '*', 'GT cyclic': 'P', 'random': 'X', 'UCA': 'v'}
# UCA is an end-to-end T learner, not a partition handed to the fitter, so it uses no fitter
# and does not vary with `faithful`. Dashed to keep that visually distinct from the fitted arms.
NOFIT = {'UCA'}
ORDER = ['GT acyclic', 'GT cyclic', 'random', 'UCA']
MET = [('rae', 'RAE', 'log'), ('recon', 'ReconErr', 'linear'),
       ('secs', 'time (s)', 'log')]
ANCHOR = dict(p=100, k_budget=10, density=0.5)

mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'axes.linewidth': 0.6, 'axes.edgecolor': '#888888',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

d = pd.read_csv(f'{B}/data/{TAG}.csv')
d = d[d.recon.notna()].copy()
d['rae_p'] = d.rae.clip(lower=1e-33)          # for the log axis; exact zeros do occur
arms = [a for a in ORDER if a in set(d.partition)]
HAND = [Line2D([], [], color=COL[a], marker=MK[a], ms=7 if a in ('GT acyclic',) else 5.5,
               lw=1.5, ls=(0, (4, 2)) if a in ('UCA',) else '-', mec='white', mew=0.6,
               label=a + (' (end-to-end, no fitter)' if a == 'UCA' else ''))
        for a in arms]


def style(ax):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, lw=0.45, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8)


# In the p sweep, k is tied to p by design (block sizes are held comparable), so p cannot be
# varied with k pinned to its anchor -- that slice is empty. These are the (p, k) pairs.
PK = [(24, 4), (40, 6), (60, 8), (100, 10)]


def sweep_slice(axis):
    """Rows belonging to one sweep: the other factors held at their anchor values."""
    if axis == 'p':
        m = (d.density == ANCHOR['density']).values
        m &= d[['p', 'k_budget']].apply(tuple, axis=1).isin(PK).values
        return d[m]
    held = {k: v for k, v in ANCHOR.items() if k != axis}
    m = np.ones(len(d), bool)
    for k, v in held.items():
        m &= (d[k] == v).values
    return d[m]


def lines(axis, xlab, fname):
    s = sweep_slice(axis)
    if s[axis].nunique() < 2:
        print(f'  skip {fname}: only {s[axis].nunique()} level(s)')
        return
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 4.3), squeeze=False)
    for ri, faith in enumerate((1, 0)):
        for ci, (m, lab, sc) in enumerate(MET):
            ax = axes[ri][ci]; style(ax)
            col = 'rae_p' if m == 'rae' else m
            for a in arms:
                t = s[(s.partition == a) & (s.faithful == faith)]
                if not len(t):
                    continue
                g = t.groupby(axis)[col]
                med, q1, q3 = g.median(), g.quantile(.25), g.quantile(.75)
                ax.fill_between(med.index, q1, q3, color=COL[a], alpha=0.12, lw=0, zorder=2)
                ax.plot(med.index, med.values, color=COL[a], lw=1.5, marker=MK[a],
                        ls=(0, (4, 2)) if a in NOFIT else '-',
                        ms=7 if a == 'GT acyclic' else 5, mec='white', mew=0.6, zorder=5)
            ax.set_yscale(sc)
            if m == 'recon':
                ax.axhline(1.0, color='#999999', lw=0.8, ls=(0, (3, 3)), zorder=1)
                ax.set_ylim(-0.06, 1.15)
            ax.set_xlabel(xlab, fontsize=8.5, color=INK, labelpad=1.5)
            if ri == 0:
                ax.set_title(lab, fontsize=9.5, color=INK, pad=4)
            if ci == 0:
                ax.set_ylabel('faithful = %s' % ('True' if faith else 'False'),
                              fontsize=9, color=INK, labelpad=3)
    fig.legend(handles=HAND, loc='lower center', ncol=4, frameon=False, fontsize=8,
               labelcolor=INK, handlelength=1.8, handletextpad=0.5, columnspacing=1.6,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle('fit_abstraction_auto: RAE cannot tell a good partition from a bad one, '
                 'reconstruction can  (and only when faithful)', fontsize=9.5, color=INK,
                 y=1.01)
    fig.tight_layout(rect=[0, 0.09, 1, 0.965], w_pad=1.6, h_pad=1.1)
    fig.savefig(f'{B}/figures/{fname}.pdf'); fig.savefig(f'{B}/figures/{fname}.png', dpi=300)
    plt.close(fig); print(f'wrote figures/{fname}.png (+pdf)')


# Only the bar figure is wanted. Pass --lines to also emit the swept line versions.
if '--lines' in sys.argv:
    for ax_, xl, fn in (('p', 'low-level nodes $p$', 'fitter_gt_random_p'),
                        ('k_budget', 'clusters $k$', 'fitter_gt_random_k'),
                        ('density', 'density', 'fitter_gt_random_density')):
        lines(ax_, xl, fn)

# ---------------- bar version at the anchor cell
m = np.ones(len(d), bool)
for k, v in ANCHOR.items():
    m &= (d[k] == v).values
s = d[m]
fig, axes = plt.subplots(2, 3, figsize=(7.8, 4.2), squeeze=False)
for ri, faith in enumerate((1, 0)):
    for ci, (met, lab, sc) in enumerate(MET):
        ax = axes[ri][ci]; style(ax)
        col = 'rae_p' if met == 'rae' else met
        xs = np.arange(len(arms))
        med = [s[(s.partition == a) & (s.faithful == faith)][col].median() for a in arms]
        q1 = [s[(s.partition == a) & (s.faithful == faith)][col].quantile(.25) for a in arms]
        q3 = [s[(s.partition == a) & (s.faithful == faith)][col].quantile(.75) for a in arms]
        med, q1, q3 = (np.array(v, float) for v in (med, q1, q3))
        if sc == 'log':
            # a log bar needs a floor, and it must come from THIS panel's data -- reusing a
            # global floor (1e-34) flattens the time panel into four identical bars
            finite = np.concatenate([v[np.isfinite(v) & (v > 0)] for v in (med, q1, q3)])
            base = (finite.min() / 50) if len(finite) else 1e-34
            top = (finite.max() * 8) if len(finite) else 1.0
        else:
            base, top = 0.0, None
        ax.bar(xs, med - base, bottom=base, color=[COL[a] for a in arms], width=0.68,
               edgecolor='white', lw=0.8, zorder=3,
               hatch=['//' if a in NOFIT else '' for a in arms])
        ax.errorbar(xs, med, yerr=[np.clip(med - q1, 0, None), np.clip(q3 - med, 0, None)],
                    fmt='none', ecolor=INK2, lw=0.9, capsize=2, zorder=5)
        ax.set_yscale(sc)
        if sc == 'log':
            ax.set_ylim(base, top)
        if met == 'recon':
            ax.axhline(1.0, color='#999999', lw=0.8, ls=(0, (3, 3)), zorder=1)
            ax.set_ylim(0, 1.15)
        ax.set_xticks(xs)
        ax.set_xticklabels(arms, fontsize=7.5, color=INK, rotation=20, ha='right')
        if ri == 0:
            ax.set_title(lab, fontsize=9.5, color=INK, pad=4)
        if ci == 0:
            ax.set_ylabel('faithful = %s' % ('True' if faith else 'False'), fontsize=9,
                          color=INK, labelpad=3)
# n is the smallest per-method count, so the title never overstates the sample size
_n = int(s.groupby(['partition', 'faithful']).size().min()) if len(s) else 0
fig.suptitle(r'$p=100$,  $k=10$,  density $=0.5$   ($n=%d$ instances per method)' % _n,
             fontsize=10, color=INK, y=1.01)
# The UCA / hatching note is deliberately NOT drawn: it belongs in the paper caption.
fig.tight_layout(rect=[0, 0.03, 1, 0.965], w_pad=1.6, h_pad=1.4)
fig.savefig(f'{B}/figures/fitter_gt_random_bars.pdf')
fig.savefig(f'{B}/figures/fitter_gt_random_bars.png', dpi=300)
plt.close(fig); print('wrote figures/fitter_gt_random_bars.png (+pdf)')

# ---------------- numbers + which x-axis is most informative
pd.set_option('display.width', 220)
print('\n=== anchor cell (p=100, k=10, d=0.5): median over seeds ===')
print(s.groupby(['faithful', 'partition'])[['rae', 'recon', 'secs', 'marg', 'norm_Ahigh']]
      .median().reindex(pd.MultiIndex.from_product([[1, 0], arms], names=['faithful', 'partition']))
      .to_string(float_format=lambda v: f'{v:.3g}'))
print('\n=== routing check: fit_method by quotient cyclicity ===')
print(pd.crosstab(d.partition, [d.qcyc, d.fit_method]).to_string())
print('\n=== which x-axis separates GT from random most? ===')
print('(log10 gap in reconstruction, faithful=True: median(random) - median(GT acyclic))')
for axis in ('p', 'k_budget', 'density'):
    ss = sweep_slice(axis)
    if ss[axis].nunique() < 2:
        continue
    rows = []
    for v in sorted(ss[axis].unique()):
        w = ss[ss[axis] == v]
        gt = w[(w.partition == 'GT acyclic') & (w.faithful == 1)].recon.median()
        rd = w[(w.partition == 'random') & (w.faithful == 1)].recon.median()
        rows.append((v, gt, rd, rd - gt))
    r = pd.DataFrame(rows, columns=[axis, 'GT', 'random', 'gap'])
    print(f'\n{axis}:'); print(r.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
