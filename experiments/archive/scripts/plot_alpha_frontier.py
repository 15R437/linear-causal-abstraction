"""Frontier figure: tolerance alpha vs reconstruction error, one panel per density.

Successor to paper_alpha_main, with ReconErr on y instead of information loss.

At tolerance alpha a method contributes its OWN ReconErr if it clears the tolerance
(RAE <= alpha) AND beats AOC Closure on ReconErr; otherwise it falls back to AOC Closure.
The AOC line is therefore a ceiling by construction and every curve starts there as
alpha -> 0, so "never worse than the safe default" is visible rather than asserted, and the
spread between curves at a given alpha is the RANGE OF CHOICES the family offers.

The ground-truth line is the floor: at blockiness 1, A is exactly T A_high T^+, so the
generating abstraction reconstructs to ~0 and the whole y range is genuinely available.
(At blockiness 0.8 that floor is already 0.43, which is why this figure needs blk=1.)
"""
import sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
TAG = sys.argv[1] if len(sys.argv) > 1 else 'alpha_frontier'
INK, INK2, GRID = '#111111', '#555555', '#dcdcdc'
AOC_COL, GT_COL = '#eb6834', '#8a8a8a'
LAM_COL = {0.0: '#86b6ef', 100.0: '#2a78d6', 1000.0: '#0d366b'}
LAM_MK = {0.0: 'o', 100.0: 's', 1000.0: '^'}
BASE = [('OC Closure', '#118a6e', 'v'), ('Greedy All-In', '#b5651d', 'D')]
ALPHAS = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3])

mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'axes.linewidth': 0.6, 'axes.edgecolor': '#888888',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

d = pd.read_csv(f'{B}/data/{TAG}.csv')
d = d[d.recon.notna()].copy()


def curve(sub, aoc):
    """Median over instances of: own ReconErr if RAE<=alpha and better than AOC, else AOC."""
    ys = []
    for al in ALPHAS:
        per = []
        for sd, g in sub.groupby('seed'):
            base = aoc.get(sd, np.nan)
            ok = g[(g.rae <= al) & (g.recon < base)]
            per.append(ok.recon.min() if len(ok) else base)
        ys.append(np.nanmedian(per))
    return np.array(ys)


dens = sorted(d.density.unique())
fig, axes = plt.subplots(1, len(dens), figsize=(5.8, 1.95), sharey=False)
axes = np.atleast_1d(axes)
for ax, dn in zip(axes, dens):
    s = d[d.density == dn]
    aoc = s[s.method == 'AOC Closure'].set_index('seed').recon
    gt = s[s.method == 'ground truth'].recon.median()
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, lw=0.45, zorder=0); ax.set_axisbelow(True)

    seen = [aoc.median(), gt]
    ax.axhline(aoc.median(), color=AOC_COL, lw=1.5, ls=(0, (4, 2)), zorder=4)
    ax.axhline(gt, color=GT_COL, lw=1.0, ls=(0, (1, 2)), zorder=4)
    for nm, col, mk in BASE:
        sub = s[s.method == nm]
        if len(sub):
            y = curve(sub, aoc); seen += list(y)
            ax.plot(ALPHAS, y, color=col, lw=1.2, marker=mk, ms=3.4, mec='white',
                    mew=0.6, zorder=5, clip_on=False)
    for lam, col in LAM_COL.items():
        sub = s[(s.method == 'GradCAP') & (s.lam_oc == lam)]
        if len(sub):
            y = curve(sub, aoc); seen += list(y)
            ax.plot(ALPHAS, y, color=col, lw=1.5, marker=LAM_MK[lam], ms=4.0, mec='white',
                    mew=0.7, zorder=6, clip_on=False)
    ax.set_xscale('log'); ax.set_xlim(7e-5, 0.55)
    seen = [v for v in seen if np.isfinite(v)]
    lo, hi = (min(seen), max(seen)) if seen else (0, 1)
    pad = max(0.08 * (hi - lo), 0.01)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1])
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlabel(r'tolerance $\alpha$', fontsize=9, color=INK, labelpad=1.5)
    ax.set_title(r'$d=%.1f$  ($n{=}%d$)' % (dn, aoc.size), fontsize=9, color=INK, pad=3)
axes[0].set_ylabel('ReconErr', fontsize=9, color=INK, labelpad=2)

handles = [Line2D([], [], color=c, lw=1.5, marker=LAM_MK[l], ms=4.0, mec='white', mew=0.7,
                  label=r'GradCAP $\lambda_{\mathrm{oc}}{=}%g$' % l) for l, c in LAM_COL.items()]
handles += [Line2D([], [], color=c, lw=1.2, marker=m, ms=3.4, mec='white', mew=0.6, label=n)
            for n, c, m in BASE]
handles += [Line2D([], [], color=AOC_COL, lw=1.5, ls=(0, (4, 2)), label='AOC Closure (fallback)'),
            Line2D([], [], color=GT_COL, lw=1.0, ls=(0, (1, 2)), label='ground truth')]
fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=7.5,
           labelcolor=INK, handlelength=1.7, handletextpad=0.5, columnspacing=1.2,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.18, 1, 1.02], w_pad=1.3)
fig.savefig(f'{B}/figures/paper_alpha_frontier.pdf')
fig.savefig(f'{B}/figures/paper_alpha_frontier.png', dpi=300)
plt.close(fig)
print('wrote figures/paper_alpha_frontier.png (+pdf)\n')
for dn in dens:
    s = d[d.density == dn]
    aoc = s[s.method == 'AOC Closure'].set_index('seed').recon
    line = 'd=%.1f n=%d  GT %.3f  AOC %.3f |' % (
        dn, aoc.size, s[s.method == 'ground truth'].recon.median(), aoc.median())
    for nm, _, _ in BASE:
        c = curve(s[s.method == nm], aoc)
        line += '  %s a=.01 %.3f a=.3 %.3f |' % (nm, c[4], c[-1])
    for lam in LAM_COL:
        c = curve(s[(s.method == 'GradCAP') & (s.lam_oc == lam)], aoc)
        line += '  GC%-5g a=.01 %.3f a=.3 %.3f |' % (lam, c[4], c[-1])
    print(line)
