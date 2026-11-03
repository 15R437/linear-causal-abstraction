"""Paper figures: information loss vs tolerated abstraction error.

HONEST BY CONSTRUCTION: each line is a SINGLE FIXED lambda_one-child, not a per-instance
pick from a sweep. Every point is therefore reproducible by one training configuration.
(At d >= 0.5 a fixed lambda = 1000 matches the per-instance selection exactly; at d = 0.1
selection would give 0.122 against fixed-300's 0.209, and that gap is stated in the caption
rather than plotted.)

Typography targets a single-column ICLR page: 5.5in text width, serif to match the body
font, 8pt ticks / 9pt labels. Main figure is 3 densities; the 5-density version goes in the
appendix.
"""
import sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
INK, INK2, GRID = '#111111', '#555555', '#dcdcdc'
# sequential blue ramp for lambda (a magnitude), ordinal steps >= 250 for the light surface
LAM_COL = {0.0: '#86b6ef', 300.0: '#2a78d6', 1000.0: '#0d366b'}
LAM_MK = {0.0: 'o', 300.0: 's', 1000.0: '^'}   # shape too, so overlapping lines separate
OCC_COL, BASE_COL = '#eb6834', '#7a7a7a'
ALPHAS = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3])

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


def curve(sub, occ, lam=None):
    """Median information loss of (fixed-lambda arm, else AOC Closure) across alphas."""
    ys = []
    for al in ALPHAS:
        per = []
        for sd, g in sub.groupby('seed'):
            base = occ.get(sd, np.nan)
            gg = g if lam is None else g[g.lam_oc == lam]
            ok = gg[(gg.rae <= al) & (gg.iem < base)]
            per.append(ok.iem.min() if len(ok) else base)
        ys.append(np.nanmedian(per))
    return np.array(ys)


def build(dens_list, out, figsize, ncol_legend):
    # Per-panel LINEAR y. Tried a shared log axis: rejected, because the data spans barely
    # one decade (0.10-0.85), so log leaves 60% of the height empty below 0.2 while
    # COMPRESSING the d=0.9 panel, whose differences are 1.19x ratios. A shared linear 0-1
    # axis hides 0.02 differences. Per-panel linear shows each panel's structure; the cost
    # is cross-panel comparability, which the caption must state.
    fig, axes = plt.subplots(1, len(dens_list), figsize=figsize, sharey=False)
    axes = np.atleast_1d(axes)
    for ax, dn in zip(axes, dens_list):
        s = d[d.density == dn]
        occ = s[s.method == 'OCC'].set_index('seed').iem
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.grid(True, color=GRID, lw=0.45, zorder=0)
        ax.set_axisbelow(True)

        seen = [occ.median()]
        ax.axhline(occ.median(), color=OCC_COL, lw=1.5, zorder=4, ls=(0, (4, 2)))
        for meth in ('UCA', 'random'):
            sub = s[s.method == meth]
            if len(sub):
                y = curve(sub, occ); seen += list(y)
                ax.plot(ALPHAS, y, color=BASE_COL, lw=0.9, alpha=0.75, zorder=3)
        for lam, col in LAM_COL.items():
            sub = s[(s.method == 'HailMary') & (s.lam_oc == lam)]
            if len(sub):
                y = curve(sub, occ, lam=lam); seen += list(y)
                ax.plot(ALPHAS, y, color=col, lw=1.5, marker=LAM_MK[lam], ms=4.0,
                        mec='white', mew=0.7, solid_capstyle='round', zorder=5,
                        clip_on=False)

        ax.set_xscale('log')
        ax.set_xlim(7e-5, 0.55)
        lo, hi = np.nanmin(seen), np.nanmax(seen)
        pad = max(0.10 * (hi - lo), 0.008)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1])
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        ax.tick_params(colors=INK2, labelsize=8)
        ax.set_xlabel(r'tolerance $\alpha$', fontsize=9, color=INK, labelpad=1.5)
        ax.set_title(r'$d=%.1f$  ($n{=}%d$)' % (dn, occ.size), fontsize=9, color=INK, pad=3)
    for ax in axes:
        ax.tick_params(axis='y', labelsize=8, colors=INK2)
    axes[0].set_ylabel('information loss', fontsize=9, color=INK, labelpad=2)

    handles = [Line2D([], [], color=c, lw=1.5, marker=LAM_MK[l], ms=4.0, mec='white',
                      mew=0.7, label=r'HailMary  $\lambda_{\mathrm{oc}}{=}%g$' % l)
               for l, c in LAM_COL.items()]
    handles += [Line2D([], [], color=OCC_COL, lw=1.5, ls=(0, (4, 2)), label='AOC Closure'),
                Line2D([], [], color=BASE_COL, lw=0.9, label='UCA / random')]
    fig.legend(handles=handles, loc='lower center', ncol=ncol_legend, frameon=False,
               fontsize=8, labelcolor=INK, handlelength=1.8, handletextpad=0.5,
               columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.10, 1, 1.02], w_pad=1.4)
    fig.savefig(f'{B}/figures/{out}.pdf')
    fig.savefig(f'{B}/figures/{out}.png', dpi=300)
    plt.close(fig)
    print('wrote figures/%s.pdf (+png)' % out)


build([0.1, 0.5, 0.9], 'paper_alpha_main', (5.5, 1.85), 5)
build(sorted(d.density.unique()), 'paper_alpha_full', (6.6, 1.8), 5)

print('\nnumbers (median information loss; fixed lambda + AOC Closure fallback):')
for dn in sorted(d.density.unique()):
    s = d[d.density == dn]
    occ = s[s.method == 'OCC'].set_index('seed').iem
    line = 'd=%.1f  n=%d  OCC %.3f  |' % (dn, occ.size, occ.median())
    for lam in LAM_COL:
        c = curve(s[(s.method == 'HailMary') & (s.lam_oc == lam)], occ, lam=lam)
        i1, i5 = list(ALPHAS).index(1e-3), list(ALPHAS).index(3e-2)
        line += '  lam=%-5g a=.001 %.3f  a=.03 %.3f |' % (lam, c[i1], c[i5])
    print(line)
