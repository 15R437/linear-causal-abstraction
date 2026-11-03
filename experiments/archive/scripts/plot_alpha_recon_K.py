"""paper_alpha_recon, fixed get_S, fit_abstraction_K vs fit_abstraction_alternative.

One figure per blockiness; rows are the two fitters, columns the densities. Same acceptance
rule as the published figure: a method's value at tolerance alpha is its own reconstruction
if it BOTH clears the tolerance (RAE <= alpha) AND beats AOC Closure, else it falls back to
AOC Closure. The AOC line is therefore a ceiling by construction and the figure reads as
"how far below the fallback can each method get". Fallback baseline is per-fitter, so each
row is internally consistent.

Also emits paper_alpha_recon_K_ceiling.png: the GT-partition arm alone, which is what shows
whether the ceiling is reachable at all under each fitter.
"""
import sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
TAG = sys.argv[1] if len(sys.argv) > 1 else 'alpha_recon_K'
INK, INK2, GRID = '#111111', '#555555', '#dcdcdc'
OCC_COL, GT_COL, GTP_COL = '#eb6834', '#8a8a8a', '#6a3d9a'
LAM_COL = {0.0: '#86b6ef', 100.0: '#2a78d6', 1000.0: '#0d366b'}
LAM_MK = {0.0: 'o', 100.0: 's', 1000.0: '^'}
BASE = [('OC Closure', '#118a6e', 'v'), ('Greedy All-In', '#b5651d', 'D')]
ALPHAS = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3])
FIT_NAME = {'K': 'fit_abstraction_K', 'alt': 'fit_abstraction_alternative\n(faithful=True)'}

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.6, 'axes.edgecolor': '#888888',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

d = pd.read_csv(f'{B}/data/{TAG}.csv')
d = d[d.recon.notna()]


def curve(sub, occ):
    ys = []
    for al in ALPHAS:
        per = []
        for sd, g in sub.groupby('seed'):
            base = occ.get(sd, np.nan)
            ok = g[(g.rae <= al) & (g.recon < base)]
            per.append(ok.recon.min() if len(ok) else base)
        ys.append(np.nanmedian(per))
    return np.array(ys)


dens = sorted(d.density.unique())
blks = sorted(d.blk.unique())
fits = [f for f in ('K', 'alt') if f in set(d.fitter)]

for blk in blks:
    db = d[d.blk == blk]
    fig, axes = plt.subplots(len(fits), len(dens), figsize=(6.4, 1.95 * len(fits) + 0.35),
                             squeeze=False)
    for ri, ft in enumerate(fits):
        for ci, dn in enumerate(dens):
            ax = axes[ri][ci]
            s = db[(db.density == dn) & (db.fitter == ft)]
            occ = s[s.method == 'AOC Closure'].set_index('seed').recon
            gt = db[(db.density == dn) & (db.method == 'ground truth')].recon.median()
            gtp = s[s.method == 'GT partition'].recon.median()
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            ax.grid(True, color=GRID, lw=0.45, zorder=0); ax.set_axisbelow(True)
            ax.tick_params(colors=INK2, labelsize=8)
            if ri == 0:
                ax.set_title(r'$d=%.1f$  ($n{=}%d$)' % (dn, occ.size), fontsize=9,
                             color=INK, pad=3)
            if ci == 0:
                ax.set_ylabel('recon. error', fontsize=8, color=INK, labelpad=2)
            if not len(occ):          # cell not populated yet (partial / truncated run)
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center',
                        va='center', fontsize=8, color=INK2)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            seen = [occ.median(), gt, gtp]
            ax.axhline(occ.median(), color=OCC_COL, lw=1.5, ls=(0, (4, 2)), zorder=4)
            ax.axhline(gt, color=GT_COL, lw=1.0, ls=(0, (1, 2)), zorder=4)
            ax.axhline(gtp, color=GTP_COL, lw=1.0, ls=(0, (3, 1, 1, 1)), zorder=4)
            for nm, col, mk in BASE:
                sub = s[s.method == nm]
                if len(sub):
                    y = curve(sub, occ); seen += list(y)
                    ax.plot(ALPHAS, y, color=col, lw=1.2, marker=mk, ms=3.4, mec='white',
                            mew=0.6, zorder=5, clip_on=False)
            for lam, col in LAM_COL.items():
                sub = s[(s.method == 'GradCAP') & (s.lam_oc == lam)]
                if len(sub):
                    y = curve(sub, occ); seen += list(y)
                    ax.plot(ALPHAS, y, color=col, lw=1.5, marker=LAM_MK[lam], ms=4.0,
                            mec='white', mew=0.7, zorder=6, clip_on=False)
            ax.set_xscale('log'); ax.set_xlim(7e-5, 0.55)
            seen = [v for v in seen if np.isfinite(v)]
            lo, hi = (np.min(seen), np.max(seen)) if seen else (0.0, 1.0)
            pad = max(0.08 * (hi - lo), 0.01)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1])
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
            ax.tick_params(colors=INK2, labelsize=8)
            if ri == len(fits) - 1:
                ax.set_xlabel(r'tolerance $\alpha$', fontsize=9, color=INK, labelpad=1.5)
            else:
                ax.set_xticklabels([])
            if ri == 0:
                ax.set_title(r'$d=%.1f$  ($n{=}%d$)' % (dn, occ.size), fontsize=9,
                             color=INK, pad=3)
        axes[ri][0].annotate(FIT_NAME[ft], xy=(-0.44, 0.5), xycoords='axes fraction',
                             rotation=90, va='center', ha='center', fontsize=7.5, color=INK,
                             family='monospace', linespacing=1.4)

    handles = [Line2D([], [], color=c, lw=1.5, marker=LAM_MK[l], ms=4.0, mec='white', mew=0.7,
               label=r'GradCAP $\lambda_{\mathrm{oc}}{=}%g$' % l) for l, c in LAM_COL.items()]
    handles += [Line2D([], [], color=c, lw=1.2, marker=m, ms=3.4, mec='white', mew=0.6,
                label=n) for n, c, m in BASE]
    handles += [Line2D([], [], color=OCC_COL, lw=1.5, ls=(0, (4, 2)), label='AOC Closure (fallback)'),
                Line2D([], [], color=GTP_COL, lw=1.0, ls=(0, (3, 1, 1, 1)), label='GT partition (fitted)'),
                Line2D([], [], color=GT_COL, lw=1.0, ls=(0, (1, 2)), label='ground truth (analytic)')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=7.5,
               labelcolor=INK, handlelength=1.7, handletextpad=0.5, columnspacing=1.2,
               bbox_to_anchor=(0.5, -0.015))
    fig.suptitle('blockiness = %g' % blk, fontsize=9.5, color=INK, y=1.005)
    fig.tight_layout(rect=[0.055, 0.13, 1, 0.985], w_pad=1.3, h_pad=0.9)
    nm = f'paper_alpha_recon_K_blk{str(blk).replace(".", "")}'
    fig.savefig(f'{B}/figures/{nm}.pdf'); fig.savefig(f'{B}/figures/{nm}.png', dpi=300)
    plt.close(fig)
    print(f'wrote figures/{nm}.png (+pdf)')

# ---- numbers -------------------------------------------------------------------------
pd.set_option('display.width', 200)
print('\n=== ceiling check: can the fitter reconstruct the GROUND-TRUTH partition? ===')
print('(blockiness=1: A is exactly T A_high T^+, so recon=0 is attainable)')
g = (d[d.method.isin(['ground truth', 'GT partition'])]
     .groupby(['blk', 'density', 'method', 'fitter'])
     .agg(recon=('recon', 'median'), rae=('rae', 'median'),
          nrm=('norm_Ahigh', 'median'), marg=('marg', 'median'), n=('recon', 'size')))
print(g.to_string(float_format=lambda v: f'{v:.3g}'))

print('\n=== curves ===')
for blk in blks:
    for ft in fits:
        print(f'\n-- blk={blk}  fitter={ft}')
        for dn in dens:
            s = d[(d.blk == blk) & (d.density == dn) & (d.fitter == ft)]
            occ = s[s.method == 'AOC Closure'].set_index('seed').recon
            if not len(occ):
                continue
            line = 'd=%.1f n=%2d  GT %.3f  GTpart %.3f  AOC %.3f |' % (
                dn, occ.size,
                d[(d.blk == blk) & (d.density == dn) & (d.method == 'ground truth')].recon.median(),
                s[s.method == 'GT partition'].recon.median(), occ.median())
            for nm, _, _ in BASE:
                c = curve(s[s.method == nm], occ)
                line += '  %s a=.01 %.3f a=.3 %.3f |' % (nm, c[4], c[-1])
            for lam in LAM_COL:
                c = curve(s[(s.method == 'GradCAP') & (s.lam_oc == lam)], occ)
                line += '  GC%-5g a=.01 %.3f a=.3 %.3f |' % (lam, c[4], c[-1])
            print(line)
