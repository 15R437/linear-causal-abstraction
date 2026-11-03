"""Figure 1: the exactness-information frontier, one panel per density.

Design notes (dataviz skill):
  * Form: scatter -- the data's job is to show a two-objective frontier, so both
    objectives are axes and every method is a point. Two separate line plots would
    invite the misreading that the baselines' retention is good, when it is only good
    because they fail on error.
  * lambda_one_child is a MAGNITUDE, so it gets a sequential single-hue (blue) ordinal
    ramp, starting no lighter than step 250 (the light-surface 2:1 floor).
  * The references are categorical identities. Scatter is an all-pairs form, where only
    the first three categorical slots validate, and blue is spent on the ramp -- so OCC
    and UCA take orange and aqua, and `random` (a null floor, not a competing method)
    takes secondary ink. Marker shape carries identity too, so nothing is colour-alone.
  * Static print figure: no hover layer, light surface only.
"""
import sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable

B = '/Users/issy/Projects/linear-causal-abstraction'
SURFACE, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
RAMP = ['#86b6ef', '#6da7ec', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#0d366b']
REF = {'OCC': ('#eb6834', 's'), 'UCA': ('#1baf7a', '^'), 'random': (INK2, 'D')}
ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.25, 1.0]
FLOOR = 1e-4          # RAE below this is exact to machine precision; pin to the axis edge

d = pd.read_csv(f'{B}/experiments/abstraction_study/data/frontier2.csv')
d = d[d.iem.notna()].copy()
d['rae_p'] = d.rae.clip(lower=FLOOR)
lams = sorted(d[d.method == 'HailMary'].lam_oc.unique())
dens = sorted(d.density.unique())

fig, axes = plt.subplots(1, len(dens), figsize=(3.45 * len(dens), 3.5), sharey=True)
fig.patch.set_facecolor(SURFACE)

for ax, dn in zip(np.atleast_1d(axes), dens):
    s = d[d.density == dn]
    ax.set_facecolor(SURFACE)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color('#d8d7d2')
    ax.grid(True, which='major', color='#ececE8', lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    # NO composite line here: its x would be the threshold alpha, whereas every point's x
    # is a REALISED error. Mixing a threshold-indexed curve with outcome-indexed points on
    # one axis is not comparable (and the curve, being a per-instance minimum, necessarily
    # sits below the per-lambda medians). The alpha rule lives in its own figure.
    n_seeds = s[s.method == 'OCC'].seed.nunique()

    # HailMary, one colour step per lambda (mean over seeds)
    for i, lam in enumerate(lams):
        g = s[(s.method == 'HailMary') & (s.lam_oc == lam)]
        if not len(g):
            continue
        ax.scatter(max(g.rae_p.median(), FLOOR), g.iem.median(), s=74,
                   color=RAMP[min(i, len(RAMP) - 1)], marker='o',
                   edgecolor=SURFACE, lw=1.6, zorder=5)
    # references
    for nm, (col, mk) in REF.items():
        g = s[s.method == nm]
        if not len(g):
            continue
        ax.scatter(max(g.rae_p.median(), FLOOR), g.iem.median(), s=110, color=col,
                   marker=mk, edgecolor=SURFACE, lw=1.6, zorder=6)
        if dn == dens[0]:
            dy = {'OCC': 10, 'UCA': -4, 'random': 8}[nm]
            ax.annotate(nm, (max(g.rae_p.median(), FLOOR), g.iem.median()),
                        textcoords='offset points', xytext=(11, dy), fontsize=8.5,
                        color=INK2, zorder=7)

    ax.set_xscale('log')
    ax.set_xlim(FLOOR * 0.6, 1.2)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('Relative Abstraction Error', fontsize=9.5, color=INK2)
    ax.set_title(f'density {dn}   (n={n_seeds})', fontsize=10, color=INK, pad=7)
    if dn == dens[0]:
        ax.annotate('median RAE at the\nnumerical floor', (FLOOR * 1.3, 0.93),
                    fontsize=7.5, color=INK2, ha='left', va='bottom')
    ax.tick_params(colors=INK2, labelsize=8.5, length=3)

np.atleast_1d(axes)[0].set_ylabel('information loss', fontsize=9.5, color=INK2)

handles = [Line2D([], [], marker='o', ls='', mfc=RAMP[3], mec=SURFACE, ms=8,
                  label=r'HailMary (colour = $\lambda_{\mathrm{oc}}$, the one-child weight)')]
handles += [Line2D([], [], marker=mk, ls='', mfc=c, mec=SURFACE, ms=8, label=nm)
            for nm, (c, mk) in REF.items()]

fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=8.5,
           labelcolor=INK2, bbox_to_anchor=(0.5, -0.05), handletextpad=0.4,
           columnspacing=1.6)
fig.suptitle('Exactness-information trade-off: bottom-left is better',
             fontsize=10.5, color=INK, y=1.0)
fig.tight_layout(rect=[0, 0.09, 0.945, 0.95])

# ordinal colourbar for lambda (index-spaced: lambda=0 has no log position)
cmap = ListedColormap(RAMP[:len(lams)])
norm = BoundaryNorm(np.arange(len(lams) + 1) - 0.5, len(lams))
cax = fig.add_axes([0.962, 0.32, 0.010, 0.46])
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                  ticks=np.arange(len(lams)))
cb.ax.set_yticklabels([('%g' % l) for l in lams], fontsize=7.5, color=INK2)
cb.outline.set_visible(False)
cb.ax.tick_params(length=0)
cb.set_label(r'$\lambda_{\mathrm{oc}}$', fontsize=9, color=INK2, labelpad=3)
out = f'{B}/experiments/abstraction_study/figures/frontier.png'
fig.savefig(out, dpi=220, facecolor=SURFACE, bbox_inches='tight')
fig.savefig(out.replace('.png', '.pdf'), facecolor=SURFACE, bbox_inches='tight')
print('wrote', out, 'and .pdf')

# the numbers behind the figure
print('\nmean over seeds, by density:')
for dn in dens:
    s = d[d.density == dn]
    t = s.groupby(['method', 'lam_oc'], dropna=False).agg(
        n=('rae', 'size'), rae=('rae', 'median'), iem=('iem', 'mean'),
        dre=('dre', 'mean'), k_used=('k_used', 'mean'), valid=('valid', 'mean'),
        marg=('marg', 'mean')).round(3)
    print(f'\n--- density {dn} ---'); print(t.to_string())
