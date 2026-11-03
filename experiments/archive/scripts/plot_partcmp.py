"""Condition violations vs information loss, for HailMary and graph-partition baselines.

Two figures, same layout: x = violation count (All-In, or one-child), y = information
loss (intra-cluster edge mass). Lower-left is better on both axes, so the plot reads as a
frontier. Faceted by density like paper_alpha_main.

HailMary is the only method with a dial, so it is drawn as a TRAJECTORY over
lambda_one-child; every baseline is a single point, direct-labelled. Colour encodes the
FAMILY an algorithm belongs to, not its identity -- identity is carried by the label, which
keeps the hue count at four:

  blue    HailMary (lambda_one-child trajectory) and its graph-only end point
  green   All-In-targeted: structural / regular equivalence and partition refinement
  orange  AOC Closure -- one-child targeted
  grey    controls: community detection (wrong objective by construction), UCA, random

Every method runs under the SAME budget K = AOC Closure's cluster count, because singletons
trivially satisfy All-In at zero intra-cluster mass.
"""
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
INK, INK2, GRID = '#111111', '#555555', '#dcdcdc'
HM, ALLIN, OCC, CTRL = '#3d6bd6', '#118a6e', '#e4572e', '#8a8a8a'
LAMS = [0., 0.3, 1., 3., 10., 30., 100., 300., 1000.]

# (method, colour, marker, label, label offset in points)
BASE = [
    ('AOC Closure',   OCC,   's', 'AOC Closure',   (5, 4)),
    ('all-in greedy', ALLIN, 'D', 'all-in greedy', (5, 4)),
    ('bisim',         ALLIN, '^', 'bisim',         (5, -9)),
    ('sig-agglom',    ALLIN, 'v', 'sig-agglom',    (5, 4)),
    ('sig-kmeans',    ALLIN, 'P', 'sig-kmeans',    (5, -9)),
    ('hc',            CTRL,  'X', 'hc',            (5, 4)),
    ('spectral-ncut', CTRL,  'o', 'spectral-ncut', (5, 4)),
    ('UCA',           CTRL,  '*', 'UCA',           (5, -9)),
    ('random',        CTRL,  '.', 'random',        (5, 4)),
]
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.6, 'axes.edgecolor': '#888888',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

d = pd.read_csv(f'{B}/data/partcmp.csv')


def build(xcol, xlabel, out, dens_list, figsize, label_points=True):
    fig, axes = plt.subplots(1, len(dens_list), figsize=figsize)
    axes = np.atleast_1d(axes)
    for ax, dn in zip(axes, dens_list):
        s = d[d.density == dn]
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.grid(True, color=GRID, lw=0.45, zorder=0); ax.set_axisbelow(True)

        # HailMary trajectory over the lambda dial
        hm = s[s.method == 'HailMary']
        xs = [hm[hm.lam_oc == l][xcol].median() for l in LAMS]
        ys = [hm[hm.lam_oc == l]['intra'].median() for l in LAMS]
        ax.plot(xs, ys, '-', color=HM, lw=1.4, zorder=5, alpha=0.9)
        ax.scatter(xs, ys, s=np.linspace(9, 34, len(LAMS)), color=HM, zorder=6,
                   edgecolor='white', linewidth=0.5)
        g = s[s.method == 'HM graph-only']
        if len(g):
            ax.scatter([g[xcol].median()], [g['intra'].median()], s=42, marker='*',
                       color=HM, zorder=7, edgecolor='white', linewidth=0.6)

        for meth, col, mk, lab, off in BASE:
            sub = s[s.method == meth]
            if not len(sub):
                continue
            x, y = sub[xcol].median(), sub['intra'].median()
            ax.scatter([x], [y], s=26, marker=mk, color=col, zorder=6,
                       edgecolor='white', linewidth=0.5)

        ax.set_xlabel(xlabel + r'  ($\downarrow$ better)', fontsize=8.5, color=INK, labelpad=1.5)
        ax.set_title(r'$d=%.1f$  ($n{=}%d$)' % (dn, s.seed.nunique()), fontsize=9,
                     color=INK, pad=3)
        ax.tick_params(labelsize=7.5, colors=INK2)
        ax.set_ylim(-0.04, 1.04)
        # symlog: most methods pile up in 0-25 while the controls sit at 250-350,
        # so a linear axis renders the interesting half as one blob.
        ax.set_xscale('symlog', linthresh=10, linscale=0.7)
        ax.set_xlim(left=-0.6)
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
            lambda v, _: ('%g' % v) if v in (0, 10, 100) else ''))
    axes[0].set_ylabel('information loss  ($\\downarrow$ better)\n(intra-cluster edge mass)', fontsize=8.5,
                       color=INK, labelpad=2)

    handles = [Line2D([], [], color=HM, lw=1.4, marker='o', ms=3.5,
                      label=r'HailMary ($\lambda_{\mathrm{oc}}$: 0$\to$1000)'),
               Line2D([], [], color=HM, lw=0, marker='*', ms=6, label='HM graph-only')]
    handles += [Line2D([], [], color=c, lw=0, marker=m, ms=4.5, label=l)
                for _, c, m, l, _ in BASE]
    fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, fontsize=7,
               labelcolor=INK, handlelength=1.6, handletextpad=0.5, columnspacing=1.4,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.15, 1, 1.0], w_pad=1.2)
    fig.savefig(f'{B}/figures/{out}.pdf'); fig.savefig(f'{B}/figures/{out}.png', dpi=300)
    plt.close(fig)
    print('wrote figures/%s.pdf (+png)' % out)


MAIN = [0.1, 0.5, 0.9]
ALL = sorted(d.density.unique())
build('allin_nodes', 'All-In violating nodes', 'paper_allin', MAIN, (7.2, 2.5))
build('oc_excess', 'one-child excess child-edges', 'paper_onechild', MAIN, (7.2, 2.5))
build('allin_nodes', 'All-In violating nodes', 'paper_allin_full', ALL, (10.5, 2.4))
build('oc_excess', 'one-child excess child-edges', 'paper_onechild_full', ALL, (10.5, 2.4))

print('\nmedians by method (pooled):')
t = d.groupby('method').agg(n=('intra', 'size'), allin_nodes=('allin_nodes', 'median'),
                            oc_excess=('oc_excess', 'median'), intra=('intra', 'median'),
                            allin_ok=('allin_ok', 'mean'), oc_ok=('oc_ok', 'mean'),
                            k_used=('k_used', 'median'), secs=('secs', 'mean')).round(3)
print(t.sort_values('allin_nodes').to_string())
