"""`fit_abstraction_K` against the oracle T, as p grows at fixed p/k and as k grows at fixed p.

    paper_tk_sweep_compact.{png,pdf}  RAE | ReconErr                   (2 rows, the figure)
    paper_tk_sweep.{png,pdf}    RAE | ReconErr | edge recall | ||A||_F (4 rows, reference)

Both formats every run; the PDF embeds Type-42 fonts, since Type-3 is rejected by many
venues. Laid out for a tight vertical budget: tight_layout runs FIRST, then the title and
legend are pinned to the resulting axes box, so neither reserves an empty band.

TWO LINES, one per REGIME (acyclic / cyclic quotient), both the SAME fitter -- `--arm=`
picks which, default `eigen` (eigen_solver.fit_abstraction_eigen: sinks free, cyclic SCCs
pinned by an eigenvector equation). `--arm='_K unfaithful'` draws fit_abstraction_K instead,
which is what this figure showed before 2026-09-26.

Title is the figure's own; the settings (density 0.5, blockiness 1, n=20) belong in the
caption, not on the axes.

THE ORACLE IS NOT DRAWN. It is a constant in every panel here -- RAE ~2e-16, recall 1.000 by
construction -- so it spends two lines on a flat reference. It is still scored in the CSV and
printed in the table below, which is where to read it.

WHY EDGE RECALL RATHER THAN ||A_high||_F. A_high is defined only up to D-conjugation, and
||.||_F is not invariant under it: regauging the oracle to the fitter's unit-column
convention moves it by -14% to +40%, growing with k, so the raw norms are not on one axis at
all (tk_sweep.py `score` now regauges, which is what makes the norm row honest). Edge recall
is invariant -- conjugation scales entry (i,j) by d_j/d_i != 0, leaving the support alone --
and it asks the question directly: where there is a low-level cross-cluster edge, is there a
high-level edge? It also collapses the two oracle lines onto a flat 1.0, where the two oracle
NORMS differ by 5x between regimes and would need separate references.

RECALL IS A SUPPORT TEST, NOT A MAGNITUDE TEST. At the 1e-3 threshold an edge attenuated 167x
still counts as present -- and that happens: on the p=40, k=4, seed=5 fit, recall is 1.000
while the two edges INTO THE SINK CLUSTER come back at 0.006 of their true magnitude and the
other two are exact. Read recall beside ReconErr, which is what charges that attenuation.

WHAT THE FIGURE SHOWS. RAE is ~1e-7 at nearly every cell in BOTH regimes -- flat, and flat
across cells where the fit has annihilated the high-level graph outright. It cannot separate
a recovered abstraction from a destroyed one. Recall can, and the two regimes fail
differently: acyclic degrades gently with k, cyclic is all-or-nothing (total annihilation at
k <= 16, then exact recovery of T at k >= 20, 1-|cos| ~ 1e-12).
"""
import sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
_args = [a for a in sys.argv[1:] if not a.startswith('-')]
TAG = _args[0] if _args else 'tk_sweep'
_m = [a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--metrics=')]
COMPACT = _m[0].split(',') if _m else ['rae', 'recon']
RATIO, P_FIXED = 10, 100
INK, INK2, GRID = '#111111', '#555555', '#dcdcdc'
_a = [a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--arm=')]
ORACLE, FIT = 'T_gt', (_a[0] if _a else 'eigen')
RCOL = {'acyclic': '#1f5fa8', 'cyclic': '#d1622b'}
AMK = {ORACLE: 'o', FIT: '*'}
ALS = {ORACLE: (0, (4, 2)), FIT: '-'}
FLOOR = 1e-18
SPEC = {'rae': ('RAE', 'log'), 'recon': ('ReconErr', 'linear'),
        'edge_recall': ('edge recall', 'linear'),
        'norm_Ahigh': (r'$\|\tilde{\mathbf{A}}\|_F$', 'log')}

mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'axes.linewidth': 0.6, 'axes.edgecolor': '#888888',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.01,
    # Type-42 (TrueType) rather than Type-3, which many venues reject outright.
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

d = pd.read_csv(f'{B}/data/{TAG}.csv')
d = d[d.recon.notna() & d.arm.isin([ORACLE, FIT])].copy()
NSEED = int(d.groupby(['p', 'k', 'arm', 'regime']).size().min())
LINES = [(r, FIT) for r in ('acyclic', 'cyclic') if len(d[(d.regime == r) & (d.arm == FIT)])]
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
    for r, a in LINES:
        s = sub[(sub.regime == r) & (sub.arm == a)]
        if not len(s):
            continue
        g = s.groupby(xcol)[col]
        med = g.median().reindex(x).values
        lo, hi = g.quantile(.25).reindex(x).values, g.quantile(.75).reindex(x).values
        if scale == 'log':
            med, lo, hi = (np.clip(v, FLOOR, None) for v in (med, lo, hi))
        if a == FIT:
            ax.fill_between(x, lo, hi, color=RCOL[r], alpha=.13, lw=0, zorder=2)
        ax.plot(x, med, color=RCOL[r], marker=AMK[a], ls=ALS[a],
                lw=1.5 if a == FIT else 1.1, ms=8 if AMK[a] == '*' else 4.5,
                mec='white', mew=.6, zorder=3, alpha=1.0 if a == FIT else .75)
    ax.set_yscale('log' if scale == 'log' else 'linear')
    if scale == 'linear':
        ax.set_ylim(-0.05, 1.12)
    ax.set_xticks(x)
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))


def figure(mets, fname, h=1.72):
    fig, axes = plt.subplots(len(mets), 2, figsize=(7.2, h * len(mets)), squeeze=False)
    for r, col in enumerate(mets):
        lab, scale = SPEC[col]
        for c, (xcol, sub, xlab) in enumerate(SWEEPS):
            ax = axes[r][c]
            style(ax)
            draw(ax, sub, xcol, col, scale)
            if c == 0:
                ax.set_ylabel(lab, fontsize=9, color=INK)
            if r == 0:
                ax.set_title(xlab, fontsize=9, color=INK, pad=2)
            if r == len(mets) - 1:
                ax.set_xlabel(xcol, fontsize=9, color=INK2, labelpad=1)
            if c == 0:
                ax.yaxis.labelpad = 2
    hand = [Line2D([], [], color=RCOL[r], marker=AMK[a], ls=ALS[a],
                   lw=1.5 if a == FIT else 1.1, ms=8 if AMK[a] == '*' else 4.5,
                   mec='white', mew=.6, label=r'%s' % r)
            for r, a in LINES]
    # tight_layout first, then place title and legend flush against the resulting axes box,
    # so neither reserves a band of empty figure. savefig(bbox='tight') crops the rest.
    fig.tight_layout(pad=0.25, h_pad=0.55, w_pad=1.0)
    top = max(a.get_position().y1 for a in axes.ravel())
    bot = min(a.get_position().y0 for a in axes.ravel())
    fig.suptitle(r'$\mathbf{T}$ fitted to ground truth partition', fontsize=10, color=INK,
                 y=top + (0.030 if len(mets) < 3 else 0.017), va='bottom')
    fig.legend(handles=hand, loc='upper center', ncol=len(hand), frameon=False,
               fontsize=8.5, bbox_to_anchor=(.5, bot - (0.105 if len(mets) < 3 else 0.052)))
    for ext in ('png', 'pdf'):
        fig.savefig(f'{B}/figures/{fname}.{ext}', dpi=220)
    plt.close(fig)
    print(f'wrote figures/{fname}.png and .pdf')


figure(COMPACT, 'paper_tk_sweep_compact')
figure(['rae', 'recon', 'edge_recall', 'norm_Ahigh'], 'paper_tk_sweep', h=1.58)

piv = (d.groupby(['regime', 'p', 'k', 'arm'])
        .agg(rae=('rae', 'median'), recon=('recon', 'median'),
             recall=('edge_recall', 'median'), normA=('norm_Ahigh', 'median'),
             terr=('terr_mean', 'median'), sinks=('n_sinks', 'median')))
pd.set_option('display.width', 220)
print(piv.to_string(float_format=lambda v: f'{v:.2e}'))
