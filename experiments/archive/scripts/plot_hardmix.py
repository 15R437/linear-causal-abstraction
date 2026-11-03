"""Hard-loss smoke test, read visually.

Every panel shares one layout: three rows = the three hard-loss arms, and the BASELINE
(soft-only) is always the reference line -- zero for the paired-difference panels, dashed
for the absolute panels. So "left of the line" means the same thing everywhere.

Rows facet lambda_one-child, because the effect REVERSES between them and a pooled view
would cancel it out. Columns are the four questions: does it help exactness (dRAE), does it
help information loss (dIEM), does it survive the acceptance rule, and what is it doing to
cluster count (the mechanism).

Paired: arms share an init seed per (instance, lambda), so each dot is one instance's
difference against its own baseline, not a difference of medians.
"""
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
INK, INK2, GRID, REF = '#111111', '#555555', '#dcdcdc', '#8a8a8a'
ARMS = [('graph', '#e4572e'), ('recon', '#118a6e'), ('both', '#3d6bd6')]   # validated
ALPHA = 0.05
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 0.6, 'axes.edgecolor': '#888888',
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 0,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

d = pd.read_csv(f'{B}/data/hardmix.csv')
hm = d[d.method == 'HailMary']
occ = d[d.method == 'OCC'].set_index(['density', 'seed']).iem
base = hm[hm.arm == 'none'].set_index(['density', 'seed', 'lam_oc'])


def paired(lam, arm, metric):
    s = hm[(hm.arm == arm) & (hm.lam_oc == lam)].set_index(['density', 'seed', 'lam_oc'])
    idx = s.index.intersection(base.index)
    return (s.loc[idx, metric] - base.loc[idx, metric]).values


def hybrid(lam, arm):
    """Acceptance rate under the paper's rule: accept a run iff RAE <= ALPHA."""
    s = hm[(hm.arm == arm) & (hm.lam_oc == lam)]
    return float((s.rae <= ALPHA).mean())


LAMS = [0.0, 1000.0]
fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.5))
YP = {a: 2 - i for i, (a, _) in enumerate(ARMS)}     # graph on top

# Shared x-limits PER COLUMN, across both lambda rows. Per-panel autoscaling made the
# dIEM effect at lambda=1000 (median -0.50) look the same size as the one at lambda=0
# (median -0.01), which is the opposite of the finding: the effect reverses AND grows.
XL = {}
for kind in ('rae', 'iem'):
    m = max(np.abs(paired(l, a, kind)).max() for l in LAMS for a, _ in ARMS)
    XL[kind] = (-m * 1.08, m * 1.08)
XL['acc'] = (0, 1.05)
_kmed = [hm[(hm.arm == a) & (hm.lam_oc == l)].k_used.median()
         for l in LAMS for a in ('none',) + tuple(a for a, _ in ARMS)]
XL['k'] = (0, max(_kmed) * 1.18)   # scale off the plotted medians, not a raw outlier

for r, lam in enumerate(LAMS):
    for c, (kind, title) in enumerate([
            ('rae', r'$\Delta$ RAE  vs baseline'),
            ('iem', r'$\Delta$ information loss'),
            ('acc', r'accepted at $\alpha=0.05$'),
            ('k', 'clusters used')]):
        ax = axes[r, c]
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        ax.grid(True, axis='x', color=GRID, lw=0.45, zorder=0)
        ax.set_axisbelow(True)

        if kind in ('rae', 'iem'):
            ax.axvline(0, color=REF, lw=1.0, zorder=2)
            for arm, col in ARMS:
                v = paired(lam, arm, kind); y = YP[arm]
                jit = np.linspace(-0.17, 0.17, len(v))
                ax.scatter(v, y + jit, s=11, color=col, alpha=0.55, lw=0, zorder=3)
                ax.plot([np.median(v)] * 2, [y - 0.32, y + 0.32], color=col, lw=2.2,
                        solid_capstyle='butt', zorder=5)
            ax.set_xlim(*XL[kind])
        else:
            f = hybrid if kind == 'acc' else (lambda l, a: hm[(hm.arm == a) & (hm.lam_oc == l)].k_used.median())
            bval = f(lam, 'none')
            ax.axvline(bval, color=REF, lw=1.0, ls=(0, (3, 2)), zorder=2)
            for arm, col in ARMS:
                ax.barh(YP[arm], f(lam, arm), height=0.52, color=col, alpha=0.85,
                        lw=0, zorder=3)
            ax.set_xlim(*XL[kind])

        ax.set_ylim(-0.7, 2.7)
        ax.set_yticks([YP[a] for a, _ in ARMS])
        ax.set_yticklabels([a for a, _ in ARMS] if c == 0 else [])
        ax.tick_params(axis='y', labelsize=8, colors=INK, pad=1)
        ax.tick_params(axis='x', labelsize=7.5, colors=INK2)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        if r == 0:
            ax.set_title(title, fontsize=8.5, color=INK, pad=4)

    axes[r, 0].set_ylabel(r'$\lambda_{\mathrm{oc}}=%g$' % lam, fontsize=9.5, color=INK,
                          labelpad=6)

for c in range(2):
    axes[1, c].set_xlabel(r'$\leftarrow$ hard term better      worse $\rightarrow$',
                          fontsize=7, color=INK2, labelpad=2)
axes[1, 2].set_xlabel('fraction of runs', fontsize=7, color=INK2, labelpad=2)
axes[1, 3].set_xlabel(r'median $k_{\mathrm{used}}$', fontsize=7, color=INK2, labelpad=2)

handles = [Line2D([], [], color=c, lw=0, marker='o', ms=4.5, label='%s-hard' % a)
           for a, c in ARMS]
handles += [Line2D([], [], color=REF, lw=1.0, ls=(0, (3, 2)),
                   label='baseline (soft only)')]
fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=8,
           labelcolor=INK, handlelength=1.6, handletextpad=0.5, columnspacing=1.6,
           bbox_to_anchor=(0.5, -0.045))
fig.suptitle('Adding a hard (argmax) copy of the loss: paired effect on 9 instances',
             fontsize=9.5, color=INK, y=1.015)
fig.tight_layout(w_pad=0.9, h_pad=1.5)
fig.savefig(f'{B}/figures/hardmix.png', dpi=300)
fig.savefig(f'{B}/figures/hardmix.pdf')
print('wrote figures/hardmix.png (+pdf)')
for lam in LAMS:
    print('\nlam_oc=%g  baseline: acc %.2f  k_used %.0f  RAE %.3f  IEM %.3f' % (
        lam, hybrid(lam, 'none'), hm[(hm.arm=='none')&(hm.lam_oc==lam)].k_used.median(),
        hm[(hm.arm=='none')&(hm.lam_oc==lam)].rae.median(),
        hm[(hm.arm=='none')&(hm.lam_oc==lam)].iem.median()))
    for arm, _ in ARMS:
        v_r, v_i = paired(lam, arm, 'rae'), paired(lam, arm, 'iem')
        print('   %-6s dRAE %+.3f [%d/%d]  dIEM %+.3f [%d/%d]  acc %.2f  k %.0f' % (
            arm, np.median(v_r), (v_r < 0).sum(), (v_r > 0).sum(),
            np.median(v_i), (v_i < 0).sum(), (v_i > 0).sum(),
            hybrid(lam, arm), hm[(hm.arm==arm)&(hm.lam_oc==lam)].k_used.median()))
