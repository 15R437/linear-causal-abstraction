"""The alpha-vs-information curve for the recommended algorithm.

    HailMary:  closure init with slack,  one_child_loss,  recon_loss or info_loss
    plus: accept the learned partition when its measured relative error <= alpha AND its
          info_eff improves on OCC's, otherwise fall back to OCC. BOTH clauses matter --
          error alone accepts a worse partition on 12-21% of instances.

x-axis: alpha, the tolerated relative observation error (fraction of high-level variance
        that is abstraction error -- computable at inference time, no ground truth needed).
y-axis: effective information loss (intra-cluster mass + mass on marginalised nodes).
alpha = 0 is OCC itself.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B = '/Users/issy/Projects/linear-causal-abstraction/experiments/abstraction_study'
src = sys.argv[1] if len(sys.argv) > 1 else 'slack_nograph_robust.csv'
d = pd.read_csv(f'{B}/data/{src}')
key = ['p', 'density', 'seed']
occ = d[d.arm == 'OCC'].set_index(key).info_eff
h = d[d.arm != 'OCC']
ALPHAS = np.concatenate([[0.0], np.logspace(-4, 0, 40)])


def curve(g):
    gg = g.set_index(key)
    base = occ.reindex(gg.index)
    out = []
    for a in ALPHAS:
        ok = ((gg.rel_err <= a) & (gg.info_eff < base)) if a > 0 else (gg.valid == 1)
        out.append((np.nanmean(np.where(ok, gg.info_eff, base)), ok.mean()))
    return np.array(out)


plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
COL = {'info': '#d62728', 'recon': '#1f77b4'}

# panel 1: best slack per arm
for arm in ('info', 'recon'):
    g = h[h.arm == arm]
    best_sl = g.groupby('slack').info_eff.mean().idxmin()
    c = curve(g[g.slack == best_sl])
    ax[0].plot(ALPHAS[1:], c[1:, 0], '-', color=COL[arm], lw=2,
               label=f'{arm}  (slack {best_sl:g})')
ax[0].axhline(occ.mean(), color='k', ls='--', lw=1.4)
ax[0].text(1.2e-4, occ.mean() + .006, 'OCC  (alpha = 0)', fontsize=8)
ax[0].set_xscale('log'); ax[0].set_xlabel('alpha  (tolerated relative error)')
ax[0].set_ylabel('effective information loss')
ax[0].legend(fontsize=8, loc='lower left')
ax[0].set_title('the trade-off', fontsize=10)

# panel 2: slack matters
g = h[h.arm == 'info']
for sl, sub in g.groupby('slack'):
    c = curve(sub)
    ax[1].plot(ALPHAS[1:], c[1:, 0], '-', lw=1.7, label=f'slack {sl:g}')
ax[1].axhline(occ.mean(), color='k', ls='--', lw=1.4)
ax[1].set_xscale('log'); ax[1].set_xlabel('alpha'); ax[1].legend(fontsize=7)
ax[1].set_title('info arm: extra clusters help', fontsize=10)

# panel 3: what fraction of instances clear the tolerance
for arm in ('info', 'recon'):
    g = h[h.arm == arm]
    best_sl = g.groupby('slack').info_eff.mean().idxmin()
    c = curve(g[g.slack == best_sl])
    ax[2].plot(ALPHAS[1:], c[1:, 1], '-', color=COL[arm], lw=2, label=arm)
ax[2].set_xscale('log'); ax[2].set_xlabel('alpha'); ax[2].set_ylim(0, 1.04)
ax[2].set_ylabel('fraction accepted (rest fall back to OCC)')
ax[2].legend(fontsize=8); ax[2].set_title('acceptance rate', fontsize=10)

fig.suptitle('Tolerate a little observational error, retain much more structure', y=1.03)
fig.savefig(f'{B}/figures/15_alpha_curve.png')
print('wrote figures/15_alpha_curve.png')

print('\nTABLE (hybrid: accept if rel_err <= alpha, else OCC)')
print('  OCC alone: info_eff %.3f\n' % occ.mean())
print('  %-7s %-6s %-6s %-10s %s' % ('alpha', 'arm', 'slack', 'info_eff', 'accepted'))
for a in (0.001, 0.01, 0.05, 0.10, 0.25):
    rows = []
    for (arm, sl), g in h.groupby(['arm', 'slack']):
        gg = g.set_index(key); bs = occ.reindex(gg.index)
        ok = (gg.rel_err <= a) & (gg.info_eff < bs)
        v = float(np.nanmean(np.where(ok, gg.info_eff, occ.reindex(gg.index))))
        rows.append((arm, sl, v, ok.mean()))
    rows.sort(key=lambda r: r[2])
    arm, sl, v, sh = rows[0]
    print('  %-7s %-6s %-6g %.3f      %.2f   (%+.0f%% vs OCC)'
          % (a, arm, sl, v, sh, 100 * (v - occ.mean()) / occ.mean()))
