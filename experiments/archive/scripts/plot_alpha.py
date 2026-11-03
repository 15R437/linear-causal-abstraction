"""Figure 1, simpler cut: "if you can tolerate error alpha, how much structure do you keep?"

x = alpha, the tolerance fixed in advance.  y = information loss of the composite
(accept the learned abstraction when RAE <= alpha AND its information loss is strictly
lower than AOC Closure's, else fall back).  Both axes are lower-is-better.  One line per method, one panel per density.  alpha -> 0 recovers
AOC Closure, so every line starts from it -- which makes the "never worse" property
visible rather than asserted.

Design: line chart (magnitude over an ordered control variable), categorical hues for
method identity, capped at the three all-pairs-validated slots plus secondary ink for
the null baseline.  Direct labels at the right edge, no legend box needed.
"""
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

B = '/Users/issy/Projects/linear-causal-abstraction'
SURFACE, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
COL = {'HailMary + fallback': '#2a78d6', 'UCA + fallback': '#1baf7a',
       'random + fallback': INK2, 'AOC Closure': '#eb6834'}
ALPHAS = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3])

d = pd.read_csv(f'{B}/data/../experiments/abstraction_study/data/frontier2.csv')
d = d[d.retained.notna()].copy()
dens = sorted(d.density.unique())

fig, axes = plt.subplots(1, len(dens), figsize=(3.3 * len(dens), 3.2), sharey=True)
fig.patch.set_facecolor(SURFACE)

for ax, dn in zip(np.atleast_1d(axes), dens):
    s = d[d.density == dn]
    occ = s[s.method == 'OCC'].set_index('seed').iem
    ax.set_facecolor(SURFACE)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color('#d8d7d2')
    ax.grid(True, color='#ececE8', lw=0.6); ax.set_axisbelow(True)

    # AOC Closure is flat in alpha -- it is the alpha=0 endpoint
    ax.axhline(occ.median(), color=COL['AOC Closure'], lw=2.4, zorder=6, ls=(0, (5, 2)))

    for nm, sub in (('HailMary + fallback', s[s.method == 'HailMary']),
                    ('UCA + fallback', s[s.method == 'UCA']),
                    ('random + fallback', s[s.method == 'random'])):
        ys = []
        for al in ALPHAS:
            per = []
            for sd, g in sub.groupby('seed'):
                base = occ.get(sd, np.nan)
                ok = g[(g.rae <= al) & (g.iem < base)]
                per.append(ok.iem.min() if len(ok) else base)
            ys.append(np.nanmedian(per))
        ax.plot(ALPHAS, ys, color=COL[nm], lw=2, solid_capstyle='round', zorder=4)
        ax.scatter(ALPHAS, ys, s=15, color=COL[nm], zorder=5)

    ax.set_xscale('log')
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r'tolerated abstraction error  $\alpha$', fontsize=9.5, color=INK2)
    ax.set_title(f'density {dn}   (n={occ.size})', fontsize=10, color=INK, pad=6)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3)

# direct labels on the last panel, no legend box
axL = np.atleast_1d(axes)[-1]
for nm, y in [(k, None) for k in COL]:
    pass
np.atleast_1d(axes)[0].set_ylabel('information loss  (lower is better)', fontsize=9.5, color=INK2)
s = d[d.density == dens[-1]]
occ = s[s.method == 'OCC'].set_index('seed').iem
ends = {'AOC Closure': occ.median()}
for nm, sub in (('HailMary + fallback', s[s.method == 'HailMary']),
                ('UCA + fallback', s[s.method == 'UCA']),
                ('random + fallback', s[s.method == 'random'])):
    per = []
    for sd, g in sub.groupby('seed'):
        base = occ.get(sd, np.nan)
        ok = g[(g.rae <= 0.3) & (g.iem < base)]
        per.append(ok.iem.min() if len(ok) else base)
    ends[nm] = np.nanmedian(per)
# HailMary and AOC Closure get direct labels; UCA and random lie ON the AOC Closure
# line (they never clear the tolerance, so the composite always falls back), so they
# share one label rather than three overlapping ones.
axL.annotate('HailMary + fallback', (0.3, ends['HailMary + fallback']),
             xytext=(0.62, ends['HailMary + fallback']), textcoords='data',
             color=COL['HailMary + fallback'], fontsize=9, va='center',
             annotation_clip=False, fontweight='medium')
y0 = ends['AOC Closure']
axL.annotate('AOC Closure', (0.3, y0), xytext=(0.62, y0 - 0.045), textcoords='data',
             color=COL['AOC Closure'], fontsize=9, va='center', annotation_clip=False,
             fontweight='medium',
             arrowprops=dict(arrowstyle='-', color=COL['AOC Closure'], lw=0.7,
                             shrinkA=1, shrinkB=1))
axL.annotate('UCA / random\n(never accepted,\nso always fall back)', (0.3, y0),
             xytext=(0.62, y0 + 0.105), textcoords='data', color=INK2, fontsize=8,
             va='center', annotation_clip=False,
             arrowprops=dict(arrowstyle='-', color=INK2, lw=0.7, shrinkA=1, shrinkB=1))

fig.suptitle('Fix a tolerance in advance; the composite is never worse than AOC Closure'
             '   (both axes lower-is-better)', fontsize=10.5, color=INK, y=1.02)
fig.tight_layout(rect=[0, 0, 0.87, 0.97])
out = f'{B}/experiments/abstraction_study/figures/alpha_curve.png'
fig.savefig(out, dpi=220, facecolor=SURFACE, bbox_inches='tight')
fig.savefig(out.replace('.png', '.pdf'), facecolor=SURFACE, bbox_inches='tight')
print('wrote', out)
for dn in dens:
    s = d[d.density == dn]; occ = s[s.method == 'OCC'].set_index('seed').iem
    row = []
    for al in (1e-3, 1e-2, 0.05, 0.3):
        per = []
        for sd, g in s[s.method == 'HailMary'].groupby('seed'):
            base = occ.get(sd, np.nan)
            ok = g[(g.rae <= al) & (g.iem < base)]
            per.append(ok.iem.min() if len(ok) else base)
        row.append('a=%-5g %.3f' % (al, np.nanmedian(per)))
    print('d=%.1f  OCC %.3f   %s' % (dn, occ.median(), '   '.join(row)))
