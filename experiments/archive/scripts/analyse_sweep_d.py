"""Summarise sweep D: the reconstruction-vs-surjective question on CONNECTED graphs.

Headline is `good` = valid AND surjective, as asked. graph_ok and high-level
connectivity are reported alongside so it is visible whether they track or diverge.

Run:  venv/bin/python experiments/abstraction_study/scripts/analyse_sweep_d.py <results_dir>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8983'
GRID, SURFACE = '#e8e7e3', '#fcfcfb'
HILITE, PLAIN = '#2a78d6', '#c9c8c2'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.linewidth': 0.8, 'axes.labelcolor': INK2,
    'axes.titlecolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.labelsize': 9, 'axes.titlesize': 10,
    'grid.color': GRID, 'grid.linewidth': 0.7, 'grid.linestyle': '-',
    'legend.frameon': False, 'legend.fontsize': 8, 'font.size': 9,
})

VALID_TOL = 1e-8
CONTROL = ('hard_count', 1.0, 10.0)      # the original code, verbatim


def arm_key(r):
    return (r['surj_mode'], r['lambda_surj'], r['lambda_recon'])


def arm_label(mode, lam, lrec):
    if (mode, lam, lrec) == CONTROL:
        return 'ORIGINAL (recon 10, hard count)'
    if mode == 'none':
        return f"no surjective term, recon {lrec:g}"
    return f"{mode} λ={lam:g}, recon {lrec:g}"


def add_metrics(df):
    df['valid'] = ((df.obs_loss < VALID_TOL) & (df.intv_loss < VALID_TOL)).astype(int)
    df['good'] = (df.valid.astype(bool) & (df.surjective == 1)).astype(int)
    df['graph_ok'] = (df.graph_loss < VALID_TOL).astype(int)
    df['connected_high'] = ((df.empty_high_dag == 0) & (df.isolated_nonempty == 0)).astype(int)
    df['strict'] = (df.good.astype(bool) & df.graph_ok.astype(bool)
                    & df.connected_high.astype(bool)).astype(int)
    df['arm'] = [arm_label(*arm_key(r)) for _, r in df.iterrows()]
    df['cell'] = df.graph_type + '  p=' + df.n_low.astype(str) + ', k=' + df.n_high.astype(str)
    return df


def tidy(ax, ylabel=None, title=None):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc='left', pad=8)


def paired(df, metric, control_label):
    """Per-seed difference against the original configuration."""
    keys = ['graph_type', 'n_low', 'n_high', 'trial']
    out = []
    for cell, g in df.groupby('cell'):
        ctrl = g[g.arm == control_label].set_index(keys)[metric]
        for arm, ga in g.groupby('arm'):
            if arm == control_label:
                continue
            x = ga.set_index(keys)[metric]
            common = ctrl.index.intersection(x.index)
            d = (x.loc[common] - ctrl.loc[common]).dropna()
            if not len(d):
                continue
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
            out.append({'cell': cell, 'arm': arm, 'metric': metric, 'delta': d.mean(),
                        'ci_lo': d.mean() - 1.96 * se, 'ci_hi': d.mean() + 1.96 * se,
                        'n_pairs': len(d)})
    r = pd.DataFrame(out)
    if len(r):
        r['verdict'] = np.where(r.ci_lo > 0, 'better',
                                np.where(r.ci_hi < 0, 'worse', 'no difference'))
    return r


def figure_rates(df, outdir, control_label):
    cells = sorted(df.cell.unique())
    order = ([control_label] +
             sorted(a for a in df.arm.unique() if a != control_label))
    metrics = [('good', 'valid AND surjective'), ('valid', 'valid'),
               ('surjective', 'surjective'), ('graph_ok', 'graph condition holds')]

    fig, axes = plt.subplots(len(metrics), len(cells),
                             figsize=(3.6 * len(cells), 2.5 * len(metrics)), squeeze=False)
    for j, cell in enumerate(cells):
        sub = df[df.cell == cell]
        for i, (metric, label) in enumerate(metrics):
            ax = axes[i][j]
            g = sub.groupby('arm')[metric].agg(['mean', 'sem']).reindex(order).dropna(how='all')
            xs = np.arange(len(g))
            colours = [HILITE if k == control_label else PLAIN for k in g.index]
            ax.bar(xs, g['mean'], width=0.72, color=colours, zorder=3)
            ax.errorbar(xs, g['mean'], yerr=1.96 * g['sem'].fillna(0), fmt='none',
                        ecolor=INK2, elinewidth=1.1, capsize=0, zorder=4)
            ax.set_xticks(xs)
            ax.set_xticklabels(list(g.index) if i == len(metrics) - 1 else [''] * len(g),
                               rotation=40, ha='right', fontsize=7)
            ax.set_ylim(0, 1.05)
            tidy(ax, ylabel=label if j == 0 else None, title=cell if i == 0 else None)
    fig.suptitle('Connected graphs only. Blue = the original configuration. '
                 f'{df.trial.nunique()} seeds, 95% CI',
                 x=0.01, ha='left', fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / 'figD1_rates.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def figure_delta(df, outdir, control_label):
    cells = sorted(df.cell.unique())
    d = paired(df, 'good', control_label)
    arms = sorted(a for a in df.arm.unique() if a != control_label)
    fig, axes = plt.subplots(1, len(cells), figsize=(3.5 * len(cells), 4.4),
                             squeeze=False, sharex=True)
    for j, cell in enumerate(cells):
        ax = axes[0][j]
        ax.axvline(0, color=MUTED, lw=1)
        sub = d[d.cell == cell].set_index('arm')
        for y, arm in enumerate(arms):
            if arm not in sub.index:
                continue
            r = sub.loc[arm]
            ax.plot([r.ci_lo, r.ci_hi], [y, y], color=HILITE, lw=1.6, alpha=0.55,
                    solid_capstyle='round')
            ax.plot([r.delta], [y], 'o', color=HILITE, ms=6, mew=1.5, mec=SURFACE, zorder=3)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels(arms if j == 0 else [''] * len(arms), fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel('Δ valid AND surjective')
        tidy(ax, title=cell)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True)
    fig.suptitle('Paired change against the original configuration, 95% CI',
                 x=0.01, ha='left', fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outdir / 'figD2_paired_delta.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    results_dir = Path(sys.argv[1])
    outdir = results_dir / 'plots'
    outdir.mkdir(parents=True, exist_ok=True)
    df = add_metrics(pd.read_csv(results_dir / 'sweep_d.csv'))
    control_label = arm_label(*CONTROL)
    pd.set_option('display.width', 240)
    pd.set_option('display.max_rows', 300)

    print(f"{len(df)} runs, {df.trial.nunique()} seeds, "
          f"postprocess failures: {int(df.postprocess_failed.sum())}\n")

    cols = ['good', 'valid', 'surjective', 'graph_ok', 'connected_high', 'strict',
            'graph_loss', 'edges_high', 'edges_high_gt', 'runtime']
    print("=== by graph family and arm (headline column is `good`) ===")
    for gt, g in df.groupby('graph_type'):
        print(f"\n--- {gt} ---")
        t = g.groupby('arm')[cols].mean().sort_values('good', ascending=False)
        print(t.round(3).to_string())

    print("\n\n=== paired vs the original configuration, on `good` ===")
    d = paired(df, 'good', control_label)
    sig = d[d.verdict != 'no difference']
    print(sig.round(3).to_string(index=False) if len(sig)
          else "  nothing separates from the original at 95% CI")

    print("\n=== same, on graph_ok ===")
    dg = paired(df, 'graph_ok', control_label)
    sig = dg[dg.verdict != 'no difference']
    print(sig.round(3).to_string(index=False) if len(sig)
          else "  nothing separates from the original at 95% CI")

    pd.concat([d, dg]).to_csv(outdir / 'paired_deltas_d.csv', index=False)
    figure_rates(df, outdir, control_label)
    figure_delta(df, outdir, control_label)
    print(f"\nwrote figures to {outdir}")
