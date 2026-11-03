"""Final confirmatory analysis: which HailMary configuration to adopt.

Success is the pair of sufficient conditions (All-In via graph_loss, plus one-child),
since those guarantee an abstraction exists. Utility is information loss. Surjectivity
is reported but is not a target. Everything is paired on the problem seed.

Run:  venv/bin/python experiments/abstraction_study/scripts/analyse_final.py <results_dir>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, INK2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#8a8983', '#e8e7e3', '#fcfcfb'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.linewidth': 0.8, 'axes.labelcolor': INK2,
    'axes.titlecolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.labelsize': 9,
    'grid.color': GRID, 'grid.linewidth': 0.7, 'legend.frameon': False, 'font.size': 9,
})

ARM_ORDER = ['ORIGINAL', 'REC_recon', 'REC_recon_clo', 'REC_intra_clo', 'CLOSURE_ONLY']
ARM_LABEL = {
    'ORIGINAL':      'ORIGINAL (recon, hard_count surj, old norm)',
    'REC_recon':     'recon + graph_const, no surj, random init',
    'REC_recon_clo': 'recon + graph_const, no surj, CLOSURE init',
    'REC_intra_clo': 'intra_mass + graph_const, no surj, CLOSURE init',
    'CLOSURE_ONLY':  'CLOSURE ONLY (no training, ~1ms)',
}
CONTROL = 'ORIGINAL'


def add_metrics(df):
    df['graph_ok'] = (df.graph_loss < 1e-8).astype(int)
    df['success'] = (df.graph_ok.astype(bool) & (df.one_child_ok == 1)).astype(int)
    df['valid'] = ((df.obs_loss < 1e-8) & (df.intv_loss < 1e-8)).astype(int)
    df['cell'] = (df.graph_type + '  p=' + df.n_low.astype(str)
                  + ',k=' + df.n_high.astype(str))
    return df


def best_of_n(df):
    """Per (cell, arm, problem): did ANY restart succeed, and among successes what is the
    lowest information loss? Selection uses graph_loss/one_child then intra_mass -- all
    computable without ground truth, so this is a legitimate inference-time procedure."""
    rows = []
    for (cell, arm, trial), g in df.groupby(['cell', 'arm_name', 'trial']):
        ok = g[g.success == 1]
        rows.append({
            'cell': cell, 'arm_name': arm, 'trial': trial,
            'n_restarts': len(g),
            'any_success': int(len(ok) > 0),
            'single_rate': g.success.mean(),
            'sel_intra': ok.intra_mass.min() if len(ok) else np.nan,
            'sel_block': (ok.loc[ok.intra_mass.idxmin(), 'block_pred_err']
                          if len(ok) else np.nan),
            'sel_maxclust': (ok.loc[ok.intra_mass.idxmin(), 'max_cluster']
                             if len(ok) else np.nan),
            'sel_clusters': (ok.loc[ok.intra_mass.idxmin(), 'n_clusters_used']
                             if len(ok) else np.nan),
            'sel_edges': (ok.loc[ok.intra_mass.idxmin(), 'edges_high'] if len(ok) else np.nan),
            'gt_maxclust': g.gt_max_cluster.iloc[0],
            'gt_edges': g.edges_high_gt.iloc[0],
            'runtime_total': g.runtime.sum(),
        })
    return pd.DataFrame(rows)


def paired(bo, metric, higher_better=True):
    out = []
    for cell, g in bo.groupby('cell'):
        ctrl = g[g.arm_name == CONTROL].set_index('trial')[metric]
        for arm, ga in g.groupby('arm_name'):
            if arm == CONTROL:
                continue
            x = ga.set_index('trial')[metric]
            common = ctrl.index.intersection(x.index)
            d = (x.loc[common] - ctrl.loc[common]).dropna()
            if len(d) < 2:
                continue
            se = d.std(ddof=1) / np.sqrt(len(d))
            lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
            v = ('better' if (lo > 0 if higher_better else hi < 0)
                 else 'worse' if (hi < 0 if higher_better else lo > 0) else 'ns')
            out.append({'cell': cell, 'arm_name': arm, 'metric': metric,
                        'delta': d.mean(), 'lo': lo, 'hi': hi, 'n': len(d), 'verdict': v})
    return pd.DataFrame(out)


def figure_summary(bo, outdir):
    cells = sorted(bo.cell.unique())
    arms = [a for a in ARM_ORDER if a in set(bo.arm_name)]
    metrics = [('any_success', 'success rate, best-of-N', True),
               ('single_rate', 'success rate, single run', True),
               ('sel_intra', 'information loss (selected)', False)]
    fig, axes = plt.subplots(len(metrics), len(cells),
                             figsize=(3.3 * len(cells), 2.7 * len(metrics)), squeeze=False)
    for j, cell in enumerate(cells):
        sub = bo[bo.cell == cell]
        for i, (m, label, up) in enumerate(metrics):
            ax = axes[i][j]
            g = sub.groupby('arm_name')[m].agg(['mean', 'sem']).reindex(arms)
            xs = np.arange(len(arms))
            colours = [SERIES[k % len(SERIES)] for k in range(len(arms))]
            ax.bar(xs, g['mean'], width=0.72, color=colours, zorder=3)
            ax.errorbar(xs, g['mean'], yerr=1.96 * g['sem'].fillna(0), fmt='none',
                        ecolor=INK2, elinewidth=1.1, capsize=0, zorder=4)
            ax.set_xticks(xs)
            ax.set_xticklabels(arms if i == len(metrics) - 1 else [''] * len(arms),
                               rotation=35, ha='right', fontsize=7)
            if m != 'sel_intra':
                ax.set_ylim(0, 1.05)
            ax.set_axisbelow(True); ax.yaxis.grid(True); ax.xaxis.grid(False)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            if j == 0:
                ax.set_ylabel(label)
            if i == 0:
                ax.set_title(cell, loc='left', pad=8, fontsize=10)
    fig.suptitle('Final comparison. Success = All-In AND one-child. '
                 'Information loss = fraction of edge mass destroyed (lower better).',
                 x=0.01, ha='left', fontsize=11, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / 'final_summary.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    rd = Path(sys.argv[1])
    outdir = rd / 'plots'
    outdir.mkdir(parents=True, exist_ok=True)
    d = add_metrics(pd.read_csv(rd / 'sweep_r.csv'))
    pd.set_option('display.width', 260)
    pd.set_option('display.max_rows', 400)

    print(f"{len(d)} runs | problems/cell: "
          f"{d.groupby('cell').trial.nunique().to_dict()} | restarts: "
          f"{sorted(d.restart.unique())}\n")

    bo = best_of_n(d)
    cols = ['any_success', 'single_rate', 'sel_intra', 'sel_block', 'sel_maxclust',
            'gt_maxclust', 'sel_clusters', 'sel_edges', 'gt_edges', 'runtime_total']
    for cell, g in bo.groupby('cell'):
        print(f"===== {cell}  ({g.trial.nunique()} problems, "
              f"{int(g.n_restarts.mean())} restarts) =====")
        t = g.groupby('arm_name')[cols].mean().reindex(
            [a for a in ARM_ORDER if a in set(g.arm_name)])
        print(t.round(3).to_string())
        print()

    print("\n===== paired vs ORIGINAL, best-of-N success =====")
    ps = paired(bo, 'any_success', True)
    print(ps.round(3).to_string(index=False) if len(ps) else "n/a")
    print("\n===== paired vs ORIGINAL, information loss (lower better) =====")
    pi = paired(bo, 'sel_intra', False)
    print(pi.round(3).to_string(index=False) if len(pi) else "n/a")

    bo.to_csv(outdir / 'best_of_n.csv', index=False)
    pd.concat([ps, pi]).to_csv(outdir / 'paired_final.csv', index=False)
    figure_summary(bo, outdir)
    print(f"\nwrote {outdir}")
