"""Summarise the surjective-loss / initialisation smoke test.

Metrics, in the order they matter:

  valid       obs_loss < 1e-8 AND intv_loss < 1e-8 -- the abstraction conditions hold.
              There is no unique valid abstraction for a given low-level model, so this
              asks whether what we found is admissible, not whether it matches T_gt.
  surjective  every high-level cluster has at least one member.
  good        valid AND surjective. Reported as the headline because validity alone is
              gameable: collapsing every node into one cluster is a trivially valid
              abstraction that marginalises almost everything away.
  recon_loss  Frobenius reconstruction error of A_low, relative to ||A_low||_F.

Every comparison is paired on the trial seed, because the graph generator is wildly
heterogeneous at a fixed density (edge counts span an order of magnitude).

Run:  venv/bin/python experiments/abstraction_study/scripts/analyse_smoke_test.py <results_dir>
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Categorical palette, validated adjacent-pair ordering (light mode).
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
          '#4a3aa7', '#e34948']
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8983'
GRID, SURFACE = '#e8e7e3', '#fcfcfb'
HILITE, PLAIN = '#2a78d6', '#c9c8c2'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.linewidth': 0.8,
    'axes.labelcolor': INK2, 'axes.titlecolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.labelsize': 9, 'axes.titlesize': 10,
    'grid.color': GRID, 'grid.linewidth': 0.7, 'grid.linestyle': '-',
    'legend.frameon': False, 'legend.fontsize': 8,
    'font.size': 9, 'savefig.facecolor': SURFACE,
})

MODE_ORDER = ['topk_barrier', 'empty_prob', 'hinge_max', 'marginal_entropy',
              'hinge_colsum', 'hard_count']
MODE_LABEL = {'hinge_max': 'hinge on column max', 'empty_prob': 'expected empty columns',
              'hinge_colsum': 'hinge on column sums', 'marginal_entropy': 'marginal entropy (KL)',
              'topk_barrier': 'top-2 log barrier', 'hard_count': 'hard count (current)',
              'none': 'no surjective term'}
INIT_ORDER = ['reg_surjective', 'random_hard', 'random_normal', 'near_uniform',
              'uniform', 'biased_0.5', 'biased_0.8', 'biased_0.95']
INIT_LABEL = {'reg_surjective': 'reg. surjective\n(current)', 'random_hard': 'random hard',
              'random_normal': 'random normal', 'near_uniform': 'near uniform',
              'uniform': 'uniform', 'biased_0.5': 'biased p=0.5',
              'biased_0.8': 'biased p=0.8', 'biased_0.95': 'biased p=0.95'}

VALID_TOL = 1e-8


def add_metrics(df):
    df['valid'] = ((df.obs_loss < VALID_TOL) & (df.intv_loss < VALID_TOL)).astype(int)
    df['good'] = (df.valid.astype(bool) & (df.surjective == 1)).astype(int)
    df['log_obs'] = np.log10(df.obs_loss + 1e-18)
    df['log_intv'] = np.log10(df.intv_loss + 1e-18)
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


def cells(df):
    return sorted(df[['n_low', 'n_high', 'density']].drop_duplicates()
                  .itertuples(index=False, name=None), key=lambda c: (c[2], c[0]))


def cell_title(c):
    return f"p={c[0]}, k={c[1]}, density={c[2]}"


def paired_delta(df, control_mask, arm_mask, metric,
                 keys=('n_low', 'n_high', 'density', 'trial')):
    ctrl = df[control_mask].set_index(list(keys))[metric]
    arm = df[arm_mask].set_index(list(keys))[metric]
    common = ctrl.index.intersection(arm.index)
    d = (arm.loc[common] - ctrl.loc[common]).dropna()
    if len(d) == 0:
        return np.nan, np.nan, np.nan, 0
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    return d.mean(), d.mean() - 1.96 * se, d.mean() + 1.96 * se, len(d)


# ---------------------------------------------------------------------------
# Sweep A - surjective loss formulations
# ---------------------------------------------------------------------------

def figure_surjective_curves(a, outdir):
    """The headline: rate of valid AND surjective solutions against lambda_surj."""
    panels = cells(a)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.7),
                             squeeze=False, sharey=True)

    for j, c in enumerate(panels):
        ax = axes[0][j]
        cell = a[(a.n_low == c[0]) & (a.n_high == c[1]) & (a.density == c[2])]
        base = cell[(cell.surj_mode == 'hard_count') & (cell.lambda_recon == 10)]
        norecon = cell[(cell.surj_mode == 'none') & (cell.lambda_recon == 0)]

        for m_i, mode in enumerate(MODE_ORDER):
            sub = (cell[(cell.surj_mode == mode) & (cell.lambda_recon == 0)]
                   .groupby('lambda_surj')['good'].mean())
            if sub.empty:
                continue
            ax.plot(sub.index, sub.values, color=SERIES[m_i], lw=2, marker='o', ms=4.5,
                    mew=1.5, mec=SURFACE, label=MODE_LABEL[mode], zorder=3)
        if not base.empty:
            ax.axhline(base['good'].mean(), color=INK, lw=1.4, zorder=2)
        if not norecon.empty:
            ax.axhline(norecon['good'].mean(), color=MUTED, lw=1.4, zorder=2)
        ax.set_xscale('log')
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel(r'$\lambda_{surj}$')
        tidy(ax, ylabel='valid AND surjective (rate)' if j == 0 else None, title=cell_title(c))

    handles, labels = axes[0][0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color=INK, lw=1.4), plt.Line2D([], [], color=MUTED, lw=1.4)]
    labels += ['current default (recon on)', 'recon off, no surjective term']
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.15))
    fig.suptitle('Replacing the reconstruction loss with a surjective loss '
                 r'($\lambda_{recon}=0$): admissible non-degenerate solutions, 10 paired seeds',
                 x=0.01, ha='left', fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outdir / 'fig1_surjective_lambda_curves.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def figure_component_heatmap(a, outdir):
    """Validity and surjectivity separately, so it is visible which one each arm buys.
    Several arms sit at exactly 1.0, so lines would occlude each other here."""
    panels = cells(a)
    lambdas = sorted(l for l in a[a.lambda_recon == 0].lambda_surj.unique() if l > 0)
    rows = [('current default (recon on)', 'hard_count', 1.0, 10.0),
            ('recon off, no surjective', 'none', 0.0, 0.0)]
    rows += [(f"{MODE_LABEL[m]}  $\\lambda$={l:g}", m, l, 0.0)
             for m in MODE_ORDER for l in lambdas]

    metrics = [('valid', 'valid abstraction\n(obs & intv < 1e-8)'),
               ('surjective', 'surjective\n(all k clusters used)'),
               ('good', 'both')]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'blues', ['#f2f6fc', '#cde2fb', '#9ec5f4', '#5598e7', '#2a78d6', '#184f95'])

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(1.25 * len(panels) * len(metrics) + 2.6,
                                      0.30 * len(rows) + 2.2), squeeze=False)
    for m_i, (metric, mlabel) in enumerate(metrics):
        ax = axes[0][m_i]
        grid = np.full((len(rows), len(panels)), np.nan)
        for i, (_, mode, lam, lrec) in enumerate(rows):
            for j, c in enumerate(panels):
                sel = a[(a.n_low == c[0]) & (a.n_high == c[1]) & (a.density == c[2])
                        & (a.surj_mode == mode) & (a.lambda_surj == lam)
                        & (a.lambda_recon == lrec)]
                if not sel.empty:
                    grid[i, j] = sel[metric].mean()
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect='auto')
        for i in range(len(rows)):
            for j in range(len(panels)):
                if np.isnan(grid[i, j]):
                    continue
                ax.text(j, i, f"{grid[i, j]:.1f}", ha='center', va='center', fontsize=7,
                        color='#ffffff' if grid[i, j] > 0.62 else INK)
        ax.set_xticks(range(len(panels)))
        ax.set_xticklabels([f"p={c[0]}\nk={c[1]}\nd={c[2]}" for c in panels], fontsize=7)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows] if m_i == 0 else [''] * len(rows), fontsize=7.5)
        ax.axhline(1.5, color=SURFACE, lw=3)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(mlabel, loc='left', pad=8, fontsize=9)
        if m_i == len(metrics) - 1:
            cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
            cb.set_label('rate over 10 seeds', fontsize=8)
            cb.outline.set_visible(False)

    fig.suptitle('What each surjective formulation actually buys',
                 x=0.01, ha='left', fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / 'fig1b_component_heatmap.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def figure_surjective_delta(a, outdir):
    panels = cells(a)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 5.4),
                             squeeze=False, sharex=True)
    lambdas = sorted(l for l in a[a.lambda_recon == 0].lambda_surj.unique() if l > 0)

    for j, c in enumerate(panels):
        ax = axes[0][j]
        cell = a[(a.n_low == c[0]) & (a.n_high == c[1]) & (a.density == c[2])]
        ctrl = (cell.surj_mode == 'hard_count') & (cell.lambda_recon == 10)
        rows, ticks = [], []
        y = 0
        for m_i, mode in enumerate(MODE_ORDER):
            for lam in lambdas:
                arm = ((cell.surj_mode == mode) & (cell.lambda_recon == 0)
                       & (cell.lambda_surj == lam))
                if not arm.any():
                    continue
                mean, lo, hi, n = paired_delta(cell, ctrl, arm, 'good')
                rows.append((y, mean, lo, hi, SERIES[m_i]))
                ticks.append((y, f"{MODE_LABEL[mode]}  $\\lambda$={lam:g}"))
                y += 1
            y += 0.6
        arm = (cell.surj_mode == 'none') & (cell.lambda_recon == 0)
        mean, lo, hi, n = paired_delta(cell, ctrl, arm, 'good')
        rows.append((y, mean, lo, hi, INK))
        ticks.append((y, 'recon off, no surjective term'))

        ax.axvline(0, color=MUTED, lw=1)
        for yy, mean, lo, hi, colour in rows:
            ax.plot([lo, hi], [yy, yy], color=colour, lw=1.6, solid_capstyle='round', alpha=0.55)
            ax.plot([mean], [yy], 'o', color=colour, ms=6, mew=1.5, mec=SURFACE, zorder=3)
        ax.set_yticks([t[0] for t in ticks])
        ax.set_yticklabels([t[1] for t in ticks] if j == 0 else [''] * len(ticks), fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel('Δ rate of valid AND surjective')
        tidy(ax, title=cell_title(c))
        ax.yaxis.grid(False)
        ax.xaxis.grid(True)

    fig.suptitle('Paired change against the current default, 95% CI '
                 '(right of the rule = better)',
                 x=0.01, ha='left', fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / 'fig2_surjective_paired_delta.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sweep B - initialisation
# ---------------------------------------------------------------------------

def figure_inits(b, outdir):
    panels = cells(b)
    metrics = [('valid', 'valid abstraction (rate)'),
               ('surjective', 'surjective (rate)'),
               ('good', 'valid AND surjective (rate)')]
    fig, axes = plt.subplots(len(metrics), len(panels),
                             figsize=(3.3 * len(panels), 2.7 * len(metrics)), squeeze=False)

    for j, c in enumerate(panels):
        cell = b[(b.n_low == c[0]) & (b.n_high == c[1]) & (b.density == c[2])]
        for i, (metric, label) in enumerate(metrics):
            ax = axes[i][j]
            g = cell.groupby('init')[metric].agg(['mean', 'sem']).reindex(INIT_ORDER).dropna(how='all')
            xs = np.arange(len(g))
            colours = [HILITE if k == 'reg_surjective' else PLAIN for k in g.index]
            ax.bar(xs, g['mean'], width=0.72, color=colours, zorder=3)
            ax.errorbar(xs, g['mean'], yerr=1.96 * g['sem'].fillna(0), fmt='none',
                        ecolor=INK2, elinewidth=1.1, capsize=0, zorder=4)
            ax.set_xticks(xs)
            ax.set_xticklabels([INIT_LABEL[k] for k in g.index] if i == len(metrics) - 1
                               else [''] * len(g), rotation=45, ha='right', fontsize=7)
            ax.set_ylim(0, 1.05)
            tidy(ax, ylabel=label if j == 0 else None,
                 title=cell_title(c) if i == 0 else None)

    fig.suptitle('Initialisation under the current default losses '
                 '(blue = what the code does today), 10 seeds, 95% CI.\n'
                 'Note the uniform start: valid but never surjective - it collapses to one cluster.',
                 x=0.01, ha='left', fontsize=11, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outdir / 'fig3_init_strategies.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def figure_init_delta(b, outdir):
    panels = cells(b)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.4),
                             squeeze=False, sharex=True)
    arms = [k for k in INIT_ORDER if k != 'reg_surjective']
    for j, c in enumerate(panels):
        ax = axes[0][j]
        cell = b[(b.n_low == c[0]) & (b.n_high == c[1]) & (b.density == c[2])]
        ctrl = cell.init == 'reg_surjective'
        ax.axvline(0, color=MUTED, lw=1)
        for y, init in enumerate(arms):
            mean, lo, hi, n = paired_delta(cell, ctrl, cell.init == init, 'good')
            ax.plot([lo, hi], [y, y], color=HILITE, lw=1.6, alpha=0.55, solid_capstyle='round')
            ax.plot([mean], [y], 'o', color=HILITE, ms=6, mew=1.5, mec=SURFACE, zorder=3)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([INIT_LABEL[k].replace('\n', ' ') for k in arms]
                           if j == 0 else [''] * len(arms), fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Δ rate of valid AND surjective')
        tidy(ax, title=cell_title(c))
        ax.yaxis.grid(False)
        ax.xaxis.grid(True)
    fig.suptitle('Paired change by initialisation against the current reg. surjective start, 95% CI',
                 x=0.01, ha='left', fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(outdir / 'fig4_init_paired_delta.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sweep C - does the recommended configuration survive a bad initialisation?
# ---------------------------------------------------------------------------

def arm_label(mode, lam, lrec):
    if lrec > 0:
        return 'current default'
    return f"{MODE_LABEL[mode]}, recon off"


def figure_interaction(c_df, outdir):
    panels = cells(c_df)
    inits = [i for i in INIT_ORDER if i in set(c_df.init)]
    arms = sorted({(m, l, r) for m, l, r in
                   zip(c_df.surj_mode, c_df.lambda_surj, c_df.lambda_recon)},
                  key=lambda t: (-t[2], t[0]))
    metrics = [('valid', 'valid abstraction (rate)'),
               ('good', 'valid AND surjective (rate)')]

    fig, axes = plt.subplots(len(metrics), len(panels),
                             figsize=(3.4 * len(panels), 3.0 * len(metrics)), squeeze=False)
    width = 0.8 / len(arms)
    for j, c in enumerate(panels):
        cell = c_df[(c_df.n_low == c[0]) & (c_df.n_high == c[1]) & (c_df.density == c[2])]
        for i, (metric, label) in enumerate(metrics):
            ax = axes[i][j]
            for a_i, (mode, lam, lrec) in enumerate(arms):
                means = [cell[(cell.init == init) & (cell.surj_mode == mode)
                              & (cell.lambda_surj == lam) & (cell.lambda_recon == lrec)][metric].mean()
                         for init in inits]
                xs = np.arange(len(inits)) + a_i * width - 0.4 + width / 2
                colour = INK2 if lrec > 0 else SERIES[a_i - 1]
                ax.bar(xs, means, width=width * 0.86, color=colour,
                       label=arm_label(mode, lam, lrec) if (i == 0 and j == 0) else None, zorder=3)
            ax.set_xticks(range(len(inits)))
            ax.set_xticklabels([INIT_LABEL[k].replace('\n', ' ') for k in inits]
                               if i == len(metrics) - 1 else [''] * len(inits),
                               rotation=20, ha='right', fontsize=8)
            ax.set_ylim(0, 1.05)
            tidy(ax, ylabel=label if j == 0 else None,
                 title=cell_title(c) if i == 0 else None)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle('Does the recommended configuration survive a bad initialisation?',
                 x=0.01, ha='left', fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    fig.savefig(outdir / 'fig5_interaction.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------

AGG = ['valid', 'surjective', 'good', 'coverage', 'recon_loss',
       'log_obs', 'log_intv', 'abstraction_residual', 'runtime']


def tables(a, b, c_df, outdir):
    lines = ["# Smoke test tables",
             "",
             "valid       = obs_loss < 1e-8 and intv_loss < 1e-8",
             "surjective  = every high-level cluster has a member",
             "good        = valid AND surjective",
             "log_obs     = median log10(obs_loss)",
             ""]
    lines.append("\n## Sweep A - surjective loss (lambda_recon = 0 unless the row says otherwise)\n")
    for c in cells(a):
        cell = a[(a.n_low == c[0]) & (a.n_high == c[1]) & (a.density == c[2])]
        t = cell.groupby(['surj_mode', 'lambda_surj', 'lambda_recon'])[AGG].agg(
            {k: ('median' if k.startswith('log_') else 'mean') for k in AGG}).round(4)
        lines += [f"\n### {cell_title(c)}\n", t.to_string(), ""]

    lines.append("\n## Sweep B - initialisation\n")
    for c in cells(b):
        cell = b[(b.n_low == c[0]) & (b.n_high == c[1]) & (b.density == c[2])]
        t = cell.groupby('init')[AGG].agg(
            {k: ('median' if k.startswith('log_') else 'mean') for k in AGG}).reindex(INIT_ORDER).round(4)
        lines += [f"\n### {cell_title(c)}\n", t.to_string(), ""]

    if c_df is not None:
        lines.append("\n## Sweep C - recommended configuration x initialisation\n")
        for c in cells(c_df):
            cell = c_df[(c_df.n_low == c[0]) & (c_df.n_high == c[1]) & (c_df.density == c[2])]
            t = cell.groupby(['init', 'surj_mode', 'lambda_recon'])[AGG].agg(
                {k: ('median' if k.startswith('log_') else 'mean') for k in AGG}).round(4)
            lines += [f"\n### {cell_title(c)}\n", t.to_string(), ""]

    (outdir / 'tables.txt').write_text("\n".join(lines))


def paired_summary(a, b, c_df, outdir):
    rows = []
    metrics = ('good', 'valid', 'surjective', 'coverage', 'recon_loss', 'log_obs', 'runtime')
    for c in cells(a):
        cell = a[(a.n_low == c[0]) & (a.n_high == c[1]) & (a.density == c[2])]
        ctrl = (cell.surj_mode == 'hard_count') & (cell.lambda_recon == 10)
        arms = [('none', 0.0)] + [(m, l) for m in MODE_ORDER
                                  for l in sorted(cell.lambda_surj.unique()) if l > 0]
        for mode, lam in arms:
            mask = (cell.surj_mode == mode) & (cell.lambda_recon == 0) & (cell.lambda_surj == lam)
            if not mask.any():
                continue
            for metric in metrics:
                mean, lo, hi, n = paired_delta(cell, ctrl, mask, metric)
                rows.append({'sweep': 'A', 'n_low': c[0], 'n_high': c[1], 'density': c[2],
                             'init': 'reg_surjective', 'arm': f"{mode}@{lam:g}",
                             'metric': metric, 'delta': mean, 'ci_lo': lo, 'ci_hi': hi,
                             'n_pairs': n})
    for c in cells(b):
        cell = b[(b.n_low == c[0]) & (b.n_high == c[1]) & (b.density == c[2])]
        ctrl = cell.init == 'reg_surjective'
        for init in INIT_ORDER:
            if init == 'reg_surjective':
                continue
            for metric in metrics:
                mean, lo, hi, n = paired_delta(cell, ctrl, cell.init == init, metric)
                rows.append({'sweep': 'B', 'n_low': c[0], 'n_high': c[1], 'density': c[2],
                             'init': init, 'arm': 'current losses', 'metric': metric,
                             'delta': mean, 'ci_lo': lo, 'ci_hi': hi, 'n_pairs': n})
    if c_df is not None:
        for c in cells(c_df):
            cell = c_df[(c_df.n_low == c[0]) & (c_df.n_high == c[1]) & (c_df.density == c[2])]
            for init in sorted(set(cell.init)):
                sub = cell[cell.init == init]
                ctrl = sub.lambda_recon > 0
                for mode in sorted(set(sub[sub.lambda_recon == 0].surj_mode)):
                    mask = (sub.surj_mode == mode) & (sub.lambda_recon == 0)
                    for metric in metrics:
                        mean, lo, hi, n = paired_delta(sub, ctrl, mask, metric)
                        rows.append({'sweep': 'C', 'n_low': c[0], 'n_high': c[1],
                                     'density': c[2], 'init': init, 'arm': f"{mode}, recon off",
                                     'metric': metric, 'delta': mean, 'ci_lo': lo,
                                     'ci_hi': hi, 'n_pairs': n})
    df = pd.DataFrame(rows)
    df['verdict'] = np.where(df.ci_lo > 0, 'better',
                             np.where(df.ci_hi < 0, 'worse', 'no difference'))
    df.to_csv(outdir / 'paired_deltas.csv', index=False)
    return df


if __name__ == '__main__':
    results_dir = Path(sys.argv[1])
    outdir = results_dir / 'plots'
    outdir.mkdir(parents=True, exist_ok=True)

    a = add_metrics(pd.read_csv(results_dir / 'sweep_a.csv'))
    b = add_metrics(pd.read_csv(results_dir / 'sweep_b.csv'))
    c_path = results_dir / 'sweep_c.csv'
    c_df = add_metrics(pd.read_csv(c_path)) if c_path.exists() else None
    print(f"sweep A: {len(a)} runs, sweep B: {len(b)} runs, "
          f"sweep C: {len(c_df) if c_df is not None else 0} runs")

    figure_surjective_curves(a, outdir)
    figure_component_heatmap(a, outdir)
    figure_surjective_delta(a, outdir)
    figure_inits(b, outdir)
    figure_init_delta(b, outdir)
    if c_df is not None:
        figure_interaction(c_df, outdir)
    tables(a, b, c_df, outdir)
    d = paired_summary(a, b, c_df, outdir)
    print(f"wrote figures and tables to {outdir}")

    pd.set_option('display.width', 220)
    print("\n=== headline: rate of valid AND surjective, sweep A ===")
    a['arm'] = np.where(a.lambda_recon == 10, 'CURRENT DEFAULT',
                        np.where(a.surj_mode == 'none', 'recon off, no surj',
                                 a.surj_mode + '@' + a.lambda_surj.astype(str)))
    a['cell'] = a.n_low.astype(str) + '/' + a.n_high.astype(str) + ' d=' + a.density.astype(str)
    t = a.pivot_table(index='arm', columns='cell', values='good')
    t['MEAN'] = t.mean(axis=1)
    print(t.sort_values('MEAN', ascending=False).round(2).to_string())
