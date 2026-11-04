"""Reproduces the "Optimizing T" figure: how RAE and ReconErr move as the problem grows.

Two sweeps, one per column:

    k = p/10    the problem and the budget grow together, p = 40 ... 200
    p = 100     the problem is fixed and only the budget k moves, k = 4 ... 20

In both, the generating model has n_high = k clusters and every learner is given a budget of
k, so the ground-truth row is a like-for-like partition rather than a fixed reference -- which
is why the GT curve moves along the k axis at all.

WHAT THE TWO ROWS ANSWER. RAE (log scale) asks whether an exact abstraction was found; it is
at machine precision for the partitions where Theorem 8 guarantees one exists (ground truth
and AOC Closure) and nowhere near it for a random partition. ReconErr asks whether the
abstraction says anything -- a method can drive RAE to zero by collapsing everything into one
cluster, and ReconErr is what catches that. Neither number means much without the other.

Each point is the median over N_INSTANCES independently sampled graphs, with the band the
interquartile range. The spread is wide: these are properties of a single draw, not of the
method, and the IQR is the honest way to show that.

    python experiments/synthetic/sweeps.py              # -> assets/fig2_sweeps.png (about 50 min)
    python experiments/synthetic/sweeps.py --from-csv results/sweeps/<run>/metrics.csv
"""

import argparse
import os

# One BLAS thread. The fitter's matrices are cluster-sized (c x c for c ~ p/k, so 10 x 10
# here), and on blocks that small the threaded BLAS spends more time coordinating threads
# than doing arithmetic -- it was observed at 655% CPU while running 16x SLOWER than a single
# thread (p=140, k=14: 145.15s -> 8.91s, with bit-identical RAE). Set before numpy is
# imported, which is when the thread pool is sized; setdefault so an explicit outer value
# still wins.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from src.utils.generate_data import sample_low_level_graph
from src.algos.greedy import aoc_closure, oc_closure, greedy_allin, random_partition
from src.algos.T_fitter import fit_abstraction_eigen
from src.utils.losses import RAE, reconstruction_loss
from src.utils.abstraction_algebra import reconstruct_adj

N_INSTANCES = 20
DENSITY, BLOCKINESS = 0.5, 0.8

# Kept in step with partition_table.py and synthetic/benchmark.py. The closures are order-dependent and
# the published method draws several random merge orders, keeping the fewest intra-cluster
# edges; at one draw OC Closure stops being distinguishable from AOC Closure.
N_RESTARTS = 12

# (p, k) per column. p = 100, k = 10 is the one setting the two columns share.
SWEEP_P = [(p, p // 10) for p in (40, 60, 80, 100, 140, 200)]
SWEEP_K = [(100, k) for k in (4, 8, 10, 16, 20)]

# RAE on an exact abstraction underflows to 0, which a log axis cannot draw. Floor it at a
# value below anything meaningful rather than silently dropping the point.
RAE_FLOOR = 1e-18

METHODS = {
    'GT partition':  None,   # the generating partition, read off T_gt
    'Greedy All-In': lambda A, k, seed: greedy_allin(A, k),
    'OC Closure':    lambda A, k, seed: oc_closure(A, k, n_restarts=N_RESTARTS, seed=seed),
    'AOC Closure':   lambda A, k, seed: aoc_closure(A, k, n_restarts=N_RESTARTS, seed=seed),
    'random':        lambda A, k, seed: random_partition(A.shape[0], k),
}

# Line styles follow the paper, and stay distinguishable in greyscale: the two guaranteed-exact
# series are the dashed dark one and the orange one, the two that give up exactness are green
# and grey, and random is the dotted purple.
STYLE = {
    'GT partition':  dict(color='#4d4d4d', ls='--', marker='o', ms=4.5),
    'Greedy All-In': dict(color='#2ca02c', ls='-',  marker='D', ms=4.5),
    'OC Closure':    dict(color='#999999', ls='-.', marker='s', ms=4.0),
    'AOC Closure':   dict(color='#ff7f0e', ls='-',  marker='*', ms=7.0),
    'random':        dict(color='#9467bd', ls=':',  marker='v', ms=4.5),
}


def evaluate(A_low, partition, E_low):
    """RAE and ReconErr for one partition, via the closed-form fitter."""
    partition = [c for c in partition if len(c)]
    A_high, T, S = fit_abstraction_eigen(A_low=A_low, partition=partition)
    A_recon = reconstruct_adj(A_high=A_high, T=T)
    return (RAE(A_low=A_low, T=T, A_high=A_high, S=S, E_low=E_low),
            reconstruction_loss(A_low=A_low, T=T, A_high=A_high, A_hat=A_recon))


def run(n_instances=N_INSTANCES, out_dir=None, resume=None, verbose=True):
    """Both sweeps, appending to a CSV as it goes so a long run is never lost.

    The large-k settings are the expensive ones -- the fitter pins cyclic clusters by
    enumerating cycles in the quotient, and at k = 20 with a dense quotient that is both slow
    and memory-hungry, enough to get the process killed on a small machine. Pass `resume`
    with an earlier run's metrics.csv and every setting already complete in it is carried
    forward untouched, so a kill costs one setting rather than the whole sweep.
    """
    settings = [('k = p/10', p, k) for p, k in SWEEP_P] + \
               [('p = 100', p, k) for p, k in SWEEP_K]

    folder = Path(out_dir or f"results/sweeps/{time.strftime('%Y%m%d-%H%M%S')}")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'metrics.csv'

    rows, header_written = [], False
    done = set()
    if resume:
        prev = pd.read_csv(resume)
        full = n_instances * len(METHODS)
        done = {key for key, n in prev.groupby(['panel', 'p', 'k']).size().items()
                if n >= full}
        rows = prev.to_dict('records')
        prev.to_csv(path, index=False)
        header_written = True
        if verbose:
            print(f"  resumed from {resume}: {len(done)}/{len(settings)} settings already done")

    for panel, p, k in settings:
        if (panel, p, k) in done:
            continue
        t0 = time.perf_counter()
        for seed in range(n_instances):
            try:
                A_low, T_gt, _, _ = sample_low_level_graph(
                    n_high=k, n_low=p, density=DENSITY, blockiness=BLOCKINESS,
                    heterogeneity=0., internal_density=0., graph_type='random',
                    shuffle=True, internal=False, seed=seed, connected=True,
                    max_attempts=10)
            except RuntimeError:
                continue   # no connected draw in 10 attempts; the seed is simply skipped

            E_low = np.linalg.inv(np.eye(p) - A_low)
            gt = [np.where(T_gt[:, j] != 0)[0] for j in range(T_gt.shape[1])]

            batch = []
            for name, fn in METHODS.items():
                partition = gt if fn is None else fn(A_low, k, seed)
                rae, recon = evaluate(A_low, partition, E_low)
                batch.append(dict(panel=panel, p=p, k=k, seed=seed, method=name,
                                  rae=rae, recon=recon))

            rows.extend(batch)
            pd.DataFrame(batch).to_csv(path, mode='a', index=False,
                                       header=not header_written)
            header_written = True

        if verbose:
            print(f"  {panel:9s} p={p:3d} k={k:2d}  "
                  f"{time.perf_counter() - t0:6.1f}s", flush=True)

    print(f"Wrote {path}")
    return pd.DataFrame(rows)


def draw(df, save_to='assets/fig2_sweeps.png'):
    mpl.rcParams.update({'font.family': 'serif',
                         'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'mathtext.fontset': 'stix'})

    panels = [('k = p/10', 'p', r'$k = p/10$'), ('p = 100', 'k', r'$p = 100$')]
    metrics = [('rae', 'RAE', 'log'), ('recon', 'ReconErr', 'linear')]

    fig, axes = plt.subplots(2, 2, figsize=(11, 5.4))
    for col, (panel, xcol, col_title) in enumerate(panels):
        sub = df[df['panel'] == panel]
        for row, (metric, ylabel, yscale) in enumerate(metrics):
            ax = axes[row][col]
            for name in METHODS:
                g = sub[sub['method'] == name].groupby(xcol)[metric]
                x = np.array(sorted(g.groups))
                med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
                if metric == 'rae':
                    med, lo, hi = (np.maximum(v, RAE_FLOOR) for v in (med, lo, hi))
                ax.plot(x, med, label=name, lw=1.4, **STYLE[name])
                ax.fill_between(x, lo, hi, color=STYLE[name]['color'], alpha=0.13, lw=0)

            ax.set_yscale(yscale)
            ax.grid(alpha=0.3, lw=0.5)
            ax.set_xticks([p for p, _ in SWEEP_P] if col == 0 else
                          [k for _, k in SWEEP_K])
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(col_title, fontsize=10)
            else:
                ax.set_xlabel(xcol)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(METHODS),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(r'Optimizing $\mathbf{T}$', fontsize=11)
    fig.tight_layout(rect=(0, 0.045, 1, 1))

    Path(save_to).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_to, dpi=300, bbox_inches='tight')
    print(f"Saved {save_to}")


def summarise(df):
    """The medians behind the figure, so a run says something without opening the PNG."""
    for panel, xcol in [('k = p/10', 'p'), ('p = 100', 'k')]:
        sub = df[df['panel'] == panel]
        print(f"\n{panel}  (median over {sub['seed'].nunique()} instances)")
        for metric in ('rae', 'recon'):
            table = sub.pivot_table(index='method', columns=xcol, values=metric,
                                    aggfunc='median').reindex(METHODS)
            fmt = '%.1e' if metric == 'rae' else '%.3f'
            print(f"  {metric}:")
            print(table.to_string(float_format=lambda v: fmt % v))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--instances', type=int, default=N_INSTANCES)
    ap.add_argument('--out', default='assets/fig2_sweeps.png')
    ap.add_argument('--from-csv', default=None,
                    help='replot an earlier run instead of recomputing')
    ap.add_argument('--resume', default=None,
                    help="an earlier run's metrics.csv; its complete settings are not redone")
    args = ap.parse_args()

    df = (pd.read_csv(args.from_csv) if args.from_csv
          else run(n_instances=args.instances, resume=args.resume))
    summarise(df)
    draw(df, save_to=args.out)
