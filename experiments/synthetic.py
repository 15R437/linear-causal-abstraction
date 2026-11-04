from pathlib import Path
import tracemalloc
import time
import numpy as np
import pandas as pd
import itertools
import yaml
from src.utils.generate_data import sample_low_level_graph, sample_by_noise_adjustment_any_order
from src.utils.helper import enforce_dag, get_A_high, get_S, reconstruct_adj
from src.utils.losses import abstraction_residual, observation_loss, intervention_loss, graph_loss, reconstruction_loss, reconstruction_loss_with_data, RAE
from src.algos.T_fitter import fit_abstraction_eigen
from src.algos.greedy import (aoc_closure, greedy_allin, oc_closure,
                                        random_partition)
from src.utils.partition import all_in_violation, partition_to_matrix
from src.algos.gradcap import gradcap
from src.utils.plots import plot_results, plot_shared_legend

# Every method here returns a PARTITION -- a list of index arrays, one per cluster -- which
# is what `fit_abstraction_eigen` consumes directly. The p x k indicator matrix is built
# once per row, only for the metrics that are genuinely matrix algebra (`graph_loss`).
# UCA is the sole exception: it learns a dense soft tau map and never commits to clusters.
# The closures are ORDER-DEPENDENT, so the published method draws several random merge
# orders and keeps the one with the fewest intra-cluster edges. n_restarts=1 is a different
# (and worse) algorithm: at one draw OC Closure lands on All-In-satisfying partitions and
# stops being distinguishable from AOC Closure. Keep this in step with
# experiments/partition_table.py.
N_RESTARTS = 12

METHODS = {'aoc_closure': lambda A_low, k, **kw: aoc_closure(A_low, k, n_restarts=N_RESTARTS),
           'greedy_allin': lambda A_low, k, **kw: greedy_allin(A_low, k),
           'oc_closure': lambda A_low, k, **kw: oc_closure(A_low, k, n_restarts=N_RESTARTS),
           'gradcap': gradcap,
           'random': lambda A_low, k, **kw: random_partition(A_low.shape[0], k)}

# regime = (n_low, n_high, k_list, method_name_list)
regimes = [(10*n_high, n_high, list(range(n_high, 10*n_high + 1, n_high)),['aoc_closure','greedy_allin','oc_closure','uca','random']) for n_high in [5,10,15,20]

            ]
def easy_benchmark(num_trials=10,regime_list=regimes,
                   density_list=[None],
                   blockiness_list=[None],
                   heterogeneity_list=[None],
                   internal_density_list=[None],
                   METHODS=METHODS,
                   graph_type='random',
                   max_attempts=10,
                   output_subdir="benchmark_run_v1"):

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    folder = Path(f"results/synthetic/{timestamp}_{output_subdir}")
    folder.mkdir(parents=True, exist_ok=True)

    config = {'num_trials':num_trials, 'regimes':regime_list}
    with open(folder/"config.yaml", "w") as f:
        yaml.dump(config,f)
    
    filename = folder / "metrics.csv"
    header_written = False
    for n in range(num_trials):
        start_time = time.time()
        for regime, density, blockiness, het, internal_density in itertools.product(regime_list, density_list, blockiness_list, heterogeneity_list, internal_density_list):
            n_low,n_high,k_list,method_name_list = regime
            density = density if density is not None else 0.4
            blockiness = blockiness if blockiness is not None else 0.5
            internal_density = internal_density if internal_density is not None else 0.
            het = het if het is not None else 0.1
            A_low, T_gt, S_gt, A_high_gt = sample_low_level_graph(n_low=n_low,n_high=n_high,density=density,blockiness=blockiness,heterogeneity=het, internal_density=internal_density, shuffle=True, internal=False, graph_type=graph_type, max_attempts=max_attempts)
            E_low = np.linalg.inv(np.eye(n_low) - A_low)
            X_test = sample_by_noise_adjustment_any_order(A_low,n_samples=10000,gennorm_param=1.8,seed=None)

            for k in k_list:
                for method_name in method_name_list:
                    method_fn = METHODS[method_name]

                    tracemalloc.start()
                    method_start = time.perf_counter()
                    output = method_fn(A_low=A_low, k=k)
                    runtime = time.perf_counter() - method_start
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    if method_name == "uca":
                        # no partition to be had: binarise the soft map by row-argmax
                        T_pred = output
                        col = np.argmax(np.abs(T_pred), axis=1)
                        partition = [np.where(col == j)[0] for j in range(T_pred.shape[1])]
                        T_hard = np.zeros_like(T_pred)
                        T_hard[np.arange(n_low), col] = 1.
                        S_pred = get_S(E_low=E_low, T=T_pred)
                        A_high_pred = enforce_dag(get_A_high(A_low=A_low, T=T_pred))
                    else:
                        partition = output
                        T_hard = partition_to_matrix(partition)
                        A_high_pred, T_pred, S_pred = fit_abstraction_eigen(A_low=A_low,
                                                                            partition=partition)
                    A_recon = reconstruct_adj(A_high=A_high_pred,T=T_pred)
                    # size from A_high_pred, not k: a method may return fewer clusters than
                    # its budget, and the partition's own length is what the fit used
                    E_high_pred = np.linalg.inv(np.eye(A_high_pred.shape[0]) - A_high_pred)
                    abs_residual = abstraction_residual(A_low=A_low,T=T_pred,A_high=A_high_pred)
                    new_result = [{'Approach': method_name,
                                    'trial': n,
                                    'n_low': n_low,
                                    'n_high': n_high,
                                    'ratio': n_low/n_high,
                                    'k':k,
                                    'density': density,
                                    'blockiness':blockiness,
                                    'heterogeneity':het,
                                    'internal_density':internal_density,
                                    # clusters actually used, as a fraction of the budget
                                    'surjective': len(partition)/k,
                                    'all_in_violation': all_in_violation(A_low, partition),
                                    'marginalisation': np.mean((np.abs(T_pred).sum(axis=1) < 1e-6)),
                                    'runtime': runtime,
                                    'memory': peak/1e6,
                                    'graph_loss':graph_loss(A_low=A_low,T_hard=T_hard),
                                    'obs_loss': observation_loss(E_low=E_low, E_high=E_high_pred, S=S_pred, T=T_pred),
                                    'intv_loss': intervention_loss(E_low=E_low, E_high=E_high_pred, S=S_pred, T=T_pred),
                                    'abstraction_residual': abs_residual,
                                    'rae': RAE(A_low=A_low, T=T_pred, A_high=A_high_pred, S=S_pred, E_low=E_low, T_hard=T_hard),
                                    'recon_loss': reconstruction_loss(A_low=A_low, T=T_pred, A_high=A_high_pred, A_hat=A_recon),
                                    'recon_loss_data':reconstruction_loss_with_data(A_low=A_low, A_high=A_high_pred, A_hat=A_recon, X=X_test, T=T_pred)
                                    }]
                    df_batch = pd.DataFrame(new_result)
                    df_batch.to_csv(
                        filename, 
                        mode='a', 
                        index=False, 
                        header=not header_written
                    )
                    header_written = True
        runtime = time.time() - start_time
        print(f"Trial {n+1}/{num_trials} complete after {runtime:.4f}s.")

        plot_easy_benchmark(timestamp=timestamp,save_dir= Path(folder/ "plots"), approaches=method_name_list)

    return folder

    
def plot_easy_benchmark(timestamp = "20260528-120000", collate_from = None, approaches=['aoc_closure','greedy_allin','oc_closure','uca','random'],x_axis='k', col_var='n_high', row_var=None, fixed_vars={}, save_dir=None, legend_only=False, title=None):
    
    assert timestamp is not None or collate_from is not None, "Expected either timestamp or collate_from to be specified."

    results_dir = Path("results/synthetic")

    if isinstance(collate_from, str):
        df_list = []
        for dir in results_dir.iterdir():
            if dir.is_dir():
                dir_name = dir.name
                if dir_name < collate_from:
                    continue

                raw_file = dir / "metrics.csv"
                if raw_file.exists():
                    data = pd.read_csv(raw_file)
                    df_list.append(data)
        
        results_df = pd.concat(df_list, ignore_index=True) if df_list else None
    else:
        folders = list(results_dir.glob(f"{timestamp}*"))
        if not folders:
            print(f"No experiment found for timestamp: {timestamp}.")
            return None
        raw_file = folders[0] / "metrics.csv"
        if raw_file.exists():
            results_df = pd.read_csv(raw_file)
        else:
            print(f"Raw file not found in {folders[0]}.")

    if results_df is not None and not results_df.empty:
        no_random_methods = approaches[:]
        if 'random' in approaches:
            no_random_methods.remove('random')

        # exact-recovery rate: both observation and intervention losses ~ 0
        results_df['accuracy'] = ((results_df['obs_loss'] < 1e-8) & (results_df['intv_loss'] < 1e-8)).astype(int)

        plot_shared_legend(results_df=results_df, approaches=approaches, save_dir=save_dir)
        if legend_only:
            return

        plot_results(results_df=results_df, approaches=no_random_methods, x_axis=x_axis, y_axis='runtime', row=row_var, col=col_var, ylog=True, fixed_vars=fixed_vars, save_dir=save_dir, title=title)
        plot_results(results_df=results_df, approaches=no_random_methods, x_axis=x_axis, y_axis='memory', row=row_var, col=col_var,ylog=True, fixed_vars=fixed_vars, save_dir=save_dir, title=title)
        plot_results(results_df=results_df, approaches=approaches, x_axis=x_axis, y_axis='recon_loss', row=row_var, col=col_var,ylog=True, fixed_vars=fixed_vars, save_dir=save_dir, title=title)
        plot_results(results_df=results_df, approaches=approaches, x_axis=x_axis, y_axis='recon_loss_data', row=row_var, col=col_var,ylog=True, fixed_vars=fixed_vars, save_dir=save_dir, title=title)
        plot_results(results_df=results_df, approaches=approaches, x_axis=x_axis, y_axis='obs_loss', row=row_var, col=col_var,ylog=True, fixed_vars=fixed_vars, save_dir=save_dir, title=title)
        plot_results(results_df=results_df, approaches=approaches, x_axis=x_axis, y_axis='intv_loss', row=row_var, col=col_var,ylog=True, fixed_vars=fixed_vars, save_dir=save_dir, title=title)
        plot_results(results_df=results_df, approaches=approaches, x_axis=x_axis, y_axis='graph_loss', row=row_var, col=col_var,ylog=True, fixed_vars=fixed_vars, save_dir=save_dir, title=title)
        plot_results(results_df=results_df, approaches=approaches, x_axis=x_axis, y_axis='surjective', row=row_var, col=col_var,ylog=False, fixed_vars=fixed_vars, save_dir=save_dir, title=title)
        plot_results(results_df=results_df, approaches=approaches, x_axis=x_axis, y_axis='marginalisation', row=row_var, col=col_var,ylog=False, fixed_vars=fixed_vars, save_dir=save_dir, title=title)
        plot_results(results_df=results_df, approaches=approaches, x_axis=x_axis, y_axis='accuracy', row=row_var, col=col_var,ylog=False, fixed_vars=fixed_vars, save_dir=save_dir, title=title)
        #plot_results(results_df=results_df, approaches=approaches, x_axis=x_axis, y_axis='graph_loss', row=row_var, col=col_var)
    else:
        print("No CSV files found in the directory.")


    
    


# --- Figure: the exactness-informativeness trade-off -------------------------------------
# Reproduces the paper's tolerance figure. Fix a tolerance alpha in advance; accept a
# method's abstraction iff its RAE <= alpha AND it reconstructs strictly better than AOC
# Closure, which Theorem 8 guarantees is exact. Otherwise fall back to AOC Closure. Each
# curve is therefore what you would actually ship under an RAE <= alpha requirement.

# GradCAP is omitted: it is the gradient-based arm, an order of magnitude slower than the
# closures here, and it does not change the picture. Add 'gradcap' to rerun with it.
TRADE_OFF_REGIMES = [(100, 10, [10],
                      ['aoc_closure', 'greedy_allin', 'oc_closure', 'random'])]
ALPHA_MIN = 1e-6

STYLE = {'AOC Closure':   ('#eb6834', '-',  'o', 2.0),
         'Greedy All-In': ('#2ca02c', '--', 'D', 1.4),
         'OC Closure':    ('#7a7a7a', '--', 's', 1.4),
         'GradCAP':       ('#17becf', '-.', '^', 1.4),
         'UCA':           ('#9467bd', ':',  'v', 1.4),
         'random':        ('#1f77b4', ':',  'x', 1.0)}
LABEL = {'aoc_closure': 'AOC Closure', 'greedy_allin': 'Greedy All-In',
         'oc_closure': 'OC Closure', 'gradcap': 'GradCAP', 'uca': 'UCA', 'random': 'random'}


def alpha_grid(rae, n=12):
    """Log grid from ALPHA_MIN to the largest RAE observed, so no arm is cut off.

    The floor is fixed rather than data-driven: exact arms sit near 1e-16, and anchoring
    there would spend ten decades on a region where every curve is flat by construction.
    """
    r = pd.Series(rae).replace([np.inf, -np.inf], np.nan).dropna()
    r = r[r > 0]
    hi = min(1.0, float(r.max()) * 3) if len(r) else 1.0
    return np.unique(np.concatenate([np.geomspace(ALPHA_MIN, max(hi, 1e-3), n), [1.0]]))


def tolerance_curves(df, alphas, fallback='aoc_closure'):
    """Per-method ReconErr after the fallback rule, median over trials, at ONE density."""
    base = df[df.Approach == fallback].set_index('trial').recon_loss.to_dict()
    out = {}
    for name in df.Approach.unique():
        if name == fallback:
            continue
        sub, ys = df[df.Approach == name], []
        for al in alphas:
            per = []
            for trial, g in sub.groupby('trial'):
                b = base.get(trial, np.nan)
                ok = g[(g.rae <= al) & (g.recon_loss < b)]
                per.append(float(ok.recon_loss.min()) if len(ok) else b)
            ys.append(np.nanmedian(per))
        out[LABEL.get(name, name)] = np.array(ys)
    out[LABEL[fallback]] = np.full(len(alphas), np.nanmedian(list(base.values())))
    return out


def plot_trade_off(metrics_csv, densities=(0.3, 0.5), k=10, n_low=100, save_to=None):
    """The tolerance figure, read back from a metrics.csv.

    density 0.1 is not a default panel. At k=10 a density-0.1 draw gives 4.5 high-level
    edges on average while weak connectivity needs 9, so `connected=True` accepts only
    0.75% of draws -- reachable with max_attempts in the hundreds, but every surviving
    instance is then drawn from that tail and carries about twice the nominal edge count.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({'font.family': 'serif',
                         'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'mathtext.fontset': 'stix'})

    df = pd.read_csv(metrics_csv)
    df = df[(df.k == k) & (df.n_low == n_low)]
    alphas = alpha_grid(df.rae)

    fig, axes = plt.subplots(1, len(densities), figsize=(4 * len(densities), 3.2), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, densities):
        sub = df[np.isclose(df.density, d)]
        if sub.empty:
            ax.set_title(f'density {d}  (no data)'); continue
        for name, ys in tolerance_curves(sub, alphas).items():
            c, ls, mk, lw = STYLE.get(name, ('#444', '-', 'o', 1.2))
            ax.plot(alphas, ys, ls, color=c, marker=mk, ms=3, lw=lw, label=name)
        ax.set_xscale('log')
        ax.set_title(f'density {d}', fontsize=10)
        ax.set_xlabel(r'tolerance $\alpha$', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.22, lw=0.5)
    axes[0].set_ylabel('ReconErr', fontsize=9)
    axes[-1].legend(fontsize=7.5, frameon=False, loc='center left')
    fig.tight_layout()
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=300, bbox_inches='tight')
        print(f'Saved {save_to}')
    return fig


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trials', type=int, default=20,
                    help='instances per density; the figure plots the median over them')
    ap.add_argument('--densities', type=float, nargs='+', default=[0.3, 0.5])
    ap.add_argument('--out', default='assets/fig3_tolerance.png')
    ap.add_argument('--from-csv', default=None,
                    help='skip the run and replot an existing metrics.csv')
    args = ap.parse_args()

    csv = args.from_csv
    if csv is None:
        folder = easy_benchmark(num_trials=args.trials, regime_list=TRADE_OFF_REGIMES,
                                density_list=list(args.densities), blockiness_list=[0.8],
                                output_subdir='trade_off')
        csv = Path(folder) / 'metrics.csv'
    plot_trade_off(csv, densities=tuple(args.densities), save_to=args.out)
