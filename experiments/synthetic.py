import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))

from pathlib import Path
import tracemalloc
import time
import numpy as np
import pandas as pd
import itertools
import yaml
from src.utils.generate_data import sample_low_level_graph, sample_by_noise_adjustment_any_order
from src.utils.helper import enforce_dag, get_A_high, get_S, reconstruct_adj, process_output
from src.utils.losses import abstraction_residual, observation_loss, intervention_loss, graph_loss, reconstruction_loss, reconstruction_loss_with_data
from src.models.wrapper import hail_mary_graph, uca, hc_fast, random
from src.utils.plots import plot_results, plot_shared_legend

METHODS = {'hm_graph': hail_mary_graph,
           'uca': uca,
           'hc_fast': hc_fast,
           'random': random}

# regime = (n_low, n_high, k_list, method_name_list)
regimes = [(10*n_high, n_high, list(range(n_high, 10*n_high + 1, n_high)),['hm_graph','uca','hc_fast','random']) for n_high in [5,10,15,20]

            ]
def easy_benchmark(num_trials=10,regime_list=regimes,
                   density_list=[None],
                   blockiness_list=[None],
                   heterogeneity_list=[None],
                   internal_density_list=[None],
                   METHODS=METHODS,
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
            A_low, T_gt, S_gt, A_high_gt = sample_low_level_graph(n_low=n_low,n_high=n_high,density=density,blockiness=blockiness,heterogeneity=het, internal_density=internal_density, shuffle=True, internal=False, graph_type='one_child')
            E_low = np.linalg.inv(np.eye(n_low) - A_low)
            X_test = sample_by_noise_adjustment_any_order(A_low,n_samples=10000,gennorm_param=1.8,seed=None)

            for k in k_list:
                for method_name in method_name_list:
                    method_fn = METHODS[method_name]

                    tracemalloc.start()
                    method_start = time.perf_counter()
                    T_hard = method_fn(A_low=A_low, k=k)
                    runtime = time.perf_counter() - method_start
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    if method_name == "uca":
                        T_pred = T_hard[:,:]
                        col_indices = np.argmax(np.abs(T_pred),axis=1)
                        T_hard = np.zeros_like(T_hard)
                        T_hard[:, col_indices] = 1.
                        #T_pred = T_pred * T_hard
                        S_pred = get_S(E_low=E_low, T=T_pred)
                        A_high_pred = enforce_dag(get_A_high(A_low=A_low, T=T_pred))
                    else:
                        S_pred, T_pred, A_high_pred = process_output(T_hard=T_hard, A_low=A_low)
                    A_recon = reconstruct_adj(A_high=A_high_pred,T=T_pred)
                    E_high_pred = np.linalg.inv(np.eye(k) - A_high_pred)
                    abs_residual = abstraction_residual(A_low=A_low,T=T_pred,A_high=A_high_pred)
                    new_result = [{'Approach': method_name,
                                    'n_low': n_low,
                                    'n_high': n_high,
                                    'ratio': n_low/n_high,
                                    'k':k,
                                    'density': density,
                                    'blockiness':blockiness,
                                    'heterogeneity':het,
                                    'internal_density':internal_density,
                                    'surjective': np.mean((np.abs(T_hard).sum(axis=0) > 0)),
                                    'marginalisation': np.mean((np.abs(T_pred).sum(axis=1) < 1e-6)),
                                    'runtime': runtime,
                                    'memory': peak/1e6,
                                    'graph_loss':graph_loss(A_low=A_low,T_hard=T_hard),
                                    'obs_loss': observation_loss(E_low=E_low, E_high=E_high_pred, S=S_pred, T=T_pred),
                                    'intv_loss': intervention_loss(E_low=E_low, E_high=E_high_pred, S=S_pred, T=T_pred),
                                    'abstraction_residual': abs_residual,
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

    
def plot_easy_benchmark(timestamp = "20260528-120000", collate_from = None, approaches=['hm_graph','uca','hc_fast','random'],x_axis='k', col_var='n_high', row_var=None, fixed_vars={}, save_dir=None, legend_only=False, title=None):
    
    assert timestamp is not None or collate_from is not None, f"Expected either timestamp or collate_from to be specified."

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


    
    

if __name__ == '__main__':
    EASY_BENCHMARK_RUN = False
    EASY_BENCHMARK_PLOT = True

    #cut_off_good_results = 20260618-100000

    if EASY_BENCHMARK_RUN:
        easy_benchmark(heterogeneity_list=[1],internal_density_list=[0], blockiness_list=[0.5], num_trials=10)
    if EASY_BENCHMARK_PLOT:
        # latest run fixes k = n_high (n_low = 10*k), so plot a single panel over k.
        # Use one run only -- collating across runs mixes different setups/scales and blows up the error bands.
        plot_easy_benchmark(timestamp="20260712-214422", approaches=['hm_graph','uca','hc_fast','random'],x_axis='k',col_var=None,
                            fixed_vars={'heterogeneity':1,'internal_density':0, 'blockiness':0.5}, title="p = 10k",
                            save_dir="results/synthetic/_collated_plots", legend_only=False)