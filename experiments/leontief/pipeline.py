import pandas as pd
import numpy as np
import pymrio
import tracemalloc
import time
import math
from pathlib import Path

from src.algos.greedy import (aoc_closure, greedy_allin, oc_closure,
                                        random_partition)
from src.utils.partition import all_in_violation, partition_to_matrix
from src.utils.abstraction_algebra import enforce_dag, reconstruct_adj
from src.algos.T_fitter import fit_abstraction_eigen
from src.utils.losses import abstraction_residual, graph_loss, observation_loss, intervention_loss, reconstruction_loss, reconstruction_loss_with_data
from src.utils.generate_data import sample_by_noise_adjustment_any_order
from src.utils.plots import plot_results, plot_shared_legend

def parse_data(cache="data/A_low.npy"):
    """GB block of the EXIOBASE technical-coefficient matrix, bordered by factor inputs.

    The result is 172 x 172. Building it from the raw archive is expensive: the full
    multi-region system is ~7987 x 7987, and every satellite account is computed before
    we slice. The cached .npy is 236 KB, so it is worth keeping.
    """
    cache = Path(cache)
    if cache.exists():
        A_low = np.load(cache)
        print(f"A_low loaded from {cache}. Shape: {A_low.shape}. "
              f"Max: {np.max(A_low)}. Min: {np.min(A_low)}.")
        return A_low

    print("No cache found - parsing EXIOBASE. This is slow and memory-hungry.")
    exio_data = pymrio.parse_exiobase3(path="data/IOT_2024_ixi.zip")
    exio_data.calc_system()
    # only factor_inputs is needed; EXIOBASE ships many other satellite accounts
    exio_data.calc_extensions(extensions=["factor_inputs"])

    idx = pd.IndexSlice
    target_regions = ["GB"]

    A_filtered = exio_data.A.loc[idx[target_regions, :], idx[target_regions, :]]
    ST = exio_data.factor_inputs.S.loc[:, idx[target_regions, :]].T   # (N, K)

    N, K = A_filtered.shape[0], ST.shape[1]

    # border A with the factor-input rows/columns, as numpy - building two dense
    # DataFrames of zeros and concatenating costs far more than it needs to
    A_low = np.zeros((N + K, N + K))
    A_low[:N, :N] = A_filtered.to_numpy()
    A_low[:N, N:] = ST.to_numpy()

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, A_low)
    print(f"A_low built and cached to {cache}. Shape: {A_low.shape}")
    return A_low


# each returns a PARTITION (list of index arrays), which the fitter consumes directly
METHODS = {'aoc_closure': aoc_closure,
           'greedy_allin': greedy_allin,
           'oc_closure': oc_closure,
           'random': lambda A_low, k, **kwds: random_partition(A_low.shape[0], k, **kwds)}

def run(A_low,k_list,num_trials=10):
    A_low = enforce_dag(A_low)
    p = A_low.shape[0]
    E_low = np.linalg.inv(np.eye(p) - A_low)
    X_test = sample_by_noise_adjustment_any_order(A_low,n_samples=10000,gennorm_param=1.8,seed=None)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    folder = Path(f"results/leontief/{timestamp}")
    folder.mkdir(parents=True, exist_ok=True)
    filename = folder / "metrics.csv"
    header_written = False 

    for n in range(num_trials):
        start_time = time.time()
        for k in k_list:
            for method_name, method_fn in METHODS.items():
                tracemalloc.start()
                method_start = time.perf_counter()
                partition = method_fn(A_low=A_low, k=k)
                runtime = time.perf_counter() - method_start
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                min_t = 1/(5 * math.sqrt(p))

                T_hard = partition_to_matrix(partition)
                A_high_pred, T_pred, S_pred = fit_abstraction_eigen(A_low=A_low,
                                                                    partition=partition)
                A_recon = reconstruct_adj(A_high=A_high_pred,T=T_pred)
                E_high_pred = np.linalg.inv(np.eye(A_high_pred.shape[0]) - A_high_pred)
                abs_residual = abstraction_residual(A_low=A_low,T=T_pred,A_high=A_high_pred)
                
                new_result = [{'Approach': method_name,
                                'n_low': p,
                                'k':k,
                                # clusters actually used, as a fraction of the budget
                                'surjective': len(partition)/k,
                                'all_in_violation': all_in_violation(A_low, partition),
                                'marginalisation': np.mean((np.abs(T_pred).sum(axis=1) < min_t)),
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

        plot_easy_benchmark(timestamp=timestamp,save_dir= Path(folder/ "plots"))

def plot_easy_benchmark(timestamp = "20260528-120000", collate_from = None, approaches=['aoc_closure','greedy_allin','oc_closure','random'],x_axis='k', col_var=None, row_var=None, fixed_vars={}, save_dir=None, legend_only=False, title=None):
    
    assert timestamp is not None or collate_from is not None, "Expected either timestamp or collate_from to be specified."

    results_dir = Path("results/leontief")

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

if __name__ == "__main__":
    RUN = False
    PLOT = True

    A_low = parse_data()
    p = A_low.shape[0]
    if RUN:
        run(A_low=A_low,k_list=list(range(6,12,2)))
    if PLOT:
        # latest run: single k-sweep over the full p = 172 Leontief graph
        plot_easy_benchmark(timestamp="20260806-001206", approaches=['aoc_closure','greedy_allin','oc_closure','random'],
                            col_var=None, title=f"p = {p}",
                            save_dir="results/leontief/_collated_plots", legend_only=False)
    

    

