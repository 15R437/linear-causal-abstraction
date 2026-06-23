import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))

import tracemalloc
import math
import time
import torch
import numpy as np
import pandas as pd
import itertools
from utils.generate_data import generate_easy_benchmark, sample_low_level_graph, sample_gennorm_lingam_population_standardized
from utils.helper import get_A_high, get_S, reconstruct_adj, process_output
from utils.losses import abstraction_residual, observation_loss, intervention_loss_direct, graph_loss, reconstruction_loss, reconstruction_loss_with_data
from utils.methods import hc_abs, hc_fast, hc_binary, random, hail_mary, hail_mary_abstract, hail_mary_graph, hail_mary_graph_topo, ground_truth,hail_mary_abstract_topo, hail_mary_intv, hail_mary_intv_topo
from utils.plots import plot_matrix, plot_results, plot_assignment_raster

def flip_T(T_gt, flip_prop, seed=None):
    rng = np.random.default_rng(seed)
    n_low, n_high = T_gt.shape
    num_flips = int(n_low * flip_prop)
    flip_indices = rng.choice(n_low, size=num_flips, replace=False)
    T_init = T_gt.copy()
    T_init[flip_indices] = 1 - T_init[flip_indices]
    return torch.from_numpy(T_init).float()

METHODS = {'hail_mary': hail_mary, 
           'hm_abstract': hail_mary_abstract, 
           'hm_graph': hail_mary_graph, 
           'hm_intv': hail_mary_intv,
           'hm_intv_topo': hail_mary_intv_topo,
           'hm_graph_topo': hail_mary_graph_topo, 
           'hm_abstract_topo': hail_mary_abstract_topo,
           'hc_fast': hc_fast, 'hc_binary': hc_binary, 'hc_abs': hc_abs,
           'random': random}

data_generating_methods = {'easy': generate_easy_benchmark, 'standard':sample_low_level_graph}

# regime = (n_low, n_high, k_list, method_name_list)
regimes = [(10*n_high, n_high, list(range(2,10*n_high+2,n_high)),['hm_graph','hc_fast','random']) for n_high in [5,10,15,20,30]
            #(2,range(60,110,10),['hail_mary','hc_fast']),
            #(5,range(10,43,3),['hail_mary','hc_fast']),
            #(10,range(10,21),['hail_mary','hc_fast'])
            ]
def easy_benchmark(num_trials=10,regime_list=regimes,
                   density_list=[None],
                   blockiness_list=[None],
                   heterogeneity_list=[None],
                   internal_density_list=[None],
                   METHODS=METHODS,
                   data_gen='standard'):
    """
    For 'easy' ground truth graphs, how often does hail_mary
    achieve perfect abstraction compared to hc as n_low grows.
    """
    generate_data_fn = data_generating_methods[data_gen]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"results/sandbox/{timestamp}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    header_written = False 
    results = []
    for n in range(num_trials):
        start_time = time.time()
        for regime, density, blockiness, het, internal_density in itertools.product(regime_list, density_list, blockiness_list, heterogeneity_list, internal_density_list):
            n_low,n_high,k_list,method_name_list = regime
            density = density if density is not None else 0.4
            blockiness = blockiness if blockiness is not None else 0.5
            internal_density = internal_density if internal_density is not None else 0.
            het = het if het is not None else 0.1
            A_low, T_gt, S_gt, A_high_gt = generate_data_fn(n_low=n_low,n_high=n_high,density=density,blockiness=blockiness,heterogeneity=het, internal_density=internal_density, shuffle=True, internal=False)
            T_init = None #if flip_prop is None else flip_T(T_gt, flip_prop)
            E_low = np.linalg.inv(np.eye(n_low) - A_low)
            X_test = sample_gennorm_lingam_population_standardized(A_low,n_samples=10000,gennorm_param=1.2,seed=None)

            for k in k_list:
                for method_name in method_name_list:
                    method_fn = METHODS[method_name]

                    tracemalloc.start()
                    method_start = time.perf_counter()
                    T_hard, dag_mask = method_fn(A_low=A_low, E_low=E_low, k=k, logits_init=T_init, T_gt=T_gt)
                    runtime = time.perf_counter() - method_start
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    min_t = 1/(5 * math.sqrt(n_low))
                    
                    S_pred, T_pred, A_high_pred = process_output(T_hard=T_hard, A_low=A_low, E_low=E_low, dag_mask=dag_mask)
                    A_recon = reconstruct_adj(A_high=A_high_pred,T=T_pred)
                    E_high_pred = np.linalg.inv(np.eye(k) - A_high_pred)
                    abs_residual = abstraction_residual(A_low=A_low,T=T_pred,A_high=A_high_pred)
                    new_result = [{'Approach': method_name,
                                    'n_low': n_low,
                                    'n_high': n_high,
                                    'k':k,
                                    'density': density,
                                    'blockiness':blockiness,
                                    'heterogeneity':het,
                                    'internal_density':internal_density,
                                    'surjective': np.mean((np.abs(T_hard).sum(axis=0) > 0)),
                                    'marginalisation': np.mean((np.abs(T_pred).sum(axis=1) < min_t)),
                                    'accuracy': np.isclose(abs_residual, 0, atol=1e-6) * 1.,
                                    'runtime': runtime,
                                    'memory': peak/1e6,
                                    'graph_loss':graph_loss(A_low=A_low,T=T_hard),
                                    'obs_loss': observation_loss(E_low=E_low, E_high=E_high_pred, S=S_pred, T=T_pred),
                                    'intv_loss': intervention_loss_direct(E_low=E_low, E_high=E_high_pred, S=S_pred, T=T_pred),
                                    'abstraction_residual': abs_residual,
                                    'recon_loss': reconstruction_loss(A_low=A_low, A_hat=A_recon),
                                    'recon_loss_data':reconstruction_loss_with_data(A_low=A_low, A_hat=A_recon, X=X_test)
                                    }]
                    results.extend(new_result)
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
    
def plot_easy_benchmark(cut_off = "20260528-120000", approaches=[],x_axis='n_high', col_var=None, row_var=None, fixed_vars={}):
    results_dir = Path("results/sandbox")
    csv_files = [f for f in results_dir.glob("*.csv") if f.stem >= cut_off]

    df_list = []
    for f in csv_files:
        data = pd.read_csv(f)
        df_list.append(data)

    if df_list:
        results_df = pd.concat(df_list, ignore_index=True)
        print(f"Collated {len(df_list)} file(s) saved after {cut_off}. Total rows: {len(results_df)}")

        no_random_methods = approaches[:]
        if 'random'in approaches:
            no_random_methods.remove('random')
    
        plot_results(results_df=results_df, approaches=no_random_methods, x_axis=x_axis, y_axis='runtime', row=row_var, col=col_var, ylog=True, fixed_vars=fixed_vars)
        plot_results(results_df=results_df, approaches=no_random_methods, x_axis=x_axis, y_axis='memory', row=row_var, col=col_var,ylog=True, fixed_vars=fixed_vars)
        plot_results(results_df=results_df, approaches=approaches, x_axis=x_axis, y_axis='recon_loss', row=row_var, col=col_var,ylog=True, fixed_vars=fixed_vars)
        plot_results(results_df=results_df, approaches=approaches, x_axis=x_axis, y_axis='obs_loss', row=row_var, col=col_var,ylog=True, fixed_vars=fixed_vars)
        plot_results(results_df=results_df, approaches=approaches, x_axis=x_axis, y_axis='intv_loss', row=row_var, col=col_var,ylog=True, fixed_vars=fixed_vars)
        plot_results(results_df=results_df, approaches=approaches, x_axis=x_axis, y_axis='graph_loss', row=row_var, col=col_var,ylog=True, fixed_vars=fixed_vars)
        plot_results(results_df=results_df, approaches=approaches, x_axis=x_axis, y_axis='surjective', row=row_var, col=col_var,ylog=False, fixed_vars=fixed_vars)
        #plot_results(results_df=results_df, approaches=approaches, x_axis=x_axis, y_axis='graph_loss', row=row_var, col=col_var)
    else:
        print("No CSV files found in the directory.")


    
    

if __name__ == '__main__':
    from pathlib import Path
    EASY_BENCHMARK_RUN = True
    EASY_BENCHMARK_PLOT = True

    #cut_off_good_results = 20260618-100000

    if EASY_BENCHMARK_RUN:
        easy_benchmark(heterogeneity_list=[1],internal_density_list=[0], blockiness_list=[0.5], num_trials=10)
    if EASY_BENCHMARK_PLOT:
        plot_easy_benchmark(cut_off="20260623-170000", approaches=['hm_graph_topo','hm_graph','hc_fast','random'],x_axis='k',col_var='n_high',
                            fixed_vars={'heterogeneity':1,'internal_density':0, 'blockiness':0.5})