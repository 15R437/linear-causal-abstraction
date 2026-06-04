import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))
import time
import torch
import numpy as np
import pandas as pd
import itertools
from utils.generate_data import generate_easy_benchmark, sample_low_level_graph
from utils.helper import get_A_high, get_S
from utils.losses import abstraction_residual, observation_loss, intervention_loss, graph_loss
from utils.methods import hc_abs, hc_fast, hail_mary, hail_mary_abstract, hail_mary_graph, hail_mary_graph_topo, ground_truth
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
           'hm_graph_topo': hail_mary_graph_topo, 
           'hc_fast': hc_fast, 'hc_abs': hc_abs,
           'ground_truth': ground_truth,
           'random': lambda A_low, k, **kwds: hail_mary(A_low, k, num_epochs=0, **kwds)}

data_generating_methods = {'easy': generate_easy_benchmark, 'standard':sample_low_level_graph}

# regime = (n_low, n_high_list, method_name_list)
regimes = [(50,range(5,55,5),['hail_mary','hm_abstract','hm_graph','hm_graph_topo','hc_fast']),
            #(2,range(60,110,10),['hail_mary','hc_fast']),
            #(5,range(10,43,3),['hail_mary','hc_fast']),
            #(10,range(10,21),['hail_mary','hc_fast'])
            ]
def easy_benchmark(num_trials=10,regime_list=regimes,
                   density_list=[None],
                   blockiness_list=[None],
                   cluster_balance_list=[None],
                   flip_prop_list=[None],
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
        for regime, density, blockiness, cluster_balance, flip_prop in itertools.product(regime_list, density_list, blockiness_list, cluster_balance_list, flip_prop_list):
            n_low,n_high_list,method_name_list = regime
            density = density if density is not None else 0.4
            blockiness = blockiness if blockiness is not None else 0.5
            cluster_balance = cluster_balance if cluster_balance is not None else 0.1
            for n_high in n_high_list:
                A_low, T_gt, S_gt, A_high_gt = generate_data_fn(n_low=n_low,n_high=n_high,density=density,blockiness=blockiness,cluster_balance=cluster_balance, shuffle=True, internal=False)
                T_init = None if flip_prop is None else flip_T(T_gt, flip_prop)
                        
                E_low = np.linalg.inv(np.eye(n_low) - A_low)
                for method_name in method_name_list:
                    method_fn = METHODS[method_name]
                    S_pred, T_pred, A_high_pred = method_fn(A_low=A_low, k=n_high, logits_init=T_init, T_gt=T_gt, S_gt=S_gt, A_high_gt=A_high_gt)
                    abs_residual = abstraction_residual(A_low=A_low,T=T_pred,A_high=A_high_pred)
                    new_result = [{'Approach': method_name,
                                    'n_low': n_low,
                                    'n_high': n_high,
                                    'density': density,
                                    'cluster_balance':cluster_balance,
                                    'flip_var':flip_prop,
                                    'accuracy': np.isclose(abs_residual, 0, atol=1e-6) * 1.,
                                    'abstraction_residual': abs_residual,
                                    'obs_loss': observation_loss(A_low=A_low,A_high=A_high_pred,S=S_pred,T=T_pred),
                                    'intv_loss': intervention_loss(E_low=E_low,S=S_pred,T=T_pred),
                                    'graph_loss': graph_loss(A_low=A_low, T=T_pred)
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
    
def plot_easy_benchmark(cut_off = "20260528-120000", x_axis='n_high', col_var=None, row_var=None):
    results_dir = Path("results/sandbox")
    csv_files = [f for f in results_dir.glob("*.csv") if f.stem >= cut_off]

    df_list = []
    for f in csv_files:
        data = pd.read_csv(f)
        df_list.append(data)

    if df_list:
        results_df = pd.concat(df_list, ignore_index=True)
        print(f"Collated {len(df_list)} file(s) saved after {cut_off}. Total rows: {len(results_df)}")
    
        plot_results(results_df=results_df, x_axis=x_axis, y_axis='accuracy', row=row_var, col=col_var)
        plot_results(results_df=results_df, x_axis=x_axis, y_axis='abstraction_residual', row=row_var, col=col_var,ylog=False)
        plot_results(results_df=results_df, x_axis=x_axis, y_axis='obs_loss', row=row_var, col=col_var,ylog=False)
        plot_results(results_df=results_df, x_axis=x_axis, y_axis='intv_loss', row=row_var, col=col_var,ylog=False)
        #plot_results(results_df=results_df, x_axis=x_axis, y_axis='graph_loss', row=row_var, col=col_var)
    else:
        print("No CSV files found in the directory.")


    
    
    

if __name__ == '__main__':
    from pathlib import Path
    EASY_BENCHMARK_RUN = True
    EASY_BENCHMARK_PLOT = True

    if EASY_BENCHMARK_RUN:
        easy_benchmark(flip_prop_list=[None],blockiness_list=[None],num_trials=1)
    if EASY_BENCHMARK_PLOT:
        plot_easy_benchmark(cut_off="20260603-140000",x_axis='n_high')
    
    
    