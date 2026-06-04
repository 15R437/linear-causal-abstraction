import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))
import tracemalloc
import numpy as np
import torch
import math
from algorithms.gumbel_grad import TrainingConfig, gumbel_grad, gg_output
from utils.plots import plot_matrix, visualize_causal_graph,plot_results
from utils.generate_data import sample_low_level_graph, sample_gennorm_lingam_population_standardized
from utils.losses import reconstruction_loss, reconstruction_loss_with_data, kl_loss, observation_loss, intervention_loss, abstraction_residual
from utils.helper import clusters_to_matrix, reconstruct_adj
from utils.methods import hc_fast,hc_abs,hail_mary
from lingam import DirectLiNGAM

LOSSES = {'recon_loss':reconstruction_loss, 'recon_loss_data':reconstruction_loss_with_data,
        'obs_loss':observation_loss, 'intv_loss':intervention_loss, 'abs_residual':abstraction_residual}
METHODS = {'hail_mary':hail_mary, 'hc_fast':hc_fast, 'hc_abs':hc_abs}

def run_abstraction_with_adjacency(n_high, n_low, density, marg_prob=0., n_test_samples=1000,
                                   gennorm_param=1.2, seed=None, trial=0, prior_info=0,
                                   METHODS= METHODS, LOSSES= LOSSES):
    results = []
    
    A_low,T_gt,S_gt,A_high_gt = sample_low_level_graph(n_high=n_high,n_low=n_low,density=density,seed=seed,shuffle=True,internal=False,marg_prob=marg_prob)
    E_low = np.linalg.inv(np.eye(n_low) - A_low)

    # prior information
    num_vars = min(int(prior_info * n_low),n_low)
    if num_vars > 0:
        low_level_vars = np.random.choice(n_low,num_vars,replace=False)
        high_level_vars = np.nonzero(T_gt[low_level_vars])[1]
        init_assignment = np.ones(n_low) * -1
        init_assignment[low_level_vars] = high_level_vars
    else:
        init_assignment = None

    X_test = sample_gennorm_lingam_population_standardized(A_low,n_samples=n_test_samples,gennorm_param=gennorm_param,seed=seed)
    for method_name,method_fn in METHODS.items():
        tracemalloc.start()
        start_time = time.perf_counter()
        S_pred, T_pred, A_high_pred = method_fn(A_low=A_low,E_low=E_low, k=n_high,init_assignment=init_assignment)
        runtime = time.perf_counter() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        A_low_recon = reconstruct_adj(A_high_pred,T_pred)
        result = { "Approach":method_name,
                    "n_high":n_high,
                    "n_low":n_low,
                    "ratio":int(n_low/n_high),
                    "density":density,
                    "marg_prob":marg_prob,
                    "gennorm_param":gennorm_param,
                    "trial":trial,
                    "runtime": runtime,
                    "memory": peak/1e6
                    }
        result = result | {loss_name:loss_fn(A_high=A_high_pred, A_low=A_low, E_low=E_low, A_hat=A_low_recon, S=S_pred, T=T_pred, X=X_test) 
                               for loss_name,loss_fn in LOSSES.items()}
        results.append(result)
        
        if result['intv_loss'] > 1 or result['obs_loss'] > 1:
            log_dir = "results/anomalous_graphs"
            os.makedirs(log_dir, exist_ok=True)
            filename = f"{log_dir}/{method_name}_{n_high}_{n_low}.npz"
            
            # 2. Save both matrices and metadata into one compressed file
            np.savez_compressed(
                filename,
                A_low=A_low,
                T_gt=T_gt,
                S_gt=S_gt,
                A_high_gt=A_high_gt,
                n_high=n_high,
                method_name=method_name,
                recon_loss_data=result['recon_loss_data'],
                obs_loss = result['obs_loss'],
                intv_loss = result['intv_loss']
            )

    return results

def plot_adjacency_results(cut_off = "20260529-120000", col_var=None, row_var=None):
    results_dir = Path("results/adjacency")
    csv_files = [f for f in results_dir.glob("*.csv") if f.stem >= cut_off]

    df_list = []
    for f in csv_files:
        data = pd.read_csv(f)
        df_list.append(data)

    if df_list:
        results_df = pd.concat(df_list, ignore_index=True)
        print(f"Collated {len(df_list)} file(s) saved after {cut_off}. Total rows: {len(results_df)}")
    
        plot_results(results_df=results_df, x_axis='n_low', y_axis='runtime', row=row_var, col=col_var)
        plot_results(results_df=results_df, x_axis='n_low', y_axis='memory', row=row_var, col=col_var)
        plot_results(results_df=results_df, x_axis='n_low', y_axis='abstraction_residual', row=row_var, col=col_var)
        plot_results(results_df=results_df, x_axis='n_low', y_axis='obs_loss', row=row_var, col=col_var)
        plot_results(results_df=results_df, x_axis='n_low', y_axis='intv_loss', row=row_var, col=col_var)
        
    else:
        print("No CSV files found in the directory.")
    

if __name__ == "__main__":
    from pathlib import Path
    import time
    import pandas as pd

    seed = None
    density_list = [0.4]
    num_trials = 3

    # regime = (n_low, n_high_list, method_name_list)
    regimes = [(60,range(10,60,10),['hail_mary','hc_fast','hc_abs']),
               #(2,range(60,110,10),['hail_mary','hc_fast']),
               #(5,range(10,43,3),['hail_mary','hc_fast']),
               #(10,range(10,21),['hail_mary','hc_fast'])
               ]

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    RUN_ABSTRACTION_WITH_ADJACENCY = False


    if RUN_ABSTRACTION_WITH_ADJACENCY:
        filename = f"results/adjacency/{timestamp}.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        header_written = False 
        
        for n_low, n_high_list, method_name_list in regimes:
            for n_high in n_high_list:
                for density in density_list:
                    t_start = time.time()
                    for trial in range(num_trials):
                        curr_seed = seed + trial if seed is not None else None
                        new_results = run_abstraction_with_adjacency(
                            n_high=n_high, n_low=n_low, density=density, seed=curr_seed, trial=trial,
                            METHODS={method_name:METHODS[method_name] for method_name in method_name_list}
                        )

                        df_batch = pd.DataFrame(new_results)
                        df_batch.to_csv(
                            filename, 
                            mode='a', 
                            index=False, 
                            header=not header_written
                        )
                        header_written = True
                    
                    print(f"Time Taken (n_high={n_high}, n_low={n_low}, density={density}): {time.time()-t_start:.6f}s")
    
    
    cut_off = "20260601-130000"
    results_dir = Path("results/adjacency")
    csv_files = [f for f in results_dir.glob("*.csv") if f.stem >= cut_off]

    df_list = []
    for f in csv_files:
        data = pd.read_csv(f)
        df_list.append(data)

    if df_list:
        results_df = pd.concat(df_list, ignore_index=True)
        print(f"Collated {len(df_list)} file(s) saved after {cut_off}. Total rows: {len(results_df)}")
    else:
        print("No CSV files found in the directory.")
    #plot_results(results_df,x_axis='n_low',y_axis='lost_weight_count',fixed_vars={'n_high':10,'n_low':20},row=None,col=None)
    #plot_results(results_df,x_axis='k',y_axis='lost_weight_count',fixed_vars={'n_high':10,'n_low':20},row=None,col=None)
    plot_results(results_df,x_axis='n_high',y_axis='runtime',fixed_vars={},row=None,col=None)
    plot_results(results_df,x_axis='n_high',y_axis='memory',fixed_vars={},row=None,col=None)
    for loss_name in LOSSES.keys():
        print(f"Showing a plot of {loss_name}...")
        plot_results(results_df, x_axis='n_high',y_axis=loss_name,fixed_vars={},row=None,col=None)
    
    

