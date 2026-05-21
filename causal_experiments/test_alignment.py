import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))
import numpy as np
import torch
import math
from algorithms.alignment import find_valid_alignment, hierarchical_clustering_heap
from algorithms.gumbel_grad import TrainingConfig, gumbel_grad, gg_output
from utils.plots import plot_matrix, visualize_causal_graph,plot_results
from utils.generate_data import sample_low_level_graph, sample_gennorm_lingam_population_standardized
from utils.losses import reconstruction_loss, reconstruction_loss_with_data, kl_loss, observation_loss, intervention_loss
from utils.helper import clusters_to_matrix, reconstruct_adj
from utils.methods import galm_binary,hc,hc_galm,hail_mary
from lingam import DirectLiNGAM

LOSSES = {'recon_loss':reconstruction_loss, 'recon_loss_data':reconstruction_loss_with_data,
        'obs_loss':observation_loss, 'intv_loss':intervention_loss}
METHODS = {'hail_mary':hail_mary, 'hc':hc}

def run_abstraction_with_adjacency(n_high, n_low, density, marg_prob=0., n_test_samples=1000,
                                   gennorm_param=1.2, seed=None, trial=0, prior_info=0,
                                   METHODS= METHODS, LOSSES= LOSSES):
    results = []
    
    A_low,T_gt,S_gt,A_high_gt = sample_low_level_graph(n_high=n_high,n_low=n_low,density=density,seed=seed,shuffle=True,internal=True,marg_prob=marg_prob)
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

    X_test = sample_gennorm_lingam_population_standardized(A_low,n_samples=n_test_samples,gennorm_param=gennorm_param,seed=seed+1)
    for method_name,method_fn in METHODS.items():
        start_time = time.time()
        S_dict, T_dict, A_high_dict = method_fn(A_low=A_low,E_low=E_low,init_assignment=init_assignment)
        runtime = time.time() - start_time
        for k in range(2,n_low):
            S_pred = S_dict[k]
            T_pred = T_dict[k]
            A_high_pred = A_high_dict[k]
            A_low_recon = reconstruct_adj(A_high_pred,T_pred)
            result = { "Approach":method_name,
                       "n_high":n_high,
                       "n_low":n_low,
                       "k":k,
                       "ratio":int(n_low/n_high),
                       "density":density,
                       "marg_prob":marg_prob,
                       "trial":trial,
                       "runtime": runtime,
                        }
            result = result | {loss_name:loss_fn(A_high=A_high_pred, A_low=A_low, E_low=E_low, A_hat=A_low_recon, S=S_pred, T=T_pred, X=X_test) 
                               for loss_name,loss_fn in LOSSES.items()}
            
            results.append(result)
    
    return results

def run_abstraction_with_data(n_high,n_low,density,n_samples,gennorm_param=1.2,n_test_samples=1000,marg_prob=0,seed=None,trial=0):
    results = []

    A_low_gt,T_gt,S_gt,A_high_gt = sample_low_level_graph(n_high=n_high,n_low=n_low,density=density,seed=seed,shuffle=True,internal=True)
    X_train = sample_gennorm_lingam_population_standardized(A_low_gt,n_samples,gennorm_param=gennorm_param,seed=seed)
    X_test = sample_gennorm_lingam_population_standardized(A_low_gt,n_samples=n_test_samples,gennorm_param=gennorm_param,seed=seed+1)

    A_lingam = DirectLiNGAM().fit(X_train).adjacency_matrix_.T
    E_lingam = torch.linalg.inv(torch.eye(n_low) - A_lingam).to(torch.float32)
    start_time = time.time()
    clusters_dict, A_high_dict = hierarchical_clustering_heap(A_lingam,n_high=1)

    clusters = clusters_dict[n_high]
    A_high_binary = A_high_dict[n_high] #binary matrix
    T_pred_binary = clusters_to_matrix(clusters,n_low) #binary matrix
    E_high_binary = torch.linalg.inv(torch.eye(n_high) - A_high_binary)

    fine_tune_epochs = int(500 * n_high * np.log10(n_low))
    train_config = TrainingConfig(alignment_epochs=0, fine_tune_epochs=fine_tune_epochs, lr=0.05, acceptable_loss=0, entropy_pen=0., scale_pen=1e-6, verbose=False, threshold= 1e-8, temp=10.0, temp_min=0.1, anneal_rate=0.0005, hard=True)
    model,_,final = gumbel_grad(E_lingam,n_high,train_config,W=torch.tensor(T_pred_binary != 0).float(),E_high_mask=E_high_binary)
    runtime = time.time() - start_time
    partition_mask, T_pred, S_pred, E_high_pred = gg_output(model,E_lingam)
    A_high_pred = np.eye(n_high) - np.linalg.inv(E_high_pred)

    
    return results

if __name__ == "__main__":
    from pathlib import Path
    import time
    import pandas as pd

    seed = 42
    n_high_list = range(10,40,10)
    ratio_list = [2**d for d in range(2,3)]
    density_list = [0.4]
    num_trials = 1
    n_samples_list = [100 * 2**d for d in range(4) ]

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    RUN_ABSTRACTION_WITH_ADJACENCY = True

    
    if RUN_ABSTRACTION_WITH_ADJACENCY:
        filename = f"results/adjacency/{timestamp}.csv"
        os.makedirs(os.path.dirname(filename),exist_ok=True)
        results_list = []
        for n_high in n_high_list:
            for ratio in ratio_list:
                n_low = math.ceil(ratio*n_high)
                for density in density_list:
                    t_start = time.time()
                    for trial in range(num_trials):
                        new_results = run_abstraction_with_adjacency(n_high=n_high,n_low=n_low,density=density,seed=seed,trial=trial)
                        results_list.extend(new_results)
                    
                    print(f"Time Taken (n_high={n_high}, n_low={n_low}, density={density}): {time.time()-t_start:.6f}s")
    
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    
    
    cut_off = "20260520-210000"
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
    plot_results(results_df,x_axis='n_low',y_axis='runtime',fixed_vars={'ratio':2},row=None,col=None)
    for loss_name in LOSSES.keys():
        plot_results(results_df[results_df['k'] >= results_df['n_high']], x_axis='k',y_axis=loss_name,fixed_vars={'ratio':4},row=None,col='n_high')
    
    

