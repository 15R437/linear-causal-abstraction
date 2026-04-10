import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))
import numpy as np
import torch
import math
import pandas as pd
import matplotlib.pyplot as plt
import time
import itertools
from algorithms.alignment import find_valid_alignment, hierarchical_clustering_heap
from algorithms.gumbel_grad import TrainingConfig, gumbel_grad, gg_output
from utils.plots import plot_matrix, visualize_causal_graph,plot_results
from utils.generate_data import sample_low_level_graph, sample_gennorm_lingam_population_standardized
from utils.losses import find_lost_weights, rand_index, frob_norm, reconstruction_rmse
from utils.helper import clusters_to_matrix, reconstruct_adj
from lingam import DirectLiNGAM

def run_clustering(n_high,n_low,density,marg_prob=0,seed=None,trial=0,approaches={"HC":hierarchical_clustering_heap}):
    results = []
    
    A_low,T_gt,S_gt,A_high_gt = sample_low_level_graph(n_high=n_high,n_low=n_low,density=density,seed=seed,shuffle=True,internal=False,marg_prob=marg_prob)
    
    for NAME,func in approaches.items():
        start_time = time.time()
        clusters_dict, A_high_dict = func(A_low,n_high=1)
        runtime = time.time() - start_time

        for k in range(1,n_low):
            clusters = clusters_dict[k]

            lost_weights = find_lost_weights(A_low,clusters)
            l1_loss = np.sum(np.abs(lost_weights)).item()
            results.append({
                "Approach":NAME,
                "n_high":n_high,
                "n_low":n_low,
                "ratio":int(n_low/n_high),
                "density":density,
                "marge_prob":marg_prob,
                "trial":trial,
                "runtime": runtime,
                "lost_weight_l1": l1_loss,
                "lost_weight_count": len(lost_weights),
                "marginalised_count": len(clusters[-1]),
                "marginalised_ratio": len(clusters[-1])/n_low,
                "k":k
                })
    
    return results

def run_abstraction_with_adjacency(n_high,n_low,density,marg_prob=0,n_test_samples=1000,gennorm_param=1.2,seed=None,trial=0):
    results = []
    
    A_low,T_gt,S_gt,A_high_gt = sample_low_level_graph(n_high=n_high,n_low=n_low,density=density,seed=seed,shuffle=True,internal=False,marg_prob=marg_prob)
    E_low = torch.linalg.inv(torch.eye(n_low) - A_low).to(torch.float32)
    X_test = sample_gennorm_lingam_population_standardized(A_low,n_samples=n_test_samples,gennorm_param=gennorm_param,seed=seed+1)
    start_time = time.time()
    clusters_dict, A_high_dict = hierarchical_clustering_heap(A_low,n_high=n_high)

    clusters = clusters_dict[n_high]
    A_high_binary = A_high_dict[n_high] #binary matrix
    T_pred_binary = clusters_to_matrix(clusters,n_low) #binary matrix
    E_high_binary = torch.linalg.inv(torch.eye(n_high) - A_high_binary)

    fine_tune_epochs = int(500 * n_high * np.log10(n_low))
    train_config = TrainingConfig(alignment_epochs=0, fine_tune_epochs=fine_tune_epochs, lr=0.05, acceptable_loss=0, entropy_pen=0., scale_pen=1e-6, verbose=False, threshold= 1e-8, temp=10.0, temp_min=0.1, anneal_rate=0.0005, hard=True)
    model,_,final = gumbel_grad(E_low,n_high,train_config,W=torch.tensor(T_pred_binary != 0).float(),E_high_mask=E_high_binary)
    runtime = time.time() - start_time
    partition_mask, T_pred, S_pred, E_high_pred = gg_output(model,E_low)
    A_high_pred = np.eye(n_high) - np.linalg.inv(E_high_pred)

    A_hat = reconstruct_adj(A_high_pred,T_pred)
    results.append({
        "Approach":'HC+GG',
        "n_high":n_high,
        "n_low":n_low,
        "ratio":int(n_low/n_high),
        "density":density,
        "marge_prob":marg_prob,
        "trial":trial,
        "runtime": runtime,
        "recon_rmse": reconstruction_rmse(A_hat,X_test),
        "Normalised A_error": frob_norm(A_hat,A_low),
        "obs_loss": final['obs_loss'].item(),
        "interv_loss": final['interv_loss'].item(),
        })
    
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

    A_hat = reconstruct_adj(A_high_pred,T_pred)
    results.append({
        "Approach":'HC+GG',
        "n_high":n_high,
        "n_low":n_low,
        "ratio":int(n_low/n_high),
        "density":density,
        "marge_prob":marg_prob,
        "n_samples":n_samples,
        "trial":trial,
        "runtime": runtime,
        "recon_rmse": reconstruction_rmse(A_hat,X_test),
        "obs_loss": final['obs_loss'].item(),
        "interv_loss": final['interv_loss'].item(),
        })
    
    return results

if __name__ == "__main__":
    seed = 42
    n_high_list = range(10,20,10)
    ratio_list = [2**d for d in range(1,2)]
    density_list = [0.4]
    num_trials = 1
    n_samples_list = [100 * 2**d for d in range(4) ]

    #timestamp = time.strftime("%Y%m%d-%H%M%S")
    RUN_CLUSTERING = False
    RUN_ABSTRACTION_WITH_ADJACENCY = False
    RUN_ABSTRACTION_WITH_DATA = False
    OPEN = True
    open_file = 'results/initial_abstraction_with_data.csv'
    if OPEN:
        results_df = pd.read_csv(open_file)

    
    if RUN_CLUSTERING:
        filename = 'results/initial_clustering.csv' #f"results/initial-{timestamp}.csv"
        os.makedirs(os.path.dirname(filename),exist_ok=True)
        results_list = []
        for n_high in n_high_list:
            for ratio in ratio_list:
                n_low = math.ceil(ratio*n_high)
                for density in density_list:
                    t_start = time.time()
                    for trial in range(num_trials):
                        new_results = run_clustering(n_high=n_high,n_low=n_low,density=density,seed=seed,trial=trial)
                        results_list.extend(new_results)
                    
                    print(f"Time Taken (n_high={n_high}, n_low={n_low}, density={density}): {time.time()-t_start:.6f}s")
    
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    
    if RUN_ABSTRACTION_WITH_ADJACENCY:
        filename = 'results/initial_abstraction_with_adjacency.csv' #f"results/initial-{timestamp}.csv"
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
    
    if RUN_ABSTRACTION_WITH_DATA:
        filename = 'results/initial_abstraction_with_data.csv' #f"results/initial-{timestamp}.csv"
        os.makedirs(os.path.dirname(filename),exist_ok=True)
        results_list = []
        for n_high in n_high_list:
            for ratio in ratio_list:
                n_low = math.ceil(ratio*n_high)
                for density in density_list:
                    t_start = time.time()
                    for n_samples in n_samples_list:
                        for trial in range(num_trials):
                            new_results = run_abstraction_with_data(n_high=n_high,n_low=n_low,density=density,n_samples=n_samples,seed=seed,trial=trial)
                            results_list.extend(new_results)
                    
                        print(f"Time Taken (n_high={n_high}, n_low={n_low}, n_samples={n_samples}): {time.time()-t_start:.6f}s")
    
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    
    #plot_results(results_df,x_axis='n_low',y_axis='lost_weight_count',fixed_vars={'n_high':10,'n_low':20},row=None,col=None)
    #plot_results(results_df,x_axis='k',y_axis='lost_weight_count',fixed_vars={'n_high':10,'n_low':20},row=None,col=None)
    plot_results(results_df,x_axis='n_samples',y_axis='recon_rmse',fixed_vars={'n_high':10,'n_low':20},row=None,col=None)
    plot_results(results_df,x_axis='n_samples',y_axis='obs_loss',fixed_vars={'n_high':10,'n_low':20},row=None,col=None)
    plot_results(results_df,x_axis='n_samples',y_axis='interv_loss',fixed_vars={'n_high':10,'n_low':20},row=None,col=None)
    #plot_results(results_df,x_axis='n_low',y_axis='obs_loss',fixed_vars={'ratio':2},row=None,col=None)
    #plot_results(results_df,x_axis='n_low',y_axis='interv_loss',fixed_vars={'ratio':2},row=None,col=None)
    

