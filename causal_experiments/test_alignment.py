import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import time
from algorithms.alignment import find_valid_alignment, hierarchical_clustering_heap
from utils.plots import plot_matrix, visualize_causal_graph,plot_results
from utils.generate_data import random_dag, generate_random_T, sample_low_level_graph
from utils.losses import find_lost_weights, rand_index
from utils.helper import clusters_to_matrix
from causabs.sampling import sample_linear_concretization

def run(n_high,n_low,density,seed=None,trial=1):
    results = []
    
    A_low,T_gt,S_gt,A_high_gt = sample_low_level_graph(n_high=n_high,n_low=n_low,density=density,seed=seed)
    start_time = time.time()
    clusters_dict, A_high_dict = hierarchical_clustering_heap(A_low,n_high=1)
    runtime = time.time() - start_time

    #compute losses

    for k in range(1,n_low):
        clusters = clusters_dict[k]
        A_high = A_high_dict[k] 
        T_pred = clusters_to_matrix(clusters) #binary matrix
        lost_weights = find_lost_weights(A_low,clusters)
        l1_loss = np.sum(np.abs(lost_weights)).item()
        results.append({
            "Approach":'HC',
            "n_high":n_high,
            "n_low":n_low,
            "ratio":int(n_low/n_high),
            "density":density,
            "trial":trial,
            "lost_weight_l1": l1_loss,
            "lost_weight_count": len(lost_weights),
            "marginalised_count": len(clusters[-1]),
            "marginalised_ratio": len(clusters[-1])/n_low,
            "runtime": runtime,
            "rand_index": rand_index(T_gt,T_pred),
            "k":k
            })
    
    return results

if __name__ == "__main__":
    seed = 42
    n_high_list = range(10,110,10)
    ratio_list = [2**d for d in range(1,2)]
    density_list = [0.4]
    num_trials = 1

    RUN = False
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    #filename = f"results/alignment_exp_{timestamp}.csv"
    filename = f"results/initial.csv"
    os.makedirs(os.path.dirname(filename),exist_ok=True)

    if RUN:
        results_list = []
        for n_high in n_high_list:
            for ratio in ratio_list:
                n_low = math.ceil(ratio*n_high)
                for density in density_list:
                    t_start = time.time()
                    for trial in range(num_trials):
                        new_results = run(n_high=n_high,n_low=n_low,density=density,seed=seed,trial=trial)
                        results_list.extend(new_results)
                    
                    print(f"Time Taken (n_high={n_high}, n_low={n_low}, density={density}): {time.time()-t_start:.6f}s")
    
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    else:
        results_df = pd.read_csv(filename)
    
    plot_results(results_df,x_axis='k',y_axis='marginalised_ratio',fixed_vars={'n_high':50,'n_low':100},row=None,col=None)
    plot_results(results_df,x_axis='n_low',y_axis='marginalised_ratio',fixed_vars={'ratio':2},row=None,col=None)
    

