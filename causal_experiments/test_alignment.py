import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import time
from algorithms.alignment import find_valid_alignment, hierarchical_clustering_heap
from utils.plots import plot_matrix, visualize_causal_graph
from utils.generate_data import random_dag, generate_random_T
from utils.losses import find_lost_weights
from causabs.sampling import sample_linear_concretization

def sample_low_level_graph(n_high,n_low,density,T_lower=(-2,1),T_upper=(-1,2),marg_prob=0,seed=None,max_attempts=10):
    adj_structure = (random_dag(n_high,density, seed=seed) != 0).astype(float)
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        weights = rng.normal(0, 0.5, size=adj_structure.shape)
        A_high = adj_structure * weights
        try:
            T_gt,block_size = generate_random_T(n_low, n_high, T_lower, T_upper, marg_prob=marg_prob, seed=seed)
            return sample_linear_concretization(A_high,T_gt,block_size,internal=False,alpha=1.,style='uniform')
        except AssertionError:
            pass
    
    raise RuntimeError(f"Could not find a stable model after {max_attempts} attempts.")

def run(n_high,n_low,density,seed=None):
    A_low,S_gt= sample_low_level_graph(n_high=n_high,n_low=n_low,density=density,seed=seed)

    clusters, A_high = hierarchical_clustering_heap(A_low,n_high=n_high)
    
    
    lost_weights = find_lost_weights(A_low,clusters)
    
    return A_low,clusters

if __name__ == "__main__":
    seed = 42
    n_high_list = range(10,110,10)
    lambda_list = [2**d for d in range(1,2)]
    density_list = [0.2,0.4,0.6,0.8]

    results = []
    count = 0
    start_time = time.time()
    for n_high in n_high_list:
        for l in lambda_list:
            n_low = math.ceil(l*n_high)
            t_start = time.time()
            for density in density_list:
                A_low,clusters = run(n_high=n_high,n_low=n_low,density=density,seed=seed)
                lost_weights = find_lost_weights(A_low,clusters)
                results.append({
                    "n_high":n_high,
                    "n_low":n_low,
                    "lambda":l,
                    "density":density,
                    "L1 lost weight": np.sum(np.abs(lost_weights)).item(),
                    "Count lost weight": len(lost_weights),
                    "Count marginalised": len(clusters[-1])
                })
                count += 1
            print(f"Time Taken (n_high={n_high}, n_low={n_low}): {time.time()-t_start:.6f}s")
    
    results_df = pd.DataFrame(results)

    