import sys
import os

# Get the directory where main.py is (/experiments)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (/linear-causal-abstraction)
project_root = os.path.dirname(current_dir)

# Force the project root to the VERY BEGINNING of the search path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now it WILL find algorithms
from algorithms.gumbel_grad import gumbel_grad
import numpy as np
from utils.generate_data import random_dag,generate_random_T,sample_gennorm_lingam_population_standardized
from causabs.sampling import sample_linear_concretization
from algorithms.gumbel_grad import gumbel_grad
from algorithms.clustering import new_unmerge,matrix_to_clusters
from lingam import DirectLiNGAM
from utils.losses import rand_index,frob_norm_T,frob_norm_A,reconstruction_rmse
import time
import pandas as pd
from utils.plots import plot_experiment_results, plot_matrix

class Metric:
    def __init__(self,name,function):
        self.name = name
        self.function = function
    
    def __call__(self,*args,**kwds):
        return self.function(*args,**kwds)

class ExperimentConfig:
    seed: int = 42
    num_repeats: int = 5
    methods: dict = {
        "gumbel_grad": gumbel_grad,
        # Add other methods here as needed
    }
    metrics: dict = {
        "rand_index": Metric("rand_index", rand_index), 
        "frobenius_norm (T)": Metric("frobenius_norm", frob_norm_T), 
        "frobenius_norm (A)": Metric("frobenius_norm", frob_norm_A),
        #"Reconstruction RMSE": Metric("reconstruction_rmse", reconstruction_rmse),
    }

class Method:
    name: str
    function: callable
    params: dict

    def __init__(self, name, function, params={}):
        self.name = name
        self.function = function
        self.params = params
    
    def __call__(self, *args, **kwds):
        if self.params is None:
            return self.function(*args, **kwds)
        else:
            return self.params

def run(n_low,n_high,edge_density,weight_mean,weight_std,experiment_config=ExperimentConfig()):
    """
    Generate ground truth A_high and abstraction matrix T_gt. Then sample A_low such that A_high is a T-abstraction of A_low.
    Use varying methods to learn T and A_high from A_low and record:

    - Rand Index between learned T and T_gt
    - Frobenius norm between learned A_high and ground truth A_high
    - Runtime
    """
    np.random.seed(seed=experiment_config.seed)
    results_list = []
    for repeat in range(experiment_config.num_repeats):
        #Generate ground truth high-level causal model
        A_high_gt = random_dag(n_high, edge_density, weight_mean, weight_std).astype(np.float64)
        T_gt = generate_random_T(n_low, n_high).astype(np.float64)
        block_size = list(np.sum((T_gt!=0),axis=0).astype(int))+[0]
        A_low,S = sample_linear_concretization(A_high_gt, T_gt, block_size=block_size)
        E_low = np.linalg.inv(np.eye(n_low)-A_low)

        X_train = sample_gennorm_lingam_population_standardized(A_low, n_samples=1000, gennorm_param=4.0, seed=experiment_config.seed+repeat)
        X_test = sample_gennorm_lingam_population_standardized(A_low, n_samples=500, gennorm_param=4.0, seed=experiment_config.seed+repeat+100)

        for name,method in experiment_config.methods.items():
            start_time = time.time()
            best_model,_ = gumbel_grad(E_low,n_high)
            runtime = time.time() - start_time
            T_pred = best_model._T().detach().cpu().numpy()
            E_high_pred = best_model._E_high().detach().cpu().numpy()
            A_high_pred = np.eye(n_high) - np.linalg.inv(E_high_pred)
            E_low_pred = None #new_unmerge(matrix_to_clusters(T_pred), E_high_pred)
            
            results_list.append({
                "method": name,
                "repeat": repeat,
                "metric": "runtime",
                "score": runtime
            })
            #Compute metrics
            for metric_name, metric_func in experiment_config.metrics.items():
                if metric_func is not None:
                    score = metric_func(T_pred=T_pred, T_gt=T_gt,S_gt=S, A_pred=A_high_pred, A_gt=A_high_gt,E_pred=E_low_pred,E_gt=E_low,X=X_test)
                else:
                    score = None

                results_list.append({
                    "method": name,
                    "repeat": repeat,
                    "metric": metric_name,
                    "score": score
                })
            

            print(f"Method {name}, Repeat {repeat+1}/{experiment_config.num_repeats} completed.")

    plot_matrix(
        matrix_list=[A_high_gt, A_high_pred],
        title_list=['Ground Truth A_high', 'Learned A_high'],
        fig_title='Comparison of Ground Truth and Learned High-Level Adjacency Matrices'
    )
    plot_matrix(
        matrix_list=[T_gt, T_pred],
        title_list=['Ground Truth T', 'Learned T'],
        fig_title='Comparison of Ground Truth and Learned Abstraction Matrices'
    )
    
    return pd.DataFrame(results_list)


if __name__ == "__main__":
    n_low = 20
    n_high = 5
    edge_density = 0.3
    weight_mean = 1.
    weight_std = 2.5

    results = run(n_low, n_high, edge_density, weight_mean, weight_std)
    print("Experiment Results:")
    print(results)
    plot_experiment_results(results)