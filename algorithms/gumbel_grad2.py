import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
# Instead of a formal package import, ensure your local folder is hit
import sys
import os

# Add the current directory to the path to prioritize local files
sys.path.insert(0, os.path.abspath(os.getcwd()))

from utils.plots import plot_matrix, visualize_causal_graph
from utils.generate_data import random_dag, generate_random_T
from causabs.sampling import sample_linear_concretization

def loss(S,T,E,Y=None):
    p,k = S.shape
    if Y is None:
        Y = torch.zeros(k,p)
    T_mask = (T != 0).float()
    #U, S_vals, Vh = torch.linalg.svd(E@T, full_matrices=False)
    #cond_penalty = torch.sum(1.0 / (S_vals + 1e-3))
    #A_tilde = (torch.linalg.pinv(E@T) + Y @ (torch.eye(p)- E@T @ torch.linalg.pinv(E@T))) @ (E@T - S)
    #dag_loss = (torch.trace(torch.exp((A_tilde * A_tilde))) - k)**2
    intv_loss = 0.
    for j in range(k):
        e_j = F.one_hot(torch.tensor(j),num_classes=k).float().reshape(k,1)
        F_ss = torch.diag((T_mask@e_j).squeeze()) 
        I = torch.eye(p)
        residual = (E-I) @ (torch.linalg.inv(I + F_ss@(E-I))@ F_ss@S - F_ss@T)
        intv_loss += torch.norm(residual)**2

    return torch.tensor(0.), intv_loss, torch.tensor(0.)

@dataclass
class GumbelConfig:
    temp: float = 5.0
    temp_min: float = 0.1
    anneal_rate: float = 0.0005
    hard: bool = True
    fine_tune_epochs: int = 2000

def gg2(E,n_high,num_iters=1000, lr=1e-3, gumbel_config=GumbelConfig()):
    p = E.shape[0]
    S = torch.randn(p,n_high, requires_grad=True)
    T = torch.randn(p,n_high, requires_grad=True)
    W = torch.randn(p,n_high, requires_grad=True)
    Y = torch.zeros(n_high,p, requires_grad=True)
    optimizer = torch.optim.Adam([S,T,W,Y], lr=lr)
    temp = gumbel_config.temp
    for i in range(num_iters):
        optimizer.zero_grad()
        T_mask = F.gumbel_softmax(W,tau=temp,hard=gumbel_config.hard,dim=1)
        masked_T = T_mask * T
        masked_S = T_mask * S
        
        try:
            dag_loss, intv_loss, cond_penalty = loss(masked_S,masked_T,E,Y)
        except:
            import pdb;pdb.set_trace()
    
        total_loss = dag_loss + intv_loss + 0.1*cond_penalty
        total_loss.backward()
        optimizer.step()
        temp = max(gumbel_config.temp_min, temp* (1 - gumbel_config.anneal_rate))
        if i % 100 == 0:
            print(f"Iter {i}: Total Loss={total_loss.item():.4f}, DAG Loss={dag_loss.item():.4f}, Interv Loss={intv_loss.item():.4f}")

    print("Final T Mask:")
    print(T_mask)
    print("Final T:")
    print(masked_T)
    print("Final S:")
    print(masked_S)
    print("ET:")
    print(E@masked_T)
if __name__ == "__main__":
    seed = None
    n_low = 5
    n_high = 3
    #A_low = random_dag(n_low, edge_density=0.5, weight_mean=0, weight_std=1, seed=seed)
    A_high_gt = random_dag(n_high, edge_density=0.5, weight_mean=0, weight_std=2, seed=seed)
    T_gt = generate_random_T(n_low, n_high, T_range=(-1.5, 1.5), seed=seed)
    block_size = list(np.sum((T_gt!=0),axis=0).astype(int))+[0]
    A_low,S_gt= sample_linear_concretization(A_high_gt,T_gt,block_size,internal=False)
    E_low = np.linalg.inv(np.eye(n_low) - A_low)

    gg2(torch.from_numpy(E_low).float(), n_high, num_iters=2000, lr=1e-3)