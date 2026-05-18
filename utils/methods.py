import numpy as np
import torch

from algorithms.gg import gumbel_grad, gg_output
from algorithms.alignment import hierarchical_clustering
from algorithms.hail_mary import HailMary, process_output


"""Wrapper function for all the algorithms."""

def ground_truth(A_low,**kwds):
    pass

def galm_core(E_low, k, T_type='binary', init_assignment=None, warm_start=None, **kwds):
    E_low_tensor = torch.from_numpy(E_low).to(torch.float32)
    best_model,_ = gumbel_grad(E_low_tensor,k,T_type=T_type,init_assignment=init_assignment, warm_start=warm_start)
    partition_mask, T_pred, S_pred, E_high, A_high, final_losses = gg_output(best_model,E_low)

    return S_pred, T_pred, A_high

def hail_mary_core(E_low, k, init_assignment=None, **kwds):
    E_low_tensor = torch.from_numpy(E_low).to(torch.float32)
    model = HailMary(n_low=E_low_tensor.shape[0], n_high=k, E=E_low_tensor, init_assignment=init_assignment)
    model.train(num_epochs=1000, verbose=False)
    S, T, E_high, W, A = model.get_params()
    S_pred, T_pred, A_high = process_output(S, T, E_high)

    return S_pred, T_pred, A_high

def hail_mary(E_low, init_assignment=None, **kwds):
    n_low = E_low.shape[0]
    S_dict = {}
    T_dict = {}
    A_high_dict = {}
    for k in range(2,n_low):
        S_pred, T_pred, A_high = hail_mary_core(E_low=E_low, k=k, init_assignment=init_assignment)
        S_dict[k] = S_pred
        T_dict[k] = T_pred
        A_high_dict[k] = A_high

    return S_dict, T_dict, A_high_dict

def galm(E_low, T_type='binary', init_assignment=None,**kwds):
    n_low = E_low.shape[0]
    S_dict = {}
    T_dict = {}
    A_high_dict = {}
    for k in range(2,n_low):
        S_pred, T_pred, A_high = galm_core(E_low=E_low, k=k, init_assignment=init_assignment, T_type=T_type)
        S_dict[k] = S_pred
        T_dict[k] = T_pred
        A_high_dict[k] = A_high

    return S_dict, T_dict, A_high_dict

def galm_binary(E_low, init_assignment=None, **kwds):
    return galm(E_low, T_type='binary', init_assignment=init_assignment)

def galm_simplex(E_low, init_assignment=None, **kwds):
    return galm(E_low, T_type='simplex', init_assignment=init_assignment)

def galm_sphere(E_low, init_assignment=None, **kwds):
    return galm(E_low, T_type='sphere', init_assignment=init_assignment)

def hc(A_low, init_assignment=None, **kwds):
    return hierarchical_clustering(A_low=A_low, init_assignment=init_assignment)

def hc_galm(A_low, E_low, init_assignment=None, **kwds):
    n_low = A_low.shape[0]
    S_dict, T_dict, A_high_dict = hc(A_low, init_assignment=init_assignment)
    S_dict_new = {}
    T_dict_new = {}
    A_high_dict_new = {}
    for k in range(2,n_low):
        warm_start = T_dict[k]
        assert warm_start.shape == (n_low,k), f"Expected shape {n_low,k}. Got f{warm_start.shape}."
        S_pred, T_pred, A_high = galm_core(E_low=E_low, k=k, init_assignment=init_assignment, warm_start=warm_start)
        S_dict_new[k] = S_pred
        T_dict_new[k] = T_pred
        A_high_dict_new[k] = A_high
    
    return S_dict_new, T_dict_new, A_high_dict_new