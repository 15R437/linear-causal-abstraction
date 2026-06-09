import numpy as np
import torch

from algorithms.gg import gumbel_grad, gg_output
from algorithms.alignment import hierarchical_clustering, heuristic_loss, abstraction_residual
from algorithms.hail_mary import HailMary, process_output
from utils.helper import get_S, get_A_high


"""Wrapper function for all the algorithms."""

def ground_truth(T_gt, S_gt, A_high_gt, **kwds):
    return S_gt, T_gt, A_high_gt

def random(A_low, k, **kwds):
    return hail_mary(A_low, k, num_epochs=0)


def hail_mary(A_low, k, logits_init=None, num_epochs=None, **kwds):
    A_low_tensor = torch.from_numpy(A_low).to(torch.float32)
    n_low = A_low.shape[0]
    model = HailMary(n_low=A_low_tensor.shape[0], n_high=k, A_low=A_low_tensor, logits_init=logits_init)
    if num_epochs is None:
        num_epochs = 1000 * int(np.log(n_low*k)+1)
    model.train(num_epochs=num_epochs, verbose=False)

    S_pred, T_pred, A_high = process_output(model)

    return S_pred, T_pred, A_high

def hail_mary_abstract(A_low, k, logits_init=None, **kwds):
    A_low_tensor = torch.from_numpy(A_low).to(torch.float32)
    n_low = A_low.shape[0]
    model = HailMary(n_low=A_low_tensor.shape[0], n_high=k, A_low=A_low_tensor, logits_init=logits_init)
    num_epochs = 1000 * int(np.log(n_low*k)+1)
    model.train(num_epochs=num_epochs, compute_abs=True, compute_graph=False, compute_topo=False, verbose=False)

    S_pred, T_pred, A_high = process_output(model,dag_enforcer='khan')

    return S_pred, T_pred, A_high

def hail_mary_graph(A_low, k, logits_init=None, **kwds):
    A_low_tensor = torch.from_numpy(A_low).to(torch.float32)
    n_low = A_low.shape[0]
    model = HailMary(n_low=A_low_tensor.shape[0], n_high=k, A_low=A_low_tensor, logits_init=logits_init)
    num_epochs = 1000 * int(np.log(n_low*k)+1)
    model.train(num_epochs=num_epochs, compute_abs=False, compute_graph=True, compute_topo=False, verbose=False)

    S_pred, T_pred, A_high = process_output(model,dag_enforcer='khan')

    return S_pred, T_pred, A_high

def hail_mary_abstract_graph(A_low, k, logits_init=None, **kwds):
    A_low_tensor = torch.from_numpy(A_low).to(torch.float32)
    n_low = A_low.shape[0]
    model = HailMary(n_low=A_low_tensor.shape[0], n_high=k, A_low=A_low_tensor, logits_init=logits_init)
    num_epochs = 1000 * int(np.log(n_low*k)+1)
    model.train(num_epochs=num_epochs, compute_abs=True, compute_graph=True, compute_topo=False, verbose=False)

    S_pred, T_pred, A_high = process_output(model,dag_enforcer='khan')

    return S_pred, T_pred, A_high

def hail_mary_abstract_topo(A_low, k, logits_init=None, **kwds):
    A_low_tensor = torch.from_numpy(A_low).to(torch.float32)
    n_low = A_low.shape[0]
    model = HailMary(n_low=A_low_tensor.shape[0], n_high=k, A_low=A_low_tensor, logits_init=logits_init)
    num_epochs = 1000 * int(np.log(n_low*k)+1)
    model.train(num_epochs=num_epochs, compute_abs=True, compute_graph=False, compute_topo=True, verbose=False)

    S_pred, T_pred, A_high = process_output(model,dag_enforcer='self')

    return S_pred, T_pred, A_high

def hail_mary_graph_topo(A_low, k, logits_init=None, **kwds):
    A_low_tensor = torch.from_numpy(A_low).to(torch.float32)
    n_low = A_low.shape[0]
    model = HailMary(n_low=A_low_tensor.shape[0], n_high=k, A_low=A_low_tensor, logits_init=logits_init)
    num_epochs = 1000 * int(np.log(n_low*k)+1)
    model.train(num_epochs=num_epochs, compute_abs=False, compute_graph=True, compute_topo=True, verbose=False)

    S_pred, T_pred, A_high = process_output(model,dag_enforcer='self')

    return S_pred, T_pred, A_high

def hail_mary_sweep(A_low, init_assignment=None, **kwds):
    n_low = A_low.shape[0]
    S_dict = {}
    T_dict = {}
    A_high_dict = {}
    for k in range(2,n_low):
        S_pred, T_pred, A_high = hail_mary(A_low=A_low, k=k, init_assignment=init_assignment)
        S_dict[k] = S_pred
        T_dict[k] = T_pred
        A_high_dict[k] = A_high

    return S_dict, T_dict, A_high_dict

def hc_sweep(A_low, init_assignment=None, **kwds):
    return hierarchical_clustering(A_low=A_low,init_assignment=init_assignment)

def hc_fast(A_low, k, init_assignment=None, **kwds):
    S_dict, T_dict, A_high_dict = hierarchical_clustering(A_low=A_low, n_high=k, init_assignment=init_assignment, loss_list=[heuristic_loss], dag_learner='khan') 
    return S_dict[k], T_dict[k], A_high_dict[k]

def hc_fast_dag(A_low, k, init_assignment=None, **kwds):
    S_dict, T_dict, A_high_dict = hierarchical_clustering(A_low=A_low, n_high=k, init_assignment=init_assignment, loss_list=[heuristic_loss], dag_learner='learn') 
    return S_dict[k], T_dict[k], A_high_dict[k]

def hc_abs(A_low, k, init_assignment=None, **kwds):
    S_dict, T_dict, A_high_dict = hierarchical_clustering(A_low=A_low, n_high=k, init_assignment=init_assignment, loss_list=[abstraction_residual, heuristic_loss]) 
    return S_dict[k], T_dict[k], A_high_dict[k]
