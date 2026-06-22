import numpy as np
import torch
import torch.nn.functional as F
from algorithms.gg import gumbel_grad, gg_output
from algorithms.alignment import hierarchical_clustering, heuristic_loss, abstraction_residual
from algorithms.hail_mary import HailMary
from utils.helper import process_output, get_S, get_A_high, regularised_surjective_init


"""Wrapper function for all the algorithms."""

def ground_truth(A_low, T_gt, E_low=None, **kwds):
    if E_low is None:
        E_low = np.linalg.inv(np.eye(A_low.shape[0]) - A_low)

    S_pred, T_pred, A_high = process_output(T_hard=T_gt, A_low=A_low, E_low=E_low)

    return S_pred, T_pred, A_high

def random(A_low, k, **kwds):
    n_low = A_low.shape[0]
    logits = regularised_surjective_init(n_low=n_low,n_high=k)
    indices = np.argmax(logits, axis=1)
    T_hard = np.zeros((n_low,k))
    T_hard[np.arange(n_low), indices] = 1.

    dag_mask = None
    return T_hard, dag_mask

def hail_mary(A_low, k, logits_init=None, num_epochs=None, **kwds):
    A_low_tensor = torch.from_numpy(A_low).to(torch.float32)
    n_low = A_low.shape[0]
    model = HailMary(n_low=A_low_tensor.shape[0], n_high=k, A_low=A_low_tensor, logits_init=logits_init)
    if num_epochs is None:
        num_epochs = 1000 * int(np.log(n_low*k)+1)
    model.train(num_epochs=num_epochs, verbose=False)

    T_hard = model.get_partition()[0].cpu().detach().numpy()
    dag_mask = model.dag_mask()[0].cpu().detach().numpy()

    return T_hard, dag_mask

def hail_mary_abstract(A_low, k, logits_init=None, **kwds):
    A_low_tensor = torch.from_numpy(A_low).to(torch.float32)
    n_low = A_low.shape[0]
    model = HailMary(n_low=A_low_tensor.shape[0], n_high=k, A_low=A_low_tensor, logits_init=logits_init)
    num_epochs = 1000 * int(np.log(n_low*k)+1)
    model.train(num_epochs=num_epochs, compute_abs=True, compute_graph=False, compute_topo=False, compute_intv=False, verbose=False)

    T_hard = model.get_partition()[0].cpu().detach().numpy()
    dag_mask = None

    return T_hard, dag_mask

def hail_mary_graph(A_low, k, logits_init=None,**kwds):
    A_low_tensor = torch.from_numpy(A_low).to(torch.float32)
    n_low = A_low.shape[0]
    model = HailMary(n_low=A_low_tensor.shape[0], n_high=k, A_low=A_low_tensor, logits_init=logits_init)
    num_epochs = 1000 * int(np.log(n_low*k)+1)
    model.train(num_epochs=num_epochs, compute_abs=False, compute_graph=True, compute_topo=False, compute_intv=False, verbose=False)

    T_hard = model.get_partition()[0].cpu().detach().numpy()
    dag_mask = None
    
    return T_hard, dag_mask

def hail_mary_intv(A_low, k, logits_init=None, **kwds):
    A_low_tensor = torch.from_numpy(A_low).to(torch.float32)
    n_low = A_low.shape[0]
    model = HailMary(n_low=A_low_tensor.shape[0], n_high=k, A_low=A_low_tensor, logits_init=logits_init)
    num_epochs = 1000 * int(np.log(n_low*k)+1)
    model.train(num_epochs=num_epochs, compute_abs=False, compute_graph=False, compute_topo=False, compute_intv=True, verbose=False)

    T_hard = model.get_partition()[0].cpu().detach().numpy()
    dag_mask = None

    return T_hard, dag_mask

def hail_mary_intv_topo(A_low, k, logits_init=None, **kwds):
    A_low_tensor = torch.from_numpy(A_low).to(torch.float32)
    n_low = A_low.shape[0]
    model = HailMary(n_low=A_low_tensor.shape[0], n_high=k, A_low=A_low_tensor, logits_init=logits_init)
    num_epochs = 1000 * int(np.log(n_low*k)+1)
    model.train(num_epochs=num_epochs, compute_abs=False, compute_graph=False, compute_topo=True, compute_intv=True, verbose=False)

    T_hard = model.get_partition()[0].cpu().detach().numpy()
    dag_mask = model.dag_mask()[0].cpu().detach().numpy()

    return T_hard, dag_mask

def hail_mary_abstract_topo(A_low, k, logits_init=None, **kwds):
    A_low_tensor = torch.from_numpy(A_low).to(torch.float32)
    n_low = A_low.shape[0]
    model = HailMary(n_low=A_low_tensor.shape[0], n_high=k, A_low=A_low_tensor, logits_init=logits_init)
    num_epochs = 1000 * int(np.log(n_low*k)+1)
    model.train(num_epochs=num_epochs, compute_abs=True, compute_graph=False, compute_topo=True, compute_intv=False, verbose=False)

    T_hard = model.get_partition()[0].cpu().detach().numpy()
    dag_mask = model.dag_mask()[0].cpu().detach().numpy()

    return T_hard, dag_mask

def hail_mary_graph_topo(A_low, k, logits_init=None, **kwds):
    A_low_tensor = torch.from_numpy(A_low).to(torch.float32)
    n_low = A_low.shape[0]
    model = HailMary(n_low=A_low_tensor.shape[0], n_high=k, A_low=A_low_tensor, logits_init=logits_init)
    num_epochs = 1000 * int(np.log(n_low*k)+1)
    model.train(num_epochs=num_epochs, compute_abs=False, compute_graph=True, compute_topo=True, compute_intv=False, verbose=False)

    T_hard = model.get_partition()[0].cpu().detach().numpy()
    dag_mask = model.dag_mask()[0].cpu().detach().numpy()

    return T_hard, dag_mask

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
    S_dict, T_dict, A_high_dict = hierarchical_clustering(A_low=A_low, n_high=k, init_assignment=init_assignment, loss_list=[heuristic_loss]) 
    T_hard = T_dict[k]
    dag_mask = None
    return T_hard, dag_mask

def hc_binary(A_low, k, init_assignment=None, **kwds):
    A_low_binary = (A_low != 0) * 1.
    S_dict, T_dict, A_high_dict = hierarchical_clustering(A_low=A_low, n_high=k, init_assignment=init_assignment, loss_list=[heuristic_loss], A_low_binary=A_low_binary) 
    T_hard = T_dict[k]
    dag_mask = None
    return T_hard, dag_mask


def hc_abs(A_low, k, init_assignment=None, **kwds):
    S_dict, T_dict, A_high_dict = hierarchical_clustering(A_low=A_low, n_high=k, init_assignment=init_assignment, loss_list=[abstraction_residual, heuristic_loss]) 
    T_hard = T_dict[k]
    dag_mask = None
    return T_hard, dag_mask
