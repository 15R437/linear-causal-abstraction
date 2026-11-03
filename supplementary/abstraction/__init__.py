"""Linear causal abstraction: partition learners and abstraction fitters.

    closure               AOC Closure / OC Closure (one-child merge closure + All-In repair)
    partition_baselines   Greedy All-In and the structural-equivalence baselines
    eigen_solver          closed-form fitter: sinks optimised, cyclic SCCs pinned
    fitters               propagation and rank-one fitters, and the router
    generate_data         synthetic low-level models with a known abstraction
    losses                RAE, reconstruction error and structural diagnostics
"""
from .closure import occ, oc_closure, conditions, budget_coarsening, information_loss
from .partition_baselines import allin_agglom, greedy_merge, violations
from .eigen_solver import fit_abstraction_eigen
from .fitters import fit_abstraction, fit_abstraction_K, fit_abstraction_auto, get_S
from .losses import RAE, reconstruction_loss
from .generate_data import sample_low_level_graph, T_concretisation_sampling, generate_random_T

__all__ = ['occ', 'oc_closure', 'conditions', 'budget_coarsening', 'information_loss',
           'allin_agglom', 'greedy_merge', 'violations', 'fit_abstraction_eigen',
           'fit_abstraction', 'fit_abstraction_K', 'fit_abstraction_auto', 'get_S',
           'RAE', 'reconstruction_loss', 'sample_low_level_graph',
           'T_concretisation_sampling', 'generate_random_T']
