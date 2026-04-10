import numpy as np
import networkx as nx

def find_A_high(A_low,A_high_binary,T):
    G = nx.from_numpy_array(A_high_binary, create_using=nx.DiGraph)
    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        print(f"Error: The high-level graph contains a cycle.")
    
    low_level_order = np.concatenate([np.nonzero(T[:,j])[0] for j in topo_order])
    #how to deal with marginalised variables with exogenous influence?