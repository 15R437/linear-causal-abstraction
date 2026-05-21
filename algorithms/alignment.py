import numpy as np
import copy
import random
import itertools
import heapq
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))

from utils.generate_data import sample_low_level_graph
from utils.plots import visualize_causal_graph
from utils.losses import reconstruction_loss, abstraction_loss, heuristic_cost


def find_valid_clusters(A,A_high,node,assignment_map,seed=None):
    #subroutine to find a list of valid clusters for a given unassigned node
    assert is_valid_alignment(A, A_high, assignment_map,verbose=False), is_valid_alignment(A, A_high, assignment_map, verbose=True)
    assert isinstance(node, int) and 0 <= node < A.shape[0], f"node must be an integer index within the bounds [0,{A.shape[0]})"
    assert node not in list(assignment_map.keys()), "Node is already assigned to a cluster."
    
    assigned = list(assignment_map.keys())
    clusters = {i:[] for i in range(A_high.shape[0])} | {-1:[]}
    for n,c in assignment_map.items():
        clusters[c].append(n)

    valid_clusters = []
    rng = np.random.default_rng(seed)
    #budget = budget if budget is not None else A_high.shape[0]*A.shape[0]

    for clst in range(len(clusters)-1):
        temp_assignment_map = {k:v for k,v in assignment_map.items()} | {node:clst}
        temp_assigned = set(temp_assignment_map.keys())
        temp_clusters = {k: list(v) for k, v in clusters.items()}
        temp_clusters[clst].append(node)
        temp_A_high = A_high.copy()

        node_parents = np.where(A[:,node]!=0)[0]
        node_children = np.where(A[node,:]!=0)[0]
        for p in node_parents:
            if p in assigned:
                temp_A_high[assignment_map[p],clst] = 1 if clst != assignment_map[p] else 0
        for c in node_children:
            if c in assigned:
                temp_A_high[clst,assignment_map[c]] = 1 if clst != assignment_map[c] else 0

        valid = False
        if is_valid_alignment(A,temp_A_high,temp_assignment_map):
            valid = True
        else:
            temp_A_high = A_high.copy()
            abandon_clst = True

            #1. Check that `node` has a child in each *current* child cluster of `clst` (as determined by A_high) or else its unassigned children can be assigned clusters to make this true.
            child_clusters = np.where(A_high[clst] != 0)[0]
            problematic_clusters = [int(child_clst) for child_clst in child_clusters if np.all(A[node, temp_clusters[child_clst]] == 0)] # < k
            unassigned_children = [int(c) for c in np.where(A[node] != 0)[0] if c not in temp_assigned] # < n_low
            
            if len(unassigned_children) > len(problematic_clusters) > 0:
                for c in unassigned_children:
                    p0 = problematic_clusters[0]
                    temp_assignment_map[c] = p0
                    unassigned_children_sample = rng.choice([s for s in unassigned_children if s != c], size=len(problematic_clusters)-1, replace=False).tolist() if len(problematic_clusters) > 1 else []
                    for i in range(1, len(problematic_clusters)):
                        temp_assignment_map[unassigned_children_sample[i-1]] = problematic_clusters[i]
                    
                    if is_valid_alignment(A, A_high,temp_assignment_map): #note A_high here is not a typo. We are just checking that this rule #1 is satisfied (so we only consider the existing children of clst)
                        abandon_clst = False
                        for p,n in zip(problematic_clusters,[c]+unassigned_children_sample):
                            temp_clusters[p] = temp_clusters[p] + [n]
                            if max(temp_clusters[p]) >= A.shape[0] or n >= A.shape[0]:
                                import pdb; pdb.set_trace()
                        temp_assigned = set(temp_assignment_map.keys())
                        break
                    else:
                        for key in set([c]+unassigned_children_sample):
                            temp_assignment_map.pop(key,None)

            elif len(problematic_clusters) ==0:
                abandon_clst = False

            if abandon_clst:
                continue  
            
            #2. Check that each cluster (child_clst != clst) that contains a child of `node` has an edge from `clst` or else each member of `clst` has an unassigned child that can be assigned to child_clst to make this true.
            child_clusters = list(set([temp_assignment_map[c] for c in np.where(A[node]!=0)[0] if c in temp_assigned]))
            child_clusters = [C for C in child_clusters if C != clst]
            for child_clst in child_clusters:
                abandon_clst = True
                problematic_nodes = [member for member in temp_clusters[clst] if np.all(A[member,temp_clusters[child_clst]] == 0)] # a node in `clst` is problematic if it has no child in child_clst
                unassigned_children = [[c for c in np.where(A[member] != 0)[0] if c not in temp_assigned] for member in problematic_nodes] # a list of unassigned children for each problematic node
                
                if not problematic_nodes:
                    abandon_clst = False
                    continue
                if all(unassigned_children):
                    for c in unassigned_children[0]:
                        temp_assignment_map[c] = child_clst
                        unassigned_children_sample = [rng.choice(lst) for lst in unassigned_children[1:]] if len(unassigned_children) > 1 else []
                        for rand_c in unassigned_children_sample:
                            temp_assignment_map[rand_c] = child_clst

                        temp_A_high[clst, child_clst] = 1 if clst != child_clst else 0
                        
                        if is_valid_alignment(A, temp_A_high, temp_assignment_map):
                            abandon_clst = False
                            temp_clusters[child_clst].extend([c]+unassigned_children_sample)
                            temp_assigned = list(temp_assignment_map.keys())
                            break
                        else:
                            for key in set([c] + unassigned_children_sample):
                                temp_assignment_map.pop(key,None)           

                if abandon_clst:
                    break

            if abandon_clst:
                continue  

            #3. Check that each cluster (parent_clst != clst) that contains a parent of `node` has an edge to `clst` or else each member of parent_clst has an unassigned child that can be assigend to `clst` to make this true.
            parent_clusters = list(set([assignment_map[c] for c in np.where(A[:,node]!=0)[0] if c in temp_assigned]))
            parent_clusters = [C for C in parent_clusters if C != clst]
            for parent_clst in parent_clusters:
                abandon_clst = True
                problematic_nodes = [member for member in temp_clusters[parent_clst] if np.all(A[member,temp_clusters[clst]]==0)]  # a node in parent_clst is problematic if it has no child in clst
                unassigned_children = [[c for c in np.where(A[member] != 0)[0] if c not in temp_assigned] for member in problematic_nodes]
                if not problematic_nodes:
                    abandon_clst = False
                    continue
                if all(unassigned_children):
                    for c in unassigned_children[0]:
                        temp_assignment_map[c] = clst
                        unassigned_children_sample = [rng.choice(lst) for lst in unassigned_children[1:]] if len(unassigned_children) > 1 else []
                        for rand_c in unassigned_children_sample:
                            temp_assignment_map[rand_c] = clst
                        
                        temp_A_high[parent_clst,clst] = 1 if clst != parent_clst else 0
                        if is_valid_alignment(A,temp_A_high,temp_assignment_map):
                            abandon_clst = False
                            temp_clusters[clst].extend([c]+unassigned_children_sample)
                            temp_assigned = list(temp_assignment_map.keys())
                            break
                        else:
                            for key in set([c] + unassigned_children_sample):
                                temp_assignment_map.pop(key,None)
                    
                if abandon_clst:
                    break
            
            if not abandon_clst:
                valid = True
                
        
        if is_valid_alignment(A_low=A,A_high=temp_A_high,assignment_map=temp_assignment_map):
            valid_clusters.append((temp_clusters,temp_assignment_map,temp_A_high))

    return valid_clusters

def find_valid_alignment(A_low,n_high,init_clusters=None,seed=None,verbose=False):
    #this version orders the weights in A in descending order and tries to preserve the strongest weights first.
    rng = np.random.default_rng(seed)
    n_low = A_low.shape[0]
    A_high = np.zeros((n_high, n_high))
    if init_clusters is None:
        init_clusters = {}
    else:
        for i,sublist1 in init_clusters.items():
            for j,sublist2 in init_clusters.items():
                if i != j:
                    if np.any(A_low[np.ix_(sublist1, sublist2)] != 0):
                        A_high[i, j] = 1
    
    #print(f"A_high \n {A_high}")
    clusters = init_clusters | {i: [] for i in range(-1,n_high) if i not in list(init_clusters.keys())}
    assignment_map = {val: i for i, sublist in init_clusters.items() for val in sublist}
    assigned = [val for sublist in init_clusters.values() for val in sublist]
    
    assert is_valid_alignment(A_low, A_high, assignment_map), is_valid_alignment(A_low, A_high, assignment_map, verbose=True)
    
    # sort weights in descending order of magnitude!
    weight_list = []
    for i in range(n_low):
        for j in range(n_low):
            weight_list.append((i,j,abs(A_low[i,j])))

    weight_list.sort(key=lambda x: x[2], reverse=True) 

    ignored_weights = [] #tracks the number of low-level weights that we are not able to preserve in the high-level alignment
    for i,j,w in weight_list:
        if i in assigned and j in assigned:
            continue
        else:
            nodes = [i,j]
            for idx,node in enumerate(nodes):
                if node in assigned:
                    continue
                
                
                valid_clusters = find_valid_clusters(A_low,A_high,node,assignment_map)
                
                if not valid_clusters:
                    log(f"No valid cluster found for node X{node+1}. Adding to bin...",verbose)
                    assignment_map[node] = -1
                    clusters[-1].append(node)
                    ignored_weights.append((i,j,w))
                    
                else:
                    if len(valid_clusters) > 1 and nodes[(idx-1)%2] in assigned:
                        valid_clusters_subset = [(temp_clusters,temp_assignment_map,temp_A_high) for temp_clusters,temp_assignment_map,temp_A_high in valid_clusters if temp_assignment_map[node]!= temp_assignment_map[nodes[idx-1]]] #preserves weight i -> j
                    else:
                        valid_clusters_subset = valid_clusters

                    _,assignment_map,A_high = random.choice(valid_clusters_subset)
                    clusters = {i: [] for i in range(-1, n_high)}
                    for n_idx, c_idx in assignment_map.items():
                        if n_idx < A_low.shape[0]:
                            clusters[c_idx].append(n_idx)
                    
                    msg = f"Assigning node X{node+1} to cluster Y{assignment_map[node]+1} from valid options {['Y'+str(c[1][node]+1) for c in valid_clusters]}"
                    msg += f" to preserve edge X{i+1} -> X{j+1}" if len(valid_clusters_subset) < len(valid_clusters) else ""
                    log(msg,verbose)
                
                assigned = list(assignment_map.keys())

                    
        if len(assigned) == n_low:
            break
    
    return assignment_map,A_high

def is_acyclic_fast(A, threshold=0):
    #Khan's Algorithm with the assumption that A[i,j] represents i -> j
    adj = np.abs(A) > threshold
    n = adj.shape[0]
    in_degree = adj.sum(axis=0) 
    queue = [i for i in range(n) if in_degree[i] == 0]
    count = 0
    
    while queue:
        u = queue.pop(0)
        count += 1
        for v in range(n):
            if adj[u, v]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
    
    return count == n

def log(msg,verbose):
    if verbose:
        print(msg)

def is_valid_alignment(A_low,A_high,assignment_map,verbose=False):
    assert (isinstance(A_low, np.ndarray) and A_low.ndim == 2 and A_low.shape[0] == A_low.shape[1]), "A_low must be a 2D square numpy array"
    assert (isinstance(A_high, np.ndarray) and A_high.ndim == 2 and A_high.shape[0] == A_high.shape[1]), "A_high must be a 2D square numpy array"
    #assert isinstance(assignment_map, dict) and all(isinstance(k, int) and isinstance(v, int) for k, v in assignment_map.items()), "assignment map must be a dictionary with integer keys and values"

    unique_clusters = sorted([c for c in set(assignment_map.values()) if c != -1])
    n_high = len(unique_clusters)
    n_low = A_low.shape[0]
    assert A_high.shape[0] == n_high


    if not is_acyclic_fast(A_high):
        log(f"Failed because A_high is Cyclic",verbose=verbose)
        return False

    # check that A_high[clst1,clst2] != 0 ==> every node1 in clst1 has a child node2 in clst2
    clusters_list = [[node for node,clst in assignment_map.items() if clst == c] for c in range(len(unique_clusters))]
    for clst1,nodes1 in enumerate(clusters_list):
        for clst2 in np.nonzero(A_high[clst1,:])[0]:
            nodes2 = clusters_list[clst2]
            if np.any(np.all(A_low[np.ix_(nodes1,nodes2)]==0, axis=1)):
                log(f"Failed because A_high[{clst1},{clst2}] is non-zero yet there exists a node in {clst1} with no child in cluster {clst2}.",verbose=verbose)
                return False

    # check that A_low[i,j] != 0  ==> A_high[assignment_map[i],assignment_map[j]] != 0  for assignment_map[i] != assignment_map[j]           
    assignment_array = np.array([assignment_map.get(i,-1) for i in range(n_low)])
    rows, cols = np.nonzero(A_low)
    filter = (assignment_array[rows] != -1) & (assignment_array[cols] != -1) & (assignment_array[rows] != assignment_array[cols])
    rows, cols = rows[filter], cols[filter]

    if np.any(A_high[assignment_array[rows],assignment_array[cols]] == 0):
        log(f"Failed because A_low has a non-zero edge not reflected in A_high.",verbose=verbose)
        return False
    
    return True

def is_valid_local(A_low,A_high,assignment_array,clusters_list,active_clusters,i,j,verbose=False):

    if not is_acyclic_fast(A_high):
        log(f"Failed because A_high is Cyclic",verbose=verbose)
        return False
    
    if j != -1:
        # we are in the merging regime
        active_clusters = [idx for idx in sorted(active_clusters) if idx != i]
        new_clst_idx = active_clusters.index(j)
        new_clst_nodes = clusters_list[i] + clusters_list[j]
        for other_clst_idx in np.nonzero(A_high[new_clst_idx,:])[0]:
            other_clst_nodes = clusters_list[active_clusters[other_clst_idx]]
            if np.any(np.all(A_low[np.ix_(new_clst_nodes,other_clst_nodes)]==0, axis=1)):
                    log(f"Failed because A_high[{new_clst_idx},{other_clst_idx}] is non-zero yet there exists a node in {new_clst_idx} with no child in cluster {other_clst_idx}.",verbose=verbose)
                    return False
    
        old2new = {orig_clst_idx: curr_clst_idx for curr_clst_idx, orig_clst_idx in enumerate(active_clusters)}   
        rows, cols = np.nonzero(A_low)
        old_rows, old_cols = assignment_array[rows], assignment_array[cols]
        old_rows[old_rows == i] = j
        old_cols[old_cols == i] = j
        filter = (old_rows != -1) & (old_cols != -1) & (old_rows !=old_cols)
        old_rows, old_cols = old_rows[filter], old_cols[filter]
        

        new_rows, new_cols = np.array([old2new[r] for r in old_rows],dtype=int), np.array([old2new[c] for c in old_cols],dtype=int)
        
        if np.any(A_high[new_rows,new_cols] == 0):
            log(f"Failed because A_low has a non-zero edge not reflected in A_high.",verbose=verbose)
            return False
        
    return True
    

def update_A_high(active_clusters,A_high,i,j,mode='merge'):
    """
    If mode = 'merge' then i and j should be different.
    If syle = 'marginalise' then cluster i is marginalised.
    """
    #j,i = min(i,j),max(i,j)
    active_clusters = sorted(active_clusters)
    pos_i = active_clusters.index(i)
    mask = np.ones(A_high.shape[0],dtype=bool)
    mask[pos_i] = False
    new_A_high = A_high[mask][:,mask]

    if mode=='merge':
        pos_j = active_clusters.index(j)
        merged_row = A_high[pos_i,mask] | A_high[pos_j,mask]
        merged_col = A_high[mask,pos_i] | A_high[mask,pos_j]
        
        pos_j = pos_j if pos_j < pos_i else pos_j - 1
        new_A_high[pos_j,:] = merged_row
        new_A_high[:,pos_j] = merged_col
        new_A_high[pos_j,pos_j] = 0
    
    return new_A_high

def build_map(active_clusters,clusters_list,marginalised_list,i,j,mode='merge'):
    """
    If mode = 'merge' then i and j should be different. i is merged into j.
    If syle = 'marginalise' then cluster i is marginalised.
    """

    assignment_map = {node: -1 for node in marginalised_list}
    active_clusters = sorted([idx for idx in active_clusters if idx!= i])

    for new_clst,orig_clst in enumerate(active_clusters):
        for node in clusters_list[orig_clst]:
            assignment_map[node] = new_clst
        
        if mode == 'merge' and orig_clst == j:
            for node in clusters_list[i]:
                assignment_map[node] = new_clst

    if mode == 'marginalise':
        for node in clusters_list[i]:
            assignment_map[i] = -1
    

    return assignment_map

def hierarchical_clustering_heap(A_low,n_high):
    n_low = A_low.shape[0]
    A_high = (A_low != 0).astype(int)
    clusters_list = [[i] for i in range(n_low)]
    assignment_array = np.arange(n_low)
    marginalised = []

    is_active_mask = np.ones(n_low,dtype=bool) #tracks active nodes
    active_clusters = set(range(n_low))
    cluster_version = {i:0 for i in range(n_low)}
    heap = []

    def push_merge_candidate(idx):
        for other in active_clusters:
            if other == idx: continue
            loss = np.sum(np.abs(A_low[np.ix_(clusters_list[idx], clusters_list[other])])) + np.sum(np.abs(A_low[np.ix_(clusters_list[other], clusters_list[idx])]))
            heapq.heappush(heap, (loss, idx, other, 'merge', cluster_version[idx], cluster_version[other]))

    def push_marginalise_candidate(idx):
        outgoing = np.sum(np.abs(A_low[clusters_list[idx]][:,is_active_mask]))
        incoming = np.sum(np.abs(A_low[is_active_mask][:,clusters_list[idx]]))
        internal = np.sum(np.abs(A_low[np.ix_(clusters_list[idx],clusters_list[idx])]))
        loss = outgoing + incoming - internal
        heapq.heappush(heap, (loss, idx, -1, 'marginalise', cluster_version[idx], 0))


    for i in range(n_low):
        push_marginalise_candidate(i)
        for j in range(i):
            loss = np.sum(np.abs(A_low[i, j])) + np.sum(np.abs(A_low[j, i]))
            heapq.heappush(heap, (loss, i, j, 'merge', 0, 0))
    
    k = n_low
    clusters_dict = {n_low:{i:[] for i in range(-1,n_low)}}
    A_high_dict = {n_low:A_high}
    while k > n_high and heap:
        loss, i, j, mode, v_i, v_j = heapq.heappop(heap)
        if i not in active_clusters or (mode == 'merge' and j not in active_clusters):
            continue
        if v_i != cluster_version[i] or (j != -1 and v_j != cluster_version[j]):
            continue

        temp_A_high = update_A_high(active_clusters,A_high,i,j,mode=mode)
        if mode == 'marginalise':
            A_high = temp_A_high
            is_active_mask[clusters_list[i]] = False
            marginalised.extend(clusters_list[i])
            assignment_array[clusters_list[i]] = -1
            active_clusters.remove(i)
            cluster_version[i] += 1
            k -= 1

            clusters_dict[k] = {-1:marginalised} | {idx:clusters_list[c] for idx,c in enumerate(sorted(active_clusters))}
            A_high_dict[k] = A_high

        elif mode == 'merge':
            if is_valid_local(A_low, temp_A_high, assignment_array, clusters_list, active_clusters, i, j):
                A_high = temp_A_high 
                clusters_list[j].extend(clusters_list[i])
                assignment_array[clusters_list[i]] = j
                active_clusters.remove(i)
                cluster_version[j] += 1
                push_merge_candidate(j)
                push_marginalise_candidate(j)
                k -= 1

                clusters_dict[k] = {-1:marginalised} | {idx:clusters_list[c] for idx,c in enumerate(sorted(active_clusters))}
                A_high_dict[k] = A_high

    return clusters_dict, A_high_dict

def enforce_dag(A):
    """
    Greedily removes edges with the smallest absolute weights that participate in cycles.
    This guarantees an acyclic graph while roughly minimizing reconstruction error.
    """
    n = A.shape[0]
    A_dag = np.zeros_like(A)
    
    # Extract all non-zero edges and sort by magnitude descending
    edges = []
    for i in range(n):
        for j in range(n):
            if A[i, j] != 0:
                edges.append((abs(A[i, j]), A[i, j], i, j))
    
    edges.sort(key=lambda x: x[0], reverse=True)
    
    # DFS to check if adding an edge u->v creates a cycle (i.e. path v->u already exists)
    def has_path(adj, u, v):
        stack = [u]
        visited = set()
        while stack:
            curr = stack.pop()
            if curr == v:
                return True
            if curr not in visited:
                visited.add(curr)
                for nxt in range(n):
                    if adj[curr, nxt] != 0:
                        stack.append(nxt)
        return False

    for _, w, i, j in edges:
        if not has_path(A_dag, j, i):
            A_dag[i, j] = w
            
    return A_dag

def get_A_high(T,A_low=None,E_low=None,priority='recon'):
    n_low,n_high = T.shape
    if priority == 'recon':
        with np.errstate(divide='ignore', invalid='ignore'):
            T_norm = T / np.sqrt(np.sum(T**2, axis=0))
        
        T_norm = np.nan_to_num(T_norm)
        A_high = T_norm.T @ A_low @ T
    elif priority == 'abstraction':
        partition_mask = (T != 0).astype(int)
        I_low = np.eye(n_low)
        I_high = np.eye(n_high)
        
        F_pi = partition_mask.T[...,None] * I_low
        inv_term = np.linalg.inv(I_low + F_pi @ (E_low - I_low))
        M = (E_low - I_low) @ inv_term @ F_pi

        lhs = M @ E_low[None,...] @ T[None,...]
        rhs_base = (E_low - I_low) @ T # Shape: (n_low, n_high)
        rhs = np.tile(rhs_base[None, ...], (n_high, 1, 1))
        dot_prod = np.sum(lhs * rhs, axis=1)
        a_norm = np.sum(lhs * lhs, axis=1)
        b_norm = np.sum(rhs * rhs, axis=1)
        
        E_high_ij = dot_prod/(b_norm + 1e-12) # (n_high,n_high)
        E_high = (E_high_ij * (1 - np.eye(n_high))) + np.eye(n_high)
        A_high = I_high - np.linalg.inv(E_high)
    
    return enforce_dag(A_high)

def get_A_hat(A_high,T):
    with np.errstate(divide='ignore', invalid='ignore'):
        T_norm = T / np.sqrt(np.sum(T**2, axis=0))
    
    T_norm = np.nan_to_num(T_norm)
    return T @ A_high @ T_norm.T


def get_S(E_low, A_high, T):
    E_high_inv = np.eye(len(A_high)) - A_high
    
    partition_mask = (T != 0).astype(int)
    S = (E_low @ T @ E_high_inv) * partition_mask
    return S

def get_T(assignment_array):
    n_low = len(assignment_array)
    unique, col_indices = np.unique(assignment_array, return_inverse=True)
    n_high = len(unique)
    
    T = np.zeros((n_low, n_high))
    T[np.arange(n_low), col_indices] = 1.0
    
    return T

def heuristic_cost(A_low, c1_nodes, c2_nodes):
    out_diff = np.mean(A_low[c1_nodes, :], axis=0) - np.mean(A_low[c2_nodes, :], axis=0)
    in_diff  = np.mean(A_low[:, c1_nodes], axis=1) - np.mean(A_low[:, c2_nodes], axis=1)

    return np.sum(out_diff**2) + np.sum(in_diff**2)

def merge_A_high(A_high, A_low, T, i, j):
    """Update A_high incrementally after merging clusters i and j into i."""
    n_low, n_high = T.shape
    # New combined column (unnormalised)
    t_new = T[:, i] + T[:, j]  # T[:, j] is about to be removed
    t_new_norm = t_new / (np.linalg.norm(t_new) + 1e-8)
    
    # Only update row i and column i — O(k * n_low)
    A_high[i, :] = t_new_norm @ A_low @ T          # new row
    A_high[:, i] = (T / np.linalg.norm(T, axis=0, keepdims=True)).T @ A_low @ t_new  # new col
    A_high[i, i] = 0.0
    
    # Remove row/col j
    mask = np.ones(n_high, dtype=bool)
    mask[j] = False
    return A_high[np.ix_(mask, mask)]

def hierarchical_clustering(A_low, n_high=1,init_assignment=None,loss_list=[heuristic_cost]):
    n_low = A_low.shape[0]
    E_low = np.linalg.inv(np.eye(len(A_low)) - A_low)
    clusters_list = [[i] for i in range(n_low)]
    assignment_array = np.arange(n_low)

    if init_assignment is None:
        init_assignment = np.ones(n_low) * -1
    
    active_clusters = set(range(n_low))
    cluster_version = {i:0 for i in range(n_low)}
    heap = []

    def push_merge_candidate(idx):
        for other in active_clusters:
            if other == idx: continue
            u, v = min(idx, other), max(idx, other)
            
            c1_nodes = clusters_list[u]
            c2_nodes = clusters_list[v]
            
            # Check consistency with initialization
            assigned1 = init_assignment[c1_nodes]
            assigned2 = init_assignment[c2_nodes]
            assigned = np.concatenate([assigned1[assigned1 != -1], assigned2[assigned2 != -1]])
            is_consistent = len(np.unique(assigned)) <= 1
            
            #losses = [1-int(is_consistent)] + [loss_fn(A_low=A_low,E_low=E_low) for loss_fn in loss_list]
            # FAST HEURISTIC: L2 difference in causal connectivity profiles
            out_diff = np.mean(A_low[c1_nodes, :], axis=0) - np.mean(A_low[c2_nodes, :], axis=0)
            in_diff  = np.mean(A_low[:, c1_nodes], axis=1) - np.mean(A_low[:, c2_nodes], axis=1)
            heuristic_cost = np.sum(out_diff**2) + np.sum(in_diff**2)
            
            # Push into heap using heuristic cost instead of full matrix inversions
            heapq.heappush(heap, (
                1 - int(is_consistent), 
                heuristic_cost, 
                u, 
                v, 
                cluster_version[u], 
                cluster_version[v]
            ))

    for i in range(n_low):
        push_merge_candidate(i)

    S_dict = {}
    T_dict = {}
    A_high_dict = {}
    k = n_low
    while k > n_high and heap:
        *losses, i, j, v_i, v_j = heapq.heappop(heap)
        if i not in active_clusters or j not in active_clusters:
            continue
        if v_i != cluster_version[i] or  v_j != cluster_version[j]:
            continue
        
        # commit 
        clusters_list[i].extend(clusters_list[j])
        assignment_array[clusters_list[j]] = i
        active_clusters.remove(j)
        cluster_version[i] += 1
        push_merge_candidate(i)
        k -= 1

        T = get_T(assignment_array)
        A_high = get_A_high(A_low=A_low,T=T)
        S = get_S(E_low=E_low,A_high=A_high,T=T)
        S_dict[k] = S
        T_dict[k] = T
        A_high_dict[k] = A_high
    
    return S_dict, T_dict, A_high_dict



    
if __name__ == "__main__":
    seed = None #23 sometimes gives an example of zero loss but graphical violations
    n_low = 100
    n_high = 5
    marg_prob = 0.
    T_lower = (-2,1)
    T_upper = (-1,2)
    edge_density = 0.4

    A_low,T_gt,S_gt,A_high_gt = sample_low_level_graph(n_high=n_high,n_low=n_low,density=edge_density,T_lower=T_lower,T_upper=T_upper,marg_prob=marg_prob,seed=seed,shuffle=True,internal=False)
    E_low = np.linalg.inv(np.eye(n_low) - A_low)

    #assignment,A_high = find_valid_alignment(A_low,n_high, init_clusters, seed=seed)
    start_time = time.time()
    S_dict, T_dict, A_high_dict = hierarchical_clustering(A_low)
    runtime = time.time() - start_time
    S_pred = S_dict[n_high]
    T_pred = T_dict[n_high]
    A_high = A_high_dict[n_high]
    A_hat = get_A_hat(A_high,T_pred)

    
    print(f"Runtime: \n{runtime:.4f}s")
    print(f"Reconstruction Loss: \n {reconstruction_loss(A_low,A_hat):4f}")
    print(f"Abstraction Loss: \n {abstraction_loss(E_low,A_low,A_high,S_pred,T_pred):4f}")

    visualize_causal_graph(A_low, A_high, T_pred, S_pred,title='Low-Level DAG with Valid Clusters')
                        


