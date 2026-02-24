import numpy as np
import copy
import random
import sys
import os
sys.path.insert(0, os.path.abspath(os.getcwd()))

from utils.generate_data import random_dag
from utils.plots import visualize_causal_graph
from utils.losses import find_lost_weights

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

    clusters = {i:[] for i in set(assignment_map.values())}
    for node,clst in assignment_map.items():
        if clst != -1: #ignore the bin cluster
            clusters[clst].append(node)

    if not is_acyclic_fast(A_high):
        log(f"Failed because A_high is Cyclic",verbose=verbose)
        return False

    for clst1,nodes1 in clusters.items():
        for clst2,nodes2 in clusters.items():
            if A_high[clst1, clst2] != 0:
                for i in nodes1:
                    if np.all(A_low[i, nodes2] == 0):
                        log(f"Failed because A_high[{clst1},{clst2}] is non-zero yet node {i} has no child in cluster {clst2}.",verbose=verbose)
                        return False
                    

    for i,clst1 in assignment_map.items():
        for j,clst2 in assignment_map.items():
            if clst1 == clst2:
                continue
            elif clst1 == -1 or clst2 == -1:
                continue
            elif A_low[i,j] != 0 and A_high[clst1, clst2] == 0:
                log(f"Failed because A_low[{i},{j}] is non-zero yet A_high[{clst1},{clst2}] is zero.",verbose=verbose)
                return False
    
    return True

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

def find_valid_alignment(A,k,init_clusters=None,seed=None,verbose=False):
    #this version orders the weights in A in descending order and tries to preserve the strongest weights first.
    rng = np.random.default_rng(seed)
    n_low = A.shape[0]
    A_high = np.zeros((k, k))
    if init_clusters is None:
        init_clusters = {}
    else:
        for i,sublist1 in init_clusters.items():
            for j,sublist2 in init_clusters.items():
                if i != j:
                    if np.any(A[np.ix_(sublist1, sublist2)] != 0):
                        A_high[i, j] = 1
    
    #print(f"A_high \n {A_high}")
    clusters = init_clusters | {i: [] for i in range(-1,k) if i not in list(init_clusters.keys())}
    assignment_map = {val: i for i, sublist in init_clusters.items() for val in sublist}
    assigned = [val for sublist in init_clusters.values() for val in sublist]
    
    assert is_valid_alignment(A, A_high, assignment_map), is_valid_alignment(A, A_high, assignment_map, verbose=True)
    
    # sort weights in descending order of magnitude!
    weight_list = []
    for i in range(n_low):
        for j in range(n_low):
            weight_list.append((i,j,abs(A[i,j])))

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
                
                
                valid_clusters = find_valid_clusters(A,A_high,node,assignment_map)
                
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
                    clusters = {i: [] for i in range(-1, k)}
                    for n_idx, c_idx in assignment_map.items():
                        if n_idx < A.shape[0]:
                            clusters[c_idx].append(n_idx)
                    
                    msg = f"Assigning node X{node+1} to cluster Y{assignment_map[node]+1} from valid options {['Y'+str(c[1][node]+1) for c in valid_clusters]}"
                    msg += f" to preserve edge X{i+1} -> X{j+1}" if len(valid_clusters_subset) < len(valid_clusters) else ""
                    log(msg,verbose)
                
                assigned = list(assignment_map.keys())

                    
        if len(assigned) == n_low:
            break
    
    return assignment_map,A_high

def hierarcical_clustering(A_low,n_high):
    n_low = A_low.shape[0]
    A_high = (A_low != 0).astype(int)
    clusters = {-1:[]} | {i:[i] for i in range(n_low)}
    for k in range(n_low,n_high,-1):
        best_A_high = None
        best_clusters = None
        best_loss = np.inf
        for i in range(k):
            for j in range(i):
                #merge node i into node j (note j < i)
                temp_clusters = {clst:nodes for clst,nodes in clusters.items() if clst < i} | {clst-1 :nodes for clst,nodes in clusters.items() if clst > i}
                temp_clusters[j] = clusters[j]+clusters[i]
                temp_A_high = A_high.copy()
                temp_A_high[j,:] = np.logical_or(A_high[i,:], A_high[j,:]).astype(int)
                temp_A_high[:,j] = np.logical_or(A_high[:,i], A_high[:,j]).astype(int)
                temp_A_high[j,j] = 0
                temp_A_high = np.delete(temp_A_high, i, axis=0)
                temp_A_high = np.delete(temp_A_high, i, axis=1)
                temp_assignment_map = {val: i for i, sublist in temp_clusters.items() for val in sublist}
                
                if is_valid_alignment(A_low,temp_A_high,temp_assignment_map):
                    lost_weights = find_lost_weights(A_low,temp_clusters)
                    loss = np.sum(np.abs(lost_weights)).item()
                    if loss < best_loss:
                        best_A_high = temp_A_high.copy()
                        best_clusters = copy.deepcopy(temp_clusters)
                        best_loss = loss
    
                #marginalise each cluster in turn
                for clst1,clst2 in zip((i,j),(j,i)):
                    temp_clusters = {clst:nodes for clst,nodes in clusters.items() if clst < clst1} | {clst-1:nodes for clst,nodes in clusters.items() if clst > clst1}
                    temp_clusters[-1] = clusters[-1] + clusters[clst1]
                    temp_clusters[clst2 - (clst2 > clst1)] = clusters[clst2]
                    temp_A_high = A_high.copy()
                    temp_A_high = np.delete(temp_A_high, clst1, axis=0)
                    temp_A_high = np.delete(temp_A_high, clst1, axis=1)
                    temp_assignment_map = {val: clst for clst, sublist in temp_clusters.items() for val in sublist}

                    if is_valid_alignment(A_low,temp_A_high,temp_assignment_map):
                        lost_weights = find_lost_weights(A_low,temp_clusters)
                        loss = np.sum(np.abs(lost_weights)).item()
                        if loss < best_loss:
                            best_A_high = temp_A_high.copy()
                            best_clusters = copy.deepcopy(temp_clusters)
                            best_loss = loss

        if best_A_high is not None:
            clusters = copy.deepcopy(best_clusters)
            A_high = copy.deepcopy(best_A_high)
        else:
            print(f"Stoping at k={k+1} as no valid pairs were found.")
            break        
    

    return clusters,A_high


    
if __name__ == "__main__":
    seed = None
    n_low = 10
    n_high = 5
    edge_density = 0.2
    A_low = random_dag(n_low, edge_density=edge_density, weight_mean=0, weight_std=1, seed=seed)

    random.seed(seed)
    random_sample = random.sample(range(n_low),k=n_high)
    init_clusters = {} #{i:[random_sample[i]] for i in range(n_high)}

    #assignment,A_high = find_valid_alignment(A_low,n_high, init_clusters, seed=seed)
    clusters,A_high = hierarcical_clustering(A_low,n_high)
    assignment = {val: i for i, sublist in clusters.items() for val in sublist}
    T_pred = np.zeros((n_low, n_high))
    for node, clst in assignment.items():
        if clst != -1:
            T_pred[node,clst] = 1
    

    print("Validtion of alignment:", is_valid_alignment(A_low, A_high, assignment))
    print(f"Clusters: \n {clusters}")
    print(f"T_pred:\n{T_pred}")

    visualize_causal_graph(A_low, A_high, T_pred, T_pred,title='Low-Level DAG with Valid Clusters')
                        


