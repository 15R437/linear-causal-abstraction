import numpy as np
import copy
import warnings
import sys
import os
import itertools
sys.path.insert(0, os.path.abspath(os.getcwd()))

from causal_experiments.generate_data import random_dag
from causal_experiments.plots import visualize_causal_graph


def old_find_valid_alignment(A,init_clusters:dict[int, list[int]],seed=None):
    """
    Find a valid clustering for abstraction.
    Params:
        A (np.ndarray): Adjacency matrix of the low-level DAG (A[i,j] is effect of i on j).
        init_clusters (dict[int, list[int]]): A dictionary where keys are cluster indices and values are lists of node indices in each cluster.
    Returns:
        dict[int, list[int]]: A dictionary where keys are cluster indices and values are lists of node indices in each cluster.
    """

    rng = np.random.default_rng(seed)
    n_low = A.shape[0]
    k = len(init_clusters)
    A_high = np.zeros((k, k))
    for i,sublist1 in init_clusters.items():
        for j,sublist2 in init_clusters.items():
            if i != j:
                if np.any(A[np.ix_(sublist1, sublist2)] != 0):
                    A_high[i, j] = 1  # Mark an edge in the high-level DAG

    clusters = copy.deepcopy(init_clusters)
    assignment_map = {val: i for i, sublist in init_clusters.items() for val in sublist}
    assigned = [val for sublist in init_clusters.values() for val in sublist]
    unassigned = list(set(range(n_low)) - set(assigned))
    assigned_copy = copy.deepcopy(assigned)

    #maybe try preserving the strongest weights first? or the ones with highest out degree? or highest in degree? or some combination of these?
    REPEAT = True
    while REPEAT:
        REPEAT = False
        while assigned_copy:
            parent_node = assigned_copy.pop(0)
            parent_clst = assignment_map[parent_node]
            children = np.where(A[parent_node] != 0)[0]
            for current_node in children:
                if current_node in assigned:
                    current_clst = assignment_map[current_node]
                    if current_clst != parent_clst:
                        # all parents of current_clst are assigned parent_clst
                        for c in clusters[current_clst]:
                            other_parents = np.where(A[:, c] != 0)[0]
                            for parent in other_parents:
                                if parent in unassigned:
                                    clusters[parent_clst].append(parent)
                                    assignment_map[parent] = parent_clst
                                    assigned.append(parent)
                                    assigned_copy.append(parent)
                                    unassigned.remove(parent)
                                    A_high[parent_clst, current_clst] = 1
                                    print(A_high)
                                    REPEAT = True
                else:
                    parent_clusters = [assignment_map[p] for p in np.where(A[:, current_node] != 0)[0] if p in assigned]
                    VALID_CLST = False
                    #import pdb;pdb.set_trace()
                    for clst in range(k):
                        #check that for every child of cluster clst, this current node is a parent of it and for every parent of current node they already have clst as a child.
                        #suppose we assign current node to clst.

                        for parent_clst in parent_clusters:
                            if A_high[parent_clst, clst] != 0:
                                VALID_CLST = True
                                clusters[clst].append(current_node)
                                assignment_map[current_node] = clst
                                assigned.append(current_node)
                                assigned_copy.append(current_node)
                                A_high[parent_clst, clst] = 1
                                unassigned.remove(current_node)
                                break
                        if VALID_CLST:
                            REPEAT = True
                            break
                        

        if unassigned and not REPEAT:
            # Randomly assign one unassigned node to a random cluster
            print("Unassigned nodes remain, randomly assigning one to a cluster.")
            REPEAT = True
            node = rng.choice(unassigned)
            cluster_idx = rng.integers(0, k)
            clusters[cluster_idx].append(node)
            assignment_map[node] = cluster_idx
            assigned.append(node)
            unassigned.remove(node)

            parents = np.where(A[:, node] != 0)[0]
            children = np.where(A[node] != 0)[0]
            for parent in parents:
                if parent in assigned:
                    parent_clst = assignment_map[parent]
                    A_high[parent_clst, cluster_idx] = 1
                    print(A_high)
            for child in children:
                if child in assigned:
                    child_clst = assignment_map[child]
                    A_high[cluster_idx, child_clst] = 1
                    print(A_high)
    
    return clusters

def find_valid_alignment_v1(A,init_clusters,seed=None):
    rng = np.random.default_rng(seed)
    n_low = A.shape[0]
    k = len(init_clusters)
    A_high = np.zeros((k, k))
    for i,sublist1 in init_clusters.items():
        for j,sublist2 in init_clusters.items():
            if i != j:
                if np.any(A[np.ix_(sublist1, sublist2)] != 0):
                    A_high[i, j] = 1

    clusters = copy.deepcopy(init_clusters)
    assignment_map = {val: i for i, sublist in init_clusters.items() for val in sublist}
    assigned = [val for sublist in init_clusters.values() for val in sublist]
    unassigned = set(range(n_low)) - set(assigned)
    unassigned_list = list(unassigned)
    for node in unassigned_list:
        if node in assigned:
            continue
        print(A_high)
        valid_clusters = []
        # assign it to the first available cluster that doesn't violate the rule
        for clst in range(k):
            valid = True
            child_clusters = np.where(A_high[clst] != 0)[0]
            for child_clst in child_clusters:
                #check that node has a child in every child cluster
                child_nodes = clusters[child_clst]
                #should we allow for currently unassigned child nodes? so if node has a child that is not yet assigned to a cluster
                if np.all(A[node, child_nodes] == 0):
                    valid = False
                    break
            if not valid:
                continue
            #check that for every child of node that is in a cluster, that cluster has an edge from clst
            for child_node in np.where(A[node] != 0)[0]:
                if child_node in assigned:
                    child_clst = assignment_map[child_node]
                    if A_high[clst, child_clst] == 0:
                        valid = False
                        break
            if not valid:
                continue

            #check that each parent of node that is assigned to a cluster has an edge to clst
            for parent_node in np.where(A[:, node] != 0)[0]:
                if parent_node in assigned:
                    parent_clst = assignment_map[parent_node]
                    if parent_clst != clst and A_high[parent_clst, clst] == 0:
                        valid = False
                        break
            if not valid:
                continue
            valid_clusters.append(clst)

        if valid_clusters:
            chosen_clst = rng.choice(valid_clusters)
            print(f"Assigning node X{node+1} to cluster Y{chosen_clst+1} from valid options {['Y'+str(c+1) for c in valid_clusters]}")
            clusters[chosen_clst].append(node)
            assignment_map[node] = chosen_clst
            assigned.append(node)
            unassigned.remove(node)
            for child_node in np.where(A[node] != 0)[0]:
                if child_node in assigned:
                    child_clst = assignment_map[child_node]
                    A_high[chosen_clst, child_clst] = 1 if chosen_clst != child_clst else 0
            for parent_node in np.where(A[:, node] != 0)[0]:
                if parent_node in assigned:
                    parent_clst = assignment_map[parent_node]
                    A_high[parent_clst, chosen_clst] = 1 if chosen_clst != parent_clst else 0

        else:
            print(f"No valid cluster found for node X{node+1}, leaving it unassigned.")

    return clusters,A_high

def find_valid_alignment_v2(A,init_clusters,seed=None):
    rng = np.random.default_rng(seed)
    n_low = A.shape[0]
    k = len(init_clusters)
    A_high = np.zeros((k, k))
    for i,sublist1 in init_clusters.items():
        for j,sublist2 in init_clusters.items():
            if i != j:
                if np.any(A[np.ix_(sublist1, sublist2)] != 0):
                    A_high[i, j] = 1

    clusters = copy.deepcopy(init_clusters)
    assignment_map = {val: i for i, sublist in init_clusters.items() for val in sublist}
    assigned = [val for sublist in init_clusters.values() for val in sublist]
    unassigned = set(range(n_low)) - set(assigned)
    unassigned_list = list(unassigned)
    for node in unassigned_list:
        print(A_high)
        if node in assigned:
            continue
        child_nodes = np.where(A[node] != 0)[0]
        parent_nodes = np.where(A[:, node] != 0)[0]
        valid_clusters = []
        for clst in range(k):
            #temporarily assign node to cluster clst and check if it violates the rule using is_valid_alignment function
            temp_clusters = copy.deepcopy(clusters)
            temp_clusters[clst].append(node)
            temp_A_high = copy.deepcopy(A_high)
            for child_node in child_nodes:
                if child_node in assigned:
                    child_clst = assignment_map[child_node]
                    temp_A_high[clst, child_clst] = 1 if clst != child_clst else 0
            for parent_node in parent_nodes:
                if parent_node in assigned:
                    parent_clst = assignment_map[parent_node]
                    temp_A_high[parent_clst, clst] = 1 if clst != parent_clst else 0
            if is_valid_alignment(A, temp_A_high, temp_clusters):
                valid_clusters.append(clst)
        if valid_clusters:
            chosen_clst = rng.choice(valid_clusters)
            print(f"Assigning node X{node+1} to cluster Y{chosen_clst+1} from valid options {['Y'+str(c+1) for c in valid_clusters]}")
            clusters[chosen_clst].append(node)
            assignment_map[node] = chosen_clst
            assigned.append(node)
            unassigned.remove(node)
            for child_node in np.where(A[node] != 0)[0]:
                if child_node in assigned:
                    child_clst = assignment_map[child_node]
                    A_high[chosen_clst, child_clst] = 1 if chosen_clst != child_clst else 0
            for parent_node in np.where(A[:, node] != 0)[0]:
                if parent_node in assigned:
                    parent_clst = assignment_map[parent_node]
                    A_high[parent_clst, chosen_clst] = 1 if chosen_clst != parent_clst else 0
        else:
            print(f"No valid cluster found for node X{node+1}, leaving it unassigned.")

    return clusters,A_high

def find_valid_clusters(A,A_high,node,clusters,seed=None,budget=None):
    #subroutine to find a list of valid clusters for a given unassigned node
    assert is_valid_alignment(A, A_high, clusters), "Input clusters and A_high do not represent a valid alignment."
    assert isinstance(node, int) and 0 <= node < A.shape[0], f"node must be an integer index within the bounds [0,{A.shape[0]})"
    assert node not in [val for sublist in clusters.values() for val in sublist], "Node is already assigned to a cluster."
    
    assigned = [val for sublist in clusters.values() for val in sublist]
    assignment_map = {val: i for i, sublist in clusters.items() for val in sublist}
    valid_clusters = []
    rng = np.random.default_rng(seed)
    budget = budget if budget is not None else A_high.shape[0]*A.shape[0]

    for clst in range(len(clusters)):
        valid = True
        to_be_assigned = {}
        temp_assignment_map = copy.deepcopy(assignment_map)
        temp_A_high = copy.deepcopy(A_high)

        #1. CHECK THAT `node` HAS A CHILD IN EACH CHILD CLUSTER OF `clst`.
        #Q: what if `node` has a child that is not yet assigned to a cluster? should we allow for this and just make sure that for each child cluster of `clst`, `node` has at least one child that is either already assigned to that child cluster or is currently unassigned (and could later be assigned to that child cluster)?
        child_clusters = np.where(A_high[clst] != 0)[0]
        problematic_clusters = [child_clst for child_clst in child_clusters if np.all(A[node, clusters[child_clst]] == 0)] # < k
        unassigned_children = [c for c in np.where(A[node] != 0)[0] if c not in assigned] # < n_low
        if len(problematic_clusters) > len(unassigned_children):
            valid = False
            continue
        else:
            #for each child cluster that doesn't have an edge from node, a lits of unassigned children of node, one of whom should be assigned to that child cluster
            valid = False
            temp_clusters = copy.deepcopy(clusters)
           
            for c in unassigned_children:
                p1 = problematic_clusters[0]
                temp_assignment_map[c] = p1
                unassigned_children_sample = rng.choice([s for s in unassigned_children if s != c], size=len(problematic_clusters)-1, replace=False) if len(problematic_clusters) > 1 else []
                for i in range(1, len(problematic_clusters)):
                    temp_assignment_map[unassigned_children_sample[i-1]] = problematic_clusters[i]

                if is_valid_alignment(A, A_high,temp_assignment_map):
                    valid = True
                    break

        if not valid:
            continue  

        #2. CHECK THAT EACH CHILD OF `node` IS EITHER i) UNASSIGNED; ii) ASSIGNED TO A CLUSTER (`child_clst`) WITH AN EXISTING EDGE FROM `clst` OR ELSE iii) EACH MEMBER OF `clst` HAS A CHILD THAT IS CURRENTLY UNASSIGNED (and could later be assigned to `child_clst`) 
        for child_node in np.where(A[node] != 0)[0]:
            if child_node in assigned:
                child_clst = assignment_map[child_node]
                temp_assignment_map[node] = clst
                temp_A_high = copy.deepcopy(A_high)
                temp_A_high[clst, child_clst] = 1 if clst != child_clst else 0
                if not is_valid_alignment(A, temp_A_high, temp_assignment_map):
                    problematic_nodes = [member for member in clusters[clst] if np.all(A[member, clusters[child_clst]] == 0)]
                    unassigned_children = [[c for c in np.where(A[member] != 0)[0] if c not in assigned] for member in problematic_nodes ]
                    if not all(unassigned_children):
                        valid = False
                        break
                    if np.any([np.all([member_child in assigned and member_child not in clusters[child_clst] for member_child in np.where(A[member] != 0)[0]]) for member in clusters[clst]]):
                        #if some member of clst has all its children assigned to clusters other than child_clst...
                        valid = False
                        break
                    #for each member of clst that doesn't already have a child in child_clst, a list of unassigned children, one of whom should be assigned to child_clst
                    to_be_assigned = to_be_assigned | {member: ([c for c in np.where(A[member] != 0)[0] if c not in assigned],child_clst) for member in clusters[clst] if all([c not in clusters[child_clst] for c in np.where(A[member] != 0)[0]])} 
            
                    
        if not valid:
            continue

        #3. CHECK THAT EACH PARENT OF `node` THAT IS ASSIGNED TO A CLUSTER (`parent_clst`) HAS i) parent_clst = clst and the resulting clustering is valid; ii) AN EXISTING EDGE TO `clst` OR ELSE ii) EACH MEMBER OF `parent_clst` HAS A CHILD THAT IS CURRENTLY UNASSIGNED (and could later be assigned to `clst`) 
        for parent_node in np.where(A[:, node] != 0)[0]:
            if parent_node in assigned:
                parent_clst = assignment_map[parent_node]
                temp_clusters = copy.deepcopy(clusters)
                temp_assignment_map[node] = clst
                temp_A_high[parent_clst, clst] = 1 if parent_clst != clst else 0
                if not is_valid_alignment(A, temp_A_high, temp_assignment_map):
                    if np.any([np.all([member_child in assigned and member_child not in temp_clusters[clst] for member_child in np.where(A[member] != 0)[0]]) for member in clusters[parent_clst]]):
                        valid = False
                        break
                    #for each member of parent_clst that doesn't already have a child in clst, a list of unassigned children, one of whom should be assigned to clst
                    to_be_assigned = to_be_assigned | {member: ([c for c in np.where(A[member] != 0)[0] if c not in assigned], clst) for member in clusters[parent_clst] if all([c not in clusters[clst] for c in np.where(A[member] != 0)[0]])} 

        if not valid:
            continue
        
        #Final check for each combination of members of clst and their assigned children in child_clst, that if we were to assign node to clst and those children to child_clst, the resulting clustering would be valid.
        
        valid_clusters.append((clst,temp_assignment_map,temp_A_high))

    return valid_clusters

def update_A_high(A,A_high,node,clst,assignment_map):
    assigned = set(assignment_map.keys())
    for parent_clst in [assignment_map[p] for p in np.where(A[:, node] != 0)[0] if p in assigned]:
        A_high[parent_clst,clst] = 1 if parent_clst != clst else 0
    for child_clst in [assignment_map[c] for c in np.where(A[node] != 0)[0] if c in assigned]:
        A_high[clst, child_clst] = 1 if clst != child_clst else 0
    
    return A_high

def find_valid_alignment(A,init_clusters=None,seed=None):
    #this version orders the weights in A in descending order and tries to preserve the strongest weights first.
    rng = np.random.default_rng(seed)
    n_low = A.shape[0]
    k = len(init_clusters)
    A_high = np.zeros((k, k))
    if init_clusters is None:
        init_clusters = {i:[] for i in range(k)}
    for i,sublist1 in init_clusters.items():
        for j,sublist2 in init_clusters.items():
            if i != j:
                if np.any(A[np.ix_(sublist1, sublist2)] != 0):
                    A_high[i, j] = 1
    
    assert is_valid_alignment(A, A_high, init_clusters), "Initial clusters and A_high do not represent a valid alignment."


    clusters = copy.deepcopy(init_clusters)
    assignment_map = {val: i for i, sublist in init_clusters.items() for val in sublist}
    assigned = [val for sublist in init_clusters.values() for val in sublist]
    weight_list = []
    for i in range(n_low):
        for j in range(n_low):
            weight_list.append((i,j,abs(A[i,j])))

    weight_list.sort(key=lambda x: x[2], reverse=True)  # Sort by weight magnitude in descending order

    ignored_weights = [] #tracks the number of low-level weights that we are not able to preserve in the high-level alignment
    for i,j,w in weight_list:
        if i in assigned and j in assigned:
            continue
        else:
            nodes = [i,j]
            for idx,node in enumerate(nodes):
                if node in assigned:
                    continue
                try:
                    valid_clusters = find_valid_clusters(A,A_high,node,clusters)
                except Exception as e:
                    print(f"Error finding valid clusters for node X{node+1}: {e}")
                    import pdb;pdb.set_trace()
                    
                if not valid_clusters:
                    raise ValueError(f"Sadly no valid cluster found for node X{node+1}....")
                    
                else:
                    if nodes[(idx-1)%2] not in assigned:
                        pass
                    elif len(valid_clusters) > 1:
                        valid_clusters = [(clst,to_be_assigned) for clst,to_be_assigned in valid_clusters if clst!= assignment_map[nodes[idx-1]]]
                    elif w > 0:
                        ignored_weights.append((i,j,w))

                    clst,to_be_assigned = rng.choice(valid_clusters)
                    print(f"Assigning node X{node+1} to cluster Y{clst+1} from valid options {['Y'+str(c[0]+1) for c in valid_clusters]}")
                    clusters[clst].append(node)
                    assignment_map[node] = clst
                    assigned.append(node)
                    A_high = update_A_high(A,A_high,node,clst,assignment_map)

                    if to_be_assigned:
                        for options,clst in to_be_assigned.values():
                            options = rng.permutation(options) #shuffle
                            for option in options:
                                if option not in assigned:
                                    print(f"Additionally assigning node X{option+1} to cluster Y{clst+1} to preserve alignment validity.") 
                                    clusters[clst].append(option)
                                    assignment_map[option] = clst
                                    assigned.append(option)
                                    A_high = update_A_high(A,A_high,option,clst,assignment_map)
                                    break
                    print(A_high)
        if len(assigned) == n_low:
            break
    
    return clusters,A_high,ignored_weights
                

def is_valid_alignment(A_low,A_high,clusters,assignment_map=None):
    assert (isinstance(A_low, np.ndarray) and A_low.ndim == 2 and A_low.shape[0] == A_low.shape[1]), "A_low must be a 2D square numpy array"
    assert (isinstance(A_high, np.ndarray) and A_high.ndim == 2 and A_high.shape[0] == A_high.shape[1]), "A_high must be a 2D square numpy array"
    assert isinstance(clusters, dict) and all(isinstance(k, int) and isinstance(v, list) for k, v in clusters.items()), "clusters must be a dictionary with integer keys and list values"
    assert A_high.shape == (len(clusters),len(clusters)), f"Expected A_high shape {(len(clusters),len(clusters))}, got {A_high.shape}."

    assignment_map = {node: clst for clst, sublist in clusters.items() for node in sublist}
    #check that A_high is acyclic!

    for clst1,nodes1 in clusters.items():
        for clst2,nodes2 in clusters.items():
            if A_high[clst1, clst2] != 0:
                for i in nodes1:
                    if np.all(A_low[i, nodes2] == 0):
                        return False
                    

    for i,clst1 in assignment_map.items():
        for j,clst2 in assignment_map.items():
            if clst1 == clst2:
                continue
            elif A_low[i,j] != 0 and A_high[clst1, clst2] == 0:
                return False
    
    return True
    
if __name__ == "__main__":
    seed = None
    n_low = 8
    n_high = 3
    edge_density = 0.3
    A_low = random_dag(n_low, edge_density=edge_density, weight_mean=0, weight_std=1, seed=None)
    #print(f"A_low:\n{A_low}")
    init_clusters = {i:[i] for i in range(n_high)} 

    #quick test
    A_low = np.array([[0,0,1,1,0],
                      [0,0,0,1,0],
                      [0,0,0,0,0],
                      [0,0,0,0,1],
                      [0,0,0,0,0]])
    init_clusters = {0:[0,1], 1:[3], 2:[4]}
    print(f"A_low:\n{A_low}")
    print(f"init_clusters: {init_clusters}")
    valid_clusters,A_high,ignored_weights = find_valid_alignment(A_low, init_clusters, seed=42)
    T_pred = np.zeros((n_low, len(valid_clusters)))
    for cluster_idx, nodes in valid_clusters.items():
        for node in nodes:
            T_pred[node, cluster_idx] = 1.0

    print("Validtion of alignment:", is_valid_alignment(A_low, A_high, valid_clusters))
    print(f"T_pred:\n{T_pred}")
    print(f"Ignored weights: {['X'+str(i+1) + ' -> X' + str(j+1) for i,j,w in ignored_weights]}")

    visualize_causal_graph(A_low, A_high, T_pred, T_pred,title='Low-Level DAG with Valid Clusters')
                        


