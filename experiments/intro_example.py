"""The README's headline example: can the partition in Figure 1 be learned?

Figure 1 shows six UK sectors from EXIOBASE grouped into Feed & ingredients, Livestock farming
and Meat processing. That grouping is stated in the paper, not derived there. This script asks
whether it is recoverable from the weight matrix alone.

It rebuilds the 6-node graph from the figure's edge weights, checks the grouping against the
sufficient conditions, and runs each partition learner at k=3.
"""

import numpy as np


from src.algos.greedy import aoc_closure, greedy_allin, oc_closure
from src.utils.partition import all_in_violation

NAMES = ["Fp", "Bv", "Ca", "Po", "Mc", "Mp"]

# a_ij is the direct effect of X_i on X_j, read off Figure 1.
EDGES = [("Fp", "Bv", 0.019), ("Bv", "Fp", 0.081),
         ("Fp", "Ca", 0.169), ("Fp", "Po", 0.075),
         ("Bv", "Ca", 0.074), ("Bv", "Po", 0.034),
         ("Ca", "Mc", 0.288), ("Po", "Mp", 0.047)]

TARGET = [[0, 1], [2, 3], [4, 5]]   # {Fp,Bv} {Ca,Po} {Mc,Mp}


def figure_one_matrix():
    idx = {n: i for i, n in enumerate(NAMES)}
    A = np.zeros((len(NAMES), len(NAMES)))
    for src, dst, w in EDGES:
        A[idx[src], idx[dst]] = w
    return A


def fmt(partition):
    clusters = sorted((sorted(c) for c in partition), key=min)
    return " | ".join("{" + ",".join(NAMES[i] for i in c) + "}" for c in clusters)


def same(partition, other):
    norm = lambda p: sorted(sorted(c) for c in p)
    return norm(partition) == norm(other)


def cluster_graph(A, partition):
    """(k, k) 0/1 matrix: an edge i->j iff some node in cluster i points into cluster j."""
    k = len(partition)
    B = A != 0
    CG = np.zeros((k, k), dtype=int)
    for i, ci in enumerate(partition):
        for j, cj in enumerate(partition):
            if i != j and B[np.ix_(ci, cj)].any():
                CG[i, j] = 1
    return CG


def main():
    A = figure_one_matrix()
    CG = cluster_graph(A, TARGET)

    print(f"target partition          {fmt(TARGET)}")
    print(f"  All-In violations       {all_in_violation(A, TARGET):.0f}")
    print(f"  max children per cluster{CG.sum(axis=1).max():>3}   (One-Child needs <= 1)")
    print(f"  cluster graph edges     {[(i, j) for i in range(len(TARGET)) for j in range(len(TARGET)) if CG[i, j]]}")

    print("\nlearned at k=3:")
    for name, learned in [("greedy_allin", greedy_allin(A, 3)),
                          ("aoc_closure", aoc_closure(A, 3, n_restarts=50, seed=0)),
                          ("oc_closure", oc_closure(A, 3, n_restarts=50, seed=0))]:
        part = [list(c) for c in learned]
        print(f"  {name:<14} {fmt(part):<34} {'matches' if same(part, TARGET) else 'differs'}")


if __name__ == "__main__":
    main()
