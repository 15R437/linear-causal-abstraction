"""How many of the 42 IIA=1 abstractions of the XOR MLP satisfy (AC') with Boolean high-level variables?

This mirrors the paper's setup. The high-level model is their Boolean formula, and each mapped formula node is a gate on its
mapped parents (unmapped negation children are merged into their parent). The abstraction map tau is:
  inputs   identity (they are already 0/1)
  output   1[y > 0.5]  (the paper's reading of the output)
  hidden   1[t . X_C > theta] on the node's cluster C. For a singleton, t = +-1 (either orientation); for a 2-neuron
           cluster, t is searched on a grid of directions. The thresholds theta are searched exhaustively: only the side of theta
           each reached value falls on matters, so every distinct cut of the reached values is tried.

(AC') with Boolean high-level variables: for every intervention (Theta, v) on clusters, tau(X^{Theta,v}) must equal the
high-level solution under do(Theta = tau(v)). Equivalently, at every non-intervened node j, tau_j(X) = gate_j(tau(X)_{pa(j)}).

Columns (number of tau's satisfying the condition; a mapping passes if there is at least one):
  iia     single-node interchange interventions, output only: the paper's IIA (sanity check, should be 42/42)
  single  single-node interchange interventions, every node
  all     every Theta, interchange values (each cluster copied from any of the 4 inputs), every node
  dense   `all` plus every non-empty Theta with random real values, every node
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ac_check import INPUTS, configs

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_SOLUTIONS = 10_000
N_ANGLES = 72


def gate(tree, mapped, j, H):
    """Value of formula node j as a function of the values H of its mapped descendants (vectorised over configs)."""
    def rec(n):
        if n != j and n in mapped:
            return H[n]
        op, ch = tree[n]["op"], tree[n]["children"]
        if op == "*":
            return rec(ch[0]) & rec(ch[1])
        if op == "+":
            return rec(ch[0]) | rec(ch[1])
        if op == "¬":
            return ~rec(ch[0])
        raise ValueError(op)
    return rec(j)


def truth(tree, j, x):
    def rec(n):
        op, ch = tree[n]["op"], tree[n]["children"]
        if op in ("A", "B"):
            return bool(x["AB".index(op)])
        if op == "*":
            return rec(ch[0]) and rec(ch[1])
        if op == "+":
            return rec(ch[0]) or rec(ch[1])
        return not rec(ch[0])
    return rec(j)


def count_solutions(rec, family):
    tree = {int(k): v for k, v in rec["tree"].items()}
    nodes = {n["id"]: n for n in rec["nodes"]}
    root = rec["root"]
    hidden = [i for i, n in nodes.items() if n["parents"] and i != root]
    leaves = [i for i, n in nodes.items() if not n["parents"]]

    if family in ("iia", "single"):
        pts = list(configs(rec, "obs")) + list(configs(rec, "single"))
    else:
        pts = list(configs(rec, "all")) + (list(configs(rec, "dense", n_dense=100)) if family == "dense" else [])
    n = len(pts)
    clean = np.array([len(th) == 0 for th, _, _ in pts])
    clean_x = [X[0] for (th, X, _), c in zip(pts, clean) if c]
    I = {j: np.array([j in th for th, _, _ in pts]) for j in hidden}
    val = {j: np.array([[X[l][k] for l, k in nodes[j]["neurons"]] for _, X, _ in pts]) for j in nodes}
    H_leaf = {i: val[i][:, 0] > 0.5 for i in leaves}
    y_bit = val[root][:, 0] > 0.5

    # candidate tau_j: boolean vectors over configs, consistent with the formula on the clean inputs, deduplicated
    cands = {}
    for j in hidden:
        V = val[j]
        dirs = [np.array([1.0]), np.array([-1.0])] if V.shape[1] == 1 else \
            [np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)]
        want = np.array([truth(tree, j, x) for x in clean_x])
        relevant = I[j] | clean if family == "iia" else np.ones(n, bool)
        seen = {}
        for t in dirs:
            s = V @ t
            u = np.unique(np.round(s, 12))
            for th in np.concatenate([[u[0] - 1], (u[:-1] + u[1:]) / 2, [u[-1] + 1]]):
                B = s > th
                if np.array_equal(B[clean], want):
                    seen.setdefault(B[relevant].tobytes(), B)
        cands[j] = list(seen.values())

    order = []  # topological: a node after its mapped hidden parents
    while len(order) < len(hidden):
        for j in hidden:
            if j not in order and all(p in leaves or p in order for p in nodes[j]["parents"]):
                order.append(j)

    sols = 0

    def search(k, H):
        nonlocal sols
        if sols >= MAX_SOLUTIONS:
            return
        if k == len(order):
            if np.array_equal(gate(tree, nodes, root, H), y_bit):
                sols += 1
            return
        j = order[k]
        g = gate(tree, nodes, j, H)
        for B in cands[j]:
            if family != "iia" and not np.array_equal(B[~I[j]], g[~I[j]]):
                continue  # (AC') row j fails at a non-intervened config
            search(k + 1, {**H, j: np.where(I[j], B, g)})

    search(0, dict(H_leaf))
    return sols, n


if __name__ == "__main__":
    records = json.load(open(os.path.join(HERE, "data", "xor_mappings.json")))
    families = ["iia", "single", "all", "dense"]
    print("Number of Boolean abstraction maps tau satisfying the condition (0 = the mapping fails).\n")
    print(f"{'#':>3} {'formula':<34}" + "".join(f"{f:>9}" for f in families))
    passed = {f: 0 for f in families}
    for i, rec in enumerate(records):
        res = {f: count_solutions(rec, f)[0] for f in families}
        for f in families:
            passed[f] += res[f] > 0
        print(f"{i:>3} {rec['formula']:<34}" + "".join(f"{res[f]:>9}" for f in families), flush=True)
    print("\nMappings with at least one valid tau:", ", ".join(f"{f}: {passed[f]}/{len(records)}" for f in families))
