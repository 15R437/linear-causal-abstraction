"""Do the IIA=1 abstractions of the MI-identifiability XOR MLP satisfy (AC') for every intervention set Theta?

Setup (see THEORY.md). The low level is the pretrained 2-3-3-1 LeakyReLU MLP. Their output layer is LeakyReLU too, not linear; the input varies over {0,1}^2 and the biases are fixed.
Each mapping (from export_mappings.py) assigns every high-level node a cluster of neurons; T is supported on those clusters.
The high-level model has the formula's graph and the same LeakyReLU on every non-input node (or its reflection
z -> -phi(-z), which absorbs a negative sign in T). S is block-structured like T, so it only supplies a constant per node
(the biases are fixed).

Row j of (AC') reads  t_j . phi(z_{C_j}) = phi_j(w_j . (T^T X)_{pa(j)} + s_j). Since phi_j is invertible this holds iff
phi_j^{-1}(t_j . X_{C_j}) is an affine function of the parents' values (T^T X)_{pa(j)}. Here W~ and S are free, so they are
just the least-squares coefficients. For a singleton cluster, phi_j^{-1}(t_j X_i) = t_j z_i, so the test needs no choice of T at all:
is neuron i's pre-activation affine in its high-level parents' neurons? A multi-neuron cluster's direction t is found by
grid search. (AC') holds over a family of interventions iff every row's relative least-squares residual is ~0 on the points
of that family where the row is not intervened on.

Intervention families (each value is set on a whole cluster):
  obs     Theta = {} (the 4 inputs)
  single  |Theta| = 1, interchange values (the cluster's activations on one of the 4 inputs): the family IIA uses
  all     every Theta, every combination of interchange values
  dense   every non-empty Theta, random real values
"""
import itertools
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-6
rng = np.random.default_rng(0)

w = json.load(open(os.path.join(HERE, "data", "xor_mlp_weights.json")))
Ws, bs, alpha = [np.array(m) for m in w["W"]], [np.array(v) for v in w["b"]], w["negative_slope"]
L = len(Ws)
INPUTS = np.array(list(itertools.product([0.0, 1.0], repeat=2)))


def phi(z):
    return np.where(z >= 0, z, alpha * z)


def phi_inv(x):
    return np.where(x >= 0, x, x / alpha)


def forward(x, interventions=None):
    """Activations X[l] and pre-activations Z[l] per layer; `interventions` maps (layer, idx) -> value."""
    interventions = interventions or {}
    X, Z = [np.array(x, dtype=float)], [np.array(x, dtype=float)]
    for l in range(L):
        z = Ws[l] @ X[-1] + bs[l]
        a = phi(z)
        for (layer, idx), val in interventions.items():
            if layer == l + 1:
                a[idx] = val
        Z.append(z)
        X.append(a)
    return X, Z


def configs(record, family, n_dense=50):
    """Yield (Theta, X, Z) for every intervention in the family, Theta a set of node ids."""
    nodes = {n["id"]: n for n in record["nodes"]}
    internal = [i for i, n in nodes.items() if n["parents"] and i != record["root"]]
    clean = [forward(x)[0] for x in INPUTS]
    if family == "obs":
        sizes = [0]
    elif family == "single":
        sizes = [1]
    else:
        sizes = range(0 if family == "all" else 1, len(internal) + 1)
    for k in sizes:
        for theta in itertools.combinations(internal, k):
            if family == "dense":
                draws = []
                for _ in range(n_dense):
                    draws.append({tuple(nn): rng.normal(0, 2) for i in theta for nn in nodes[i]["neurons"]})
            else:
                draws = [{tuple(nn): clean[s][nn[0]][nn[1]] for i, s in zip(theta, srcs) for nn in nodes[i]["neurons"]}
                         for srcs in itertools.product(range(4), repeat=k)]
            for x in INPUTS:
                for iv in draws:
                    X, Z = forward(x, iv)
                    yield set(theta), X, Z


def rel_residual(target, regressors):
    A = np.column_stack(regressors + [np.ones(len(target))])
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)
    res = target - A @ coef
    scale = target.std()
    return np.sqrt(np.mean(res ** 2)) / (scale if scale > 1e-9 else 1.0)


def check(record, family):
    """Max relative (AC') residual over rows, optimised over the direction of any multi-neuron cluster."""
    nodes = {n["id"]: n for n in record["nodes"]}
    rows = [i for i, n in nodes.items() if n["parents"]]
    pts = list(configs(record, family))
    val = lambda X, i: np.array([X[l][k] for l, k in nodes[i]["neurons"]])  # cluster activations
    multi = [i for i in rows if len(nodes[i]["neurons"]) > 1]
    assert len(multi) <= 1, "grid search handles one multi-neuron cluster"
    angles = np.linspace(0, np.pi * 2, 721)[:-1] if multi else [None]

    best = None
    for ang in angles:
        t = {i: np.ones(len(nodes[i]["neurons"])) for i in nodes}
        if multi:
            t[multi[0]] = np.array([np.cos(ang), np.sin(ang)])
        out = {}
        for j in rows:
            active = [(X, Z) for th, X, Z in pts if j not in th]
            pars = list(dict.fromkeys(nodes[j]["parents"]))  # repeated leaves are one regressor
            pars = list({tuple(map(tuple, nodes[p]["neurons"])): p for p in pars}.values())
            regs = [np.array([t[p] @ val(X, p) for X, _ in active]) for p in pars]
            if len(nodes[j]["neurons"]) == 1:
                (l, k), = nodes[j]["neurons"]
                targets = [np.array([Z[l][k] for _, Z in active])]  # = phi^{-1}(X_i); the sign of t_j is irrelevant
            else:
                s = np.array([t[j] @ val(X, j) for X, _ in active])
                targets = [phi_inv(s), -phi_inv(-s)]  # high-level activation phi or its reflection
            out[j] = min(rel_residual(tg, regs) for tg in targets)
        if best is None or max(out.values()) < max(best.values()):
            best = out
    return best, len(pts)


if __name__ == "__main__":
    records = json.load(open(os.path.join(HERE, "data", "xor_mappings.json")))
    families = ["obs", "single", "all", "dense"]
    print("Max relative (AC') residual over rows; 0 means satisfied (tol %.0e).\n" % TOL)
    print(f"{'#':>3} {'formula':<34}" + "".join(f"{f:>10}" for f in families) + "   worst row (all)")
    summary = {f: 0 for f in families}
    for n, rec in enumerate(records):
        res = {f: check(rec, f) for f in families}
        for f in families:
            summary[f] += max(res[f][0].values()) < TOL
        rows_all = res["all"][0]
        worst = max(rows_all, key=rows_all.get)
        lab = {nd["id"]: nd["label"] for nd in rec["nodes"]}[worst]
        print(f"{n:>3} {rec['formula']:<34}" + "".join(f"{max(res[f][0].values()):>10.2e}" for f in families)
              + f"   {lab} -> {[tuple(x) for x in {nd['id']: nd['neurons'] for nd in rec['nodes']}[worst]]}")
    print("\nMappings satisfying (AC') exactly, per family:",
          ", ".join(f"{f}: {summary[f]}/{len(records)}" for f in families))
