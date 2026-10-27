"""Heaviside abstractions tau_j = H(t_j . x_{C_j} + beta_j) of the XOR MLP with free threshold gates, two ways.

The high-level model is a threshold network: node j = H(sum_i w_ij h_i + gamma_j), with h_i = x_i on inputs. H is not
injective, so row j of the abstraction condition only asks for the SIGNS to agree:

    sign(t_j . x_{C_j} + beta_j) = sign(sum_i w_ij h_i + gamma_j)    at every admissible x with j not intervened on.

Non-triviality: tau_root must equal XOR on the 4 clean inputs, and every hidden tau_j must be non-constant on them
(otherwise all-constant tau's satisfy every row).

Methods
  LP   Layer-wise hard-margin search. In topological order, with the parents' bits fixed, try every labelling of the distinct
       parent patterns. A labelling is kept if it is a threshold gate (an LP with margin 1 on the parent patterns) and the
       cluster readout realises it (an LP with margin 1 on x_{C_j}). The margin fixes the scale, so t = 0 is excluded.
       Children branch on the parents' choices (DFS). Exact and exhaustive: counts the distinct labellings that work.
  ANN  Joint annealed fit. sigmoid(k L_j) is matched to sigmoid(k R_j), with L, R the two sides divided by |t_j| and
       |w_j| (signed distances to the hyperplanes). The temperature k rises from 1 to 200, so the loss tends to sign
       agreement; the parents enter through sigmoid(k L_i). Penalties enforce the non-triviality above. Scored by hard signs.

Usage: python step_tau.py [family]   (family: obs, single, all; default all)
"""
import itertools
import json
import os
import sys

import numpy as np
import torch
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sigmoid_tau import INPUTS, Problem, xor_net  # noqa: E402

XOR = np.array([a != b for a, b in INPUTS])


def separator(F, y):
    """(v, b) with y_n (v . f_n + b) >= 1 for every n (y in {False, True}), or None if not linearly separable."""
    n, d = F.shape
    A = -(2.0 * y - 1)[:, None] * np.column_stack([F, np.ones(n)])
    r = linprog(np.zeros(d + 1), A_ub=A, b_ub=-np.ones(n), bounds=[(None, None)] * (d + 1), method="highs")
    return r.x if r.status == 0 else None


def clean_values(P, net):
    clean = [net.forward(x) for x in INPUTS]
    return {j: np.array([[X[l][k] for l, k in P.C[j]] for X in clean]) for j in P.order}


def lp_search(P, net, root, cap=1000):
    cv = clean_values(P, net)
    bits = lambda p, V, sep: V[:, 0] > 0.5 if p in P.src else V @ sep[p][:-1] + sep[p][-1] > 0
    sols = []

    def dfs(k, sep):
        if len(sols) >= cap:
            return
        if k == len(P.order):
            sols.append(sep)
            return
        j = P.order[k]
        Xj, Vp, _ = P.rows[j]
        H = np.column_stack([bits(p, Vp[p], sep) for p in P.pa[j]]).astype(float)
        pats, inv = np.unique(H, axis=0, return_inverse=True)
        for lab in itertools.product([False, True], repeat=len(pats)):
            lab = np.array(lab)
            if lab.all() or not lab.any() or separator(pats, lab) is None:
                continue  # constant, or not a threshold gate
            F, y = Xj, lab[inv.ravel()]
            if j == root:
                F, y = np.vstack([F, cv[j]]), np.concatenate([y, XOR])
            s = separator(F, y)
            if s is None:
                continue
            c = cv[j] @ s[:-1] + s[-1] > 0
            if c.all() or not c.any():
                continue
            dfs(k + 1, {**sep, j: s})

    dfs(0, {})
    return sols


def annealed(P, net, root, starts=4, steps=1500, seed=0):
    gen = torch.Generator().manual_seed(seed)
    T = torch.as_tensor
    rows = {j: (T(Xj), {p: T(V) for p, V in Vp.items()}, T(w / w.sum())) for j, (Xj, Vp, w) in P.rows.items()}
    cv = {j: T(v) for j, v in clean_values(P, net).items()}
    xor = T(XOR.astype(float))
    best = None
    for _ in range(starts):
        t = {j: torch.randn(len(P.C[j]), generator=gen).requires_grad_() for j in P.order}
        beta = {j: torch.randn((), generator=gen).requires_grad_() for j in P.order}
        wt = {j: torch.randn(len(P.pa[j]), generator=gen).requires_grad_() for j in P.order}
        gam = {j: torch.zeros((), requires_grad=True) for j in P.order}
        opt = torch.optim.Adam([*t.values(), *beta.values(), *wt.values(), *gam.values()], lr=0.03)
        L = lambda j, V: (V @ t[j] + beta[j]) / t[j].norm()
        for step in range(steps):
            k = 200.0 ** (step / (steps - 1))
            loss = torch.zeros(())
            for j in P.order:
                Xj, Vp, w = rows[j]
                G = torch.stack([Vp[p][:, 0] if p in P.src else torch.sigmoid(k * L(p, Vp[p])) for p in P.pa[j]], 1)
                R = (G @ wt[j] + gam[j]) / wt[j].norm()
                loss = loss + (w * (torch.sigmoid(k * L(j, Xj)) - torch.sigmoid(k * R)) ** 2).sum()
                pc = torch.sigmoid(k * L(j, cv[j]))
                if j == root:
                    loss = loss + ((pc - xor) ** 2).mean()
                else:
                    loss = loss + (1 - (pc.max() - pc.min())) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():  # hard score: weighted fraction of sign disagreements per row
            err = {}
            for j in P.order:
                Xj, Vp, w = rows[j]
                G = torch.stack([Vp[p][:, 0] if p in P.src else (L(p, Vp[p]) > 0).double() for p in P.pa[j]], 1)
                err[j] = float((w * ((L(j, Xj) > 0) != (G @ wt[j] + gam[j] > 0)).double()).sum())
                c = L(j, cv[j]) > 0
                if j == root:
                    err[j] += float((c != xor.bool()).double().mean())
                elif c.all() or not c.any():
                    err[j] += 1.0
        if best is None or max(err.values()) < max(best.values()):
            best = err
    return best


if __name__ == "__main__":
    family = sys.argv[1] if len(sys.argv) > 1 else "all"
    net = xor_net()
    records = json.load(open(os.path.join(HERE, "data", "xor_mappings.json")))
    print(f"Heaviside tau, free threshold gates, family '{family}'.")
    print("LP = number of exact labellings found (0 = none exists); ANN = worst row's fraction of sign errors (0 = exact).\n")
    print(f"{'#':>3} {'formula':<34}{'LP':>6}{'ANN':>10}")
    for n, rec in enumerate(records):
        P = Problem(net, rec, family)
        n_lp = len(lp_search(P, net, rec["root"]))
        ann = max(annealed(P, net, rec["root"]).values())
        print(f"{n:>3} {rec['formula']:<34}{n_lp:>6}{ann:>10.3f}", flush=True)
