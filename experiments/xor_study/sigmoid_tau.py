"""Learn sigmoid abstraction maps tau_j(x) = sigmoid(t_j . x_{C_j} + beta_j) for the XOR MLP's mappings, three ways.

Setup. Inputs are singleton high-level sources with tau = identity. Every other high-level node j has the abstraction above
and the high-level equation sigmoid(sum_i w_ij g_i + gamma_j), where g_i = x_i for an input parent and g_i = tau_i(x)
otherwise. Since sigmoid is injective, row j of the abstraction condition is linear in the node's own logit:

    t_j . x_{C_j} = sum_i w_ij g_i + c_j        at every admissible x with j not intervened on   (c_j = gamma_j - beta_j)

A row's residual is measured relative to the spread of its left-hand side: rel_j = RMS(residual) / std(t_j . x). This is the
unit-variance normalisation that rules out the trivial solution t = 0. Once the parents are fixed, rel_j^2 is a generalised
Rayleigh quotient in t_j. rel_j = 0 on every row means the abstraction is exact on that family.

Methods
  A1  Directions from the network. Differentiating row j in a parent cluster C_i gives J(x)^T t_j || t_i, where J is the
      Jacobian of x_{C_j} in x_{C_i} with all of j's parents held fixed. If the reached contexts excite every read unit, each
      row of J is parallel to t_i, so t_i is the top singular vector of the stacked gradients over all children and contexts.
      For adjacent layers these gradients are rescaled weight rows, so the direction needs no data. Scale and shift are then
      fitted as in A2. rank1 = s2/s1 of the stack (0 = rank one).
  A2  Layer-wise variable projection. With the parents fixed, t_j is the smallest generalised eigenvector of row j and
      (w_j, c_j) are least-squares coefficients. Each non-sink node keeps that direction and contributes a scale a_i and a shift
      b_i on its standardised projection. These are optimised jointly over the children's residuals (multi-start L-BFGS-B).
  A3  Joint gradient fit. All t, beta, w and c are optimised with L-BFGS on sum_j rel_j^2 from random starts. With
      --rank-penalty L it adds L times the A1 condition: the share of J(x)^T t_j orthogonal to t_i.

On singleton clusters the direction is trivial, so A1 = A2 there. Only multi-neuron clusters separate them: mappings 40 and
41, and the planted check.

Families (each intervention sets a whole cluster; the base input ranges over {0,1}^2):
  obs     Theta = {}
  single  obs plus |Theta| = 1 with interchange values
  all     every Theta, every combination of interchange values
  dense   all plus every non-empty Theta with random real values

Usage: python sigmoid_tau.py [family] [--rank-penalty L] [--planted]
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch
from scipy.linalg import eigh
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = np.array(list(itertools.product([0.0, 1.0], repeat=2)))
torch.set_default_dtype(torch.float64)


def sigmoid(z):
    return 0.5 * (1 + np.tanh(0.5 * z))


class Net:
    def __init__(self, Ws, bs, acts):
        self.Ws, self.bs, self.acts = Ws, bs, acts

    def forward(self, x, iv=None):
        """Post-activations per layer; `iv` maps (layer, idx) -> value, set after the activation."""
        X = [np.array(x, dtype=float)]
        for l, (W, b, act) in enumerate(zip(self.Ws, self.bs, self.acts), start=1):
            a = act(W @ X[-1] + b)
            for (layer, idx), val in (iv or {}).items():
                if layer == l:
                    a[idx] = val
            X.append(a)
        return X


def xor_net():
    w = json.load(open(os.path.join(HERE, "data", "xor_mlp_weights.json")))
    alpha = w["negative_slope"]
    leaky = lambda z: np.where(z >= 0, z, alpha * z)
    return Net([np.array(m) for m in w["W"]], [np.array(v) for v in w["b"]], [leaky] * len(w["W"]))


def contexts(net, rec, family, n_dense=20, seed=0):
    """(Theta, base input, interventions, X) for every intervention in the family."""
    rng = np.random.default_rng(seed)
    nodes = {n["id"]: n for n in rec["nodes"]}
    internal = [i for i, n in nodes.items() if n["parents"] and i != rec["root"]]
    clean = [net.forward(x) for x in INPUTS]
    sizes = {"obs": [0], "single": [0, 1]}.get(family, range(len(internal) + 1))
    out = []
    for k in sizes:
        for theta in itertools.combinations(internal, k):
            draws = [{tuple(nn): clean[s][nn[0]][nn[1]] for i, s in zip(theta, srcs) for nn in nodes[i]["neurons"]}
                     for srcs in itertools.product(range(4), repeat=k)]
            if family == "dense" and k:
                draws += [{tuple(nn): rng.normal(0, 2) for i in theta for nn in nodes[i]["neurons"]}
                          for _ in range(n_dense)]
            for x in INPUTS:
                for iv in draws:
                    out.append((set(theta), x, iv, net.forward(x, iv)))
    return out


class Problem:
    """Per-row data of the abstraction condition. Repeated states are merged and weighted by their count."""

    def __init__(self, net, rec, family):
        self.net = net
        nodes = {n["id"]: n for n in rec["nodes"]}
        self.C = {i: [tuple(nn) for nn in n["neurons"]] for i, n in nodes.items()}
        self.src = {i for i, n in nodes.items() if not n["parents"]}
        # repeated leaves (same input neuron) are one regressor
        self.pa = {i: list({tuple(self.C[p]): p for p in n["parents"]}.values()) for i, n in nodes.items() if n["parents"]}
        self.order = []  # topological: a node after its non-source parents
        while len(self.order) < len(self.pa):
            for j in self.pa:
                if j not in self.order and all(p in self.src or p in self.order for p in self.pa[j]):
                    self.order.append(j)
        self.children = {i: [j for j in self.order if i in self.pa[j]] for i in self.order}
        self.hid = [i for i in self.order if self.children[i]]  # non-source nodes whose tau feeds a child
        self.ctx = contexts(net, rec, family)
        val = lambda X, i: np.array([X[l][k] for l, k in self.C[i]])
        self.rows = {}
        for j in self.order:
            D = np.array([np.concatenate([val(X, j)] + [val(X, p) for p in self.pa[j]])
                          for th, _, _, X in self.ctx if j not in th])
            U, w = np.unique(np.round(D, 12), axis=0, return_counts=True)
            cut = np.cumsum([len(self.C[j])] + [len(self.C[p]) for p in self.pa[j]])
            blocks = np.split(U, cut[:-1], axis=1)
            self.rows[j] = (blocks[0], dict(zip(self.pa[j], blocks[1:])), w.astype(float))

    def jacobians(self, i, j, eps=1e-6):
        """d x_{C_j} / d x_{C_i} at every context where j is not intervened on, all of j's parents held fixed."""
        held = [nn for p in self.pa[j] if p not in self.src for nn in self.C[p]]
        out = []
        for th, x, iv, X in self.ctx:
            if j in th:
                continue
            base = {**iv, **{nn: X[nn[0]][nn[1]] for nn in held}}
            y0 = np.array([X[l][k] for l, k in self.C[j]])
            J = np.empty((len(self.C[j]), len(self.C[i])))
            for c, nn in enumerate(self.C[i]):
                Y = self.net.forward(x, {**base, nn: X[nn[0]][nn[1]] + eps})
                J[:, c] = (np.array([Y[l][k] for l, k in self.C[j]]) - y0) / eps
            out.append(J)
        return np.array(out)


def canon(t):
    t = t / np.linalg.norm(t)
    return t * np.sign(t[np.argmax(np.abs(t))])


def row_fit(Xj, G, w, t=None):
    """Smallest generalised eigenvector t (or the given t) of one row; returns t, rel^2 and the coefficients on [G, 1]."""
    sw = np.sqrt(w)[:, None]
    M = np.column_stack([G, np.ones(len(Xj))]) * sw
    Xw = Xj * sw
    R = Xw - M @ np.linalg.lstsq(M, Xw, rcond=None)[0]
    Cw = (Xj - np.average(Xj, axis=0, weights=w)) * sw
    A, B = R.T @ R, Cw.T @ Cw
    if t is None:
        reg = 1e-12 * np.trace(B) + 1e-300
        t = canon(eigh(A, B + reg * np.eye(len(B)))[1][:, 0])
    rel2 = (t @ A @ t) / max(t @ B @ t, 1e-300)
    return t, rel2, np.linalg.lstsq(M, Xw @ t, rcond=None)[0]


def sweep(P, theta, dirs):
    """A1/A2 forward pass: the parameters and rel^2 per row, given (a, b) for every node in P.hid."""
    ab = dict(zip(P.hid, np.reshape(theta, (-1, 2))))
    fit = {}
    for j in P.order:
        Xj, Vp, w = P.rows[j]
        G = np.column_stack([Vp[p][:, 0] if p in P.src else sigmoid(Vp[p] @ fit[p]["t"] + fit[p]["beta"])
                             for p in P.pa[j]])
        u, rel2, coef = row_fit(Xj, G, w, dirs.get(j))
        k, beta = 1.0, 0.0  # a sink's scale and shift are free
        if j in ab:
            s = Xj @ u
            mu = np.average(s, weights=w)
            sd = np.sqrt(np.average((s - mu) ** 2, weights=w)) or 1.0
            k, beta = ab[j][0] / sd, ab[j][1] - ab[j][0] * mu / sd
        fit[j] = {"t": k * u, "beta": beta, "w": k * coef[:-1], "c": k * coef[-1], "rel2": rel2}
    return fit


def varpro(P, dirs=None, starts=8, seed=0):
    dirs = dirs or {}
    if not P.hid:
        return sweep(P, [], dirs)
    rng = np.random.default_rng(seed)
    f = lambda th: sum(r["rel2"] for r in sweep(P, th, dirs).values())
    best = None
    for _ in range(starts):
        n = len(P.hid)
        th0 = np.column_stack([rng.choice([-1, 1], n) * rng.uniform(0.5, 6, n), rng.uniform(-2, 2, n)]).ravel()
        r = minimize(f, th0, method="L-BFGS-B", bounds=[(-25, 25)] * len(th0), options={"ftol": 1e-15, "gtol": 1e-10})
        if best is None or r.fun < best.fun:
            best = r
    return sweep(P, best.x, dirs)


def a1_directions(P):
    """t_i for every multi-neuron hidden node from the stacked gradients of its children; also s2/s1 of the stack."""
    dirs, rank1 = {}, {}
    for i in P.hid:
        if len(P.C[i]) < 2:
            continue
        rows = np.concatenate([P.jacobians(i, j).reshape(-1, len(P.C[i])) for j in P.children[i]])
        nrm = np.linalg.norm(rows, axis=1)
        keep = nrm > 1e-9 * max(nrm.max(), 1e-300)
        if not keep.any():
            continue  # no child reads this cluster: fall back to its own row
        _, s, Vt = np.linalg.svd(rows[keep] / nrm[keep, None], full_matrices=False)
        dirs[i], rank1[i] = canon(Vt[0]), s[1] / s[0]
    return dirs, rank1


def joint(P, rank_penalty=0.0, starts=4, seed=0):
    gen = torch.Generator().manual_seed(seed)
    rows = {j: (torch.as_tensor(Xj), {p: torch.as_tensor(V) for p, V in Vp.items()}, torch.as_tensor(w / w.sum()))
            for j, (Xj, Vp, w) in P.rows.items()}
    pairs = []  # (i, j, Jacobians) for the rank-one penalty
    if rank_penalty:
        pairs = [(i, j, torch.as_tensor(P.jacobians(i, j))) for i in P.hid if len(P.C[i]) > 1 for j in P.children[i]]

    def rel2(t, beta, wt, c):
        """Row j uses the unit direction t_j / |t_j|: rel_j is scale-invariant, so |t_j| only acts through the children's
        sigmoids and shrinking t_j cannot fake a small residual. (w_j, c_j) are in units of that direction."""
        out = {}
        for j in P.order:
            Xj, Vp, w = rows[j]
            G = torch.stack([Vp[p][:, 0] if p in P.src else torch.sigmoid(Vp[p] @ t[p] + beta[p]) for p in P.pa[j]], 1)
            s = Xj @ (t[j] / t[j].norm())
            var = (w * (s - (w * s).sum()) ** 2).sum()
            out[j] = (w * (s - G @ wt[j] - c[j]) ** 2).sum() / var
        return out

    def penalty(t):
        pen = torch.zeros(())
        for i, j, J in pairs:
            M = torch.einsum("nab,a->nb", J, t[j])
            u = t[i] / t[i].norm()
            pen = pen + ((M - (M @ u)[:, None] * u) ** 2).sum() / (M ** 2).sum().clamp_min(1e-12)
        return pen

    best = None
    for _ in range(starts):
        t = {j: torch.randn(len(P.C[j]), generator=gen).requires_grad_() for j in P.order}
        beta = {j: torch.randn((), generator=gen).requires_grad_() for j in P.order}
        wt = {j: torch.zeros(len(P.pa[j]), requires_grad=True) for j in P.order}
        c = {j: torch.zeros((), requires_grad=True) for j in P.order}
        params = [*t.values(), *beta.values(), *wt.values(), *c.values()]
        opt = torch.optim.LBFGS(params, max_iter=500, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-12, tolerance_change=1e-16)

        def closure():
            opt.zero_grad()
            loss = sum(rel2(t, beta, wt, c).values()) + rank_penalty * penalty(t)
            loss.backward()
            return loss

        for _ in range(4):
            opt.step(closure)
        with torch.no_grad():
            r = rel2(t, beta, wt, c)
            total = float(sum(r.values()) + rank_penalty * penalty(t))
        if best is None or total < best[0]:
            k = {j: float(t[j].norm()) for j in P.order}  # back to the units of t_j
            best = (total, {j: {"t": t[j].detach().numpy().copy(), "beta": float(beta[j]) if j in P.hid else 0.0,
                                "w": k[j] * wt[j].detach().numpy(), "c": k[j] * float(c[j]), "rel2": float(r[j])}
                            for j in P.order})
    return best[1]


def fit_all(P, rank_penalty):
    a2 = varpro(P)
    dirs, rank1 = a1_directions(P)
    a1 = varpro(P, dirs) if dirs else a2
    a3 = joint(P, rank_penalty)
    return {"A1": a1, "A2": a2, "A3": a3}, rank1


def worst(fit):
    return max(np.sqrt(max(r["rel2"], 0.0)) for r in fit.values())


def to_json(P, fit):
    return {str(j): {"t": np.asarray(r["t"]).tolist(), "beta": r["beta"], "c": r["c"], "rel": float(np.sqrt(max(r["rel2"], 0))),
                     "w": {str(p): float(v) for p, v in zip(P.pa[j], r["w"])}} for j, r in fit.items()}


def planted(rank_penalty):
    """A net with an exact sigmoid abstraction whose 2-neuron clusters are only identifiable through their children.

    inputs -> 4 linear units (clusters HA = {0,1}, HB = {2,3}) -> 2 sigmoid units, each reading one cluster along tA / tB
    -> linear output. The sigmoid layer is marginalised. Then tau_HA = sigmoid(tA . x_HA + bA) etc. is exact for every
    intervention, but HA's own row holds for every direction, since the cluster is linear in the inputs.
    """
    rng = np.random.default_rng(1)
    tA, tB = canon(rng.normal(size=2)), canon(rng.normal(size=2))
    W2 = np.zeros((2, 4))
    W2[0, :2], W2[1, 2:] = 2 * tA, 2 * tB
    ident = lambda z: z
    net = Net([rng.normal(size=(4, 2)), W2, np.array([[1.5, -2.0]])],
              [rng.normal(size=4), np.array([0.3, -0.5]), np.array([0.1])], [ident, sigmoid, ident])
    leaf = lambda i, k: {"id": i, "neurons": [[0, k]], "parents": []}
    rec = {"root": 0, "nodes": [{"id": 0, "neurons": [[3, 0]], "parents": [1, 2]},
                                {"id": 1, "neurons": [[1, 0], [1, 1]], "parents": [3, 4]},
                                {"id": 2, "neurons": [[1, 2], [1, 3]], "parents": [3, 4]}, leaf(3, 0), leaf(4, 1)]}
    print("Planted check: exact abstraction exists; angle = degrees between fitted and planted t for HA, HB.\n")
    for family in ["all", "dense"]:
        P = Problem(net, rec, family)
        fits, rank1 = fit_all(P, 0.0)
        fits["A3+rank"] = joint(P, rank_penalty or 1.0)
        ang = lambda t, ref: np.degrees(np.arccos(min(1.0, abs(canon(t) @ ref))))
        print(f"  {family:<6} rank1 = " + ", ".join(f"{v:.1e}" for v in rank1.values()))
        for m, fit in fits.items():
            print(f"    {m:<8} max rel {worst(fit):.2e}   angle HA {ang(fit[1]['t'], tA):5.1f}  HB {ang(fit[2]['t'], tB):5.1f}",
                  flush=True)
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("family", nargs="?", default="all", choices=["obs", "single", "all", "dense"])
    ap.add_argument("--rank-penalty", type=float, default=0.0)
    ap.add_argument("--planted", action="store_true", help="run only the planted check")
    args = ap.parse_args()

    planted(args.rank_penalty)
    if args.planted:
        raise SystemExit
    net = xor_net()
    records = json.load(open(os.path.join(HERE, "data", "xor_mappings.json")))
    print(f"XOR MLP, family '{args.family}': max relative row residual (0 = exact). rank1 = s2/s1 of A1's gradient stack.\n")
    print(f"{'#':>3} {'formula':<34}{'A1':>10}{'A2':>10}{'A3':>10}   rank1")
    out = []
    for n, rec in enumerate(records):
        P = Problem(net, rec, args.family)
        fits, rank1 = fit_all(P, args.rank_penalty)
        print(f"{n:>3} {rec['formula']:<34}" + "".join(f"{worst(f):>10.2e}" for f in fits.values())
              + ("   " + ", ".join(f"{v:.1e}" for v in rank1.values()) if rank1 else ""), flush=True)
        out.append({"formula": rec["formula"], "rank1": {str(k): v for k, v in rank1.items()},
                    **{m: to_json(P, f) for m, f in fits.items()}})
    path = os.path.join(HERE, "data", f"sigmoid_tau_fits_{args.family}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nFitted parameters -> {path}")
