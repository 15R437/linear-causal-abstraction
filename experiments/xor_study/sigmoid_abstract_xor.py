"""Boolean abstraction of the XOR MLP's hidden layers by learned directions, and a planted recovery check.

Two tests of `src.algos.sigmoid_abstract`, both with a known answer.

XOR MLP (2-3-3-1 LeakyReLU, the MI-identifiability net). Cutting after the second hidden layer, the net's decision is
1[leaky(w_3 . h_2 + b_3) > 0.5] = 1[w_3 . h_2 + b_3 > 0.5], a single halfspace, so k = 1 must reach IIA 1: a known
answer for the fitter to hit. The *direction* is not a known answer, though -- with only 4 clean inputs, every hyperplane
that splits those activations the same way satisfies every interchange intervention, so t_1 need not be w_3/|w_3| and a
fit that scores IIA 1 with cos(t_1, w_3) well below 1 is a different valid abstraction, not a failure. The seed spread
below measures how big that solution set is. Four inputs also make the k sweep uninformative here: IIA saturates from
k = 1, so this test is about correctness of the machinery, not about compression.

Planted (--planted). A random orthonormal U in R^{64 x 3} and a random Boolean function f: {0,1}^3 -> {0,..,3} define a
tail that reads only the three signs, f(1[U^T h > 0]). The fitter sees nothing but behaviour, so recovering it means
IIA 1 at k = 3 and ||U U^T - T T^T||_F ~ 0, and the k sweep should show the knee at 3. This is the test that exercises
the optimiser; the XOR net is too small to.

Usage: python sigmoid_abstract_xor.py [--planted] [--family single] [--steps 2000]
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.algos.sigmoid_abstract import (degeneracy, evaluate, fit, graded_baseline, split_sequential,
                                        subspace_error, truth_table)

torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = torch.tensor(list(itertools.product([0.0, 1.0], repeat=2)))


def xor_net():
    """The pretrained XOR MLP as an nn.Sequential. Every layer carries the activation, the output one included."""
    w = json.load(open(os.path.join(HERE, "data", "xor_mlp_weights.json")))
    layers = []
    for W, b in zip(w["W"], w["b"]):
        W, b = np.array(W), np.array(b)
        linear = nn.Linear(W.shape[1], W.shape[0])
        with torch.no_grad():
            linear.weight.copy_(torch.tensor(W))
            linear.bias.copy_(torch.tensor(b))
        layers += [linear, nn.LeakyReLU(w["negative_slope"])]
    return nn.Sequential(*layers).eval(), w


def report(name, scores, extra=""):
    cols = " ".join(f"{f} {s['iia']:.3f}" for f, s in scores.items())
    s = scores["all"]
    print(f"  {name:22s} IIA[{cols}]  margin {s['margin_min']:.3f}/{s['margin_mean']:.3f} sharp {s['sharp']:.1f} "
          f"cells {s['cells']}/{s['cells_total']} unseen {s['unseen']:.2f} purity {s['purity']:.3f} {extra}")


def run_xor(args):
    net, w = xor_net()
    X = INPUTS.clone()
    print("XOR MLP 2-3-3-1, decision = 1[y > 0.5], fitted on the `single` family\n")
    for cut, label in ((2, "h1 (after layer 1)"), (4, "h2 (after layer 2)")):
        to_layer, from_layer = split_sequential(net, cut)
        p = to_layer(X).shape[1]
        n_distinct, dof, shatterable = degeneracy(to_layer(X).detach(), 1)
        print(f"cut at {label}, p = {p}: {n_distinct} distinct activations vs {dof} free parameters at k = 1"
              + ("  -- SHATTERABLE: a halfspace realises every labelling of them, so agreement on the"
                 " observational activations cannot fail" if shatterable else ""))
        for k in range(1, p + 1):
            model, _ = fit(to_layer, from_layer, X, k, family=args.family, steps=args.steps, verbose=False)
            scores = evaluate(model, to_layer, from_layer, X)
            with torch.no_grad():
                assert torch.allclose(model.T.T @ model.T, torch.eye(k), atol=1e-8), "columns of T not orthonormal"
            extra = ""
            if cut == 4 and k == 1:
                w3 = torch.tensor(np.array(w["W"][-1])[0])
                extra = f"|cos(t_1, w_3)| {abs(float(model.T[:, 0] @ w3 / w3.norm())):.4f}"
            report(f"k = {k}", scores, extra)
            graded = graded_baseline(model, to_layer, from_layer, X, family=args.family)
            print(f"    graded (linear in T^T h) readout accuracy: {graded:.3f}")
        print()

    # how big is the solution set? every seed that scores IIA 1 is a valid abstraction, so the spread of the recovered
    # direction is a measure of how little the interchange interventions on 4 inputs pin T down.
    to_layer, from_layer = split_sequential(net, 4)
    w3 = torch.tensor(np.array(w["W"][-1])[0])
    print("h2, k = 1 over seeds (every IIA 1.0 row is an equally valid abstraction):")
    for seed in range(5):
        model, _ = fit(to_layer, from_layer, X, 1, family=args.family, steps=args.steps, seed=seed, verbose=False)
        scores = evaluate(model, to_layer, from_layer, X)
        table = " ".join(f"{row[0]}->{row[-1]}" for row in truth_table(model))
        print(f"    seed {seed}  IIA(all) {scores['all']['iia']:.3f}  "
              f"|cos(t_1, w_3)| {abs(float(model.T[:, 0] @ w3 / w3.norm())):.4f}  mechanism {table}")


def planted_tail(U, table, n_classes):
    """f(1[U^T h > 0]) as one-hot logits, so argmax recovers the planted Boolean function."""
    pow2 = 2 ** torch.arange(U.shape[1] - 1, -1, -1)

    def from_layer(h):
        idx = (((h @ U) > 0).long() * pow2).sum(-1)
        return torch.eye(n_classes)[table[idx]]

    return from_layer


def run_planted(args):
    p, k_true, n_classes, n_data = 64, 3, 4, 4096
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    U = torch.tensor(np.linalg.qr(rng.normal(size=(p, k_true)))[0])
    table = torch.tensor(rng.integers(0, n_classes, size=2 ** k_true))
    X = torch.tensor(rng.normal(size=(n_data, p)))
    to_layer, from_layer = (lambda h: h), planted_tail(U, table, n_classes)

    print(f"planted: p = {p}, k* = {k_true}, {n_classes} classes, table {table.tolist()}")
    print(f"fitted on the `{args.family}` family, {args.steps} steps\n")
    for k in range(1, k_true + 2):
        model, _ = fit(to_layer, from_layer, X, k, family=args.family, steps=args.steps, verbose=args.verbose)
        scores = evaluate(model, to_layer, from_layer, X)
        extra = f"subspace err {subspace_error(model, U.numpy()):.4f}" if k == k_true else ""
        report(f"k = {k}", scores, extra)
        if k == k_true:
            print("    recovered mechanism (bits -> class):",
                  " ".join(f"{''.join(map(str, row[:-1]))}:{row[-1]}" for row in truth_table(model)))
            print(f"    planted mechanism            : "
                  f"{' '.join(f'{i:03b}:{v}' for i, v in enumerate(table.tolist()))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--planted", action="store_true", help="run the planted recovery check instead of the XOR net")
    parser.add_argument("--family", default="single", choices=("obs", "single", "all", "dense"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    (run_planted if args.planted else run_xor)(args)


if __name__ == "__main__":
    main()
