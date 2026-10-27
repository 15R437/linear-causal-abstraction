"""Abstraction of a hierarchical-equality MLP: a task whose solution *must* carry several variables.

The MNIST and XOR nets only ever abstracted at one high-level variable, and the reason is structural rather than a
fitting failure: a classifier's decision is one region of activation space, so one multi-valued variable on a small
block already says everything the condition can test, and nothing forces the network to hold two independently
settable pieces of state. Hierarchical equality does force it.

Task (Geiger et al.'s hierarchical equality, generalised to P pairs). An input is 2P objects, each a vector in
R^n; pair i is made identical with probability 1/2. The bits are b_i = 1[x_{2i} = x_{2i+1}] and the label is their
parity,

    y = 1[ b_1 + ... + b_P  is even ].

At P = 2 that is the usual "(a = b) = (c = d)". Two properties make it the right test:
  - the bits are *independent* and uniform, so all 2^P bit patterns occur in the data and an interchange
    intervention on one bit names a state the network itself produces;
  - parity has no margin-free shortcut: y depends on every bit, so a net that solves the task must hold all P bits
    at once, and cannot trade one off against another the way a halfspace can.

NOTE ON NOTATION. P is the number of pairs in the TASK; k is the number of high-level VARIABLES in the abstraction
(`model.k` in sigmoid_abstract.py). They coincide only when we fit one variable per pair, i.e. the spec `P:2:d`;
the single multi-valued fits such as `1:8:7` have k = 1 at P = 3. The `--dim` flag is n, the object dimension, and
is unrelated to d_i, the dimensions of the layer each variable reads.

The cut matters, and is the point. Cut after the LAST hidden layer and the tail is linear, so the decision is a single
halfspace and k = 1 with d = 1 reaches IIA 1 for free -- the degenerate case the theory note warns about. Cut after the
FIRST hidden layer and the parity is still ahead of us in the network: a high-level model with k variables now has to
get each bit's subspace right *and* carry the parity table, and intervening on one bit must flip the output.

Diagnostics beyond IIA, both aimed at "is this really k variables?":
  deps     for each variable, how often changing its value alone changes the predicted class, over the realised
           cells. A 2-variable model that scores IIA 1 by putting one variable on the decision direction and ignoring
           the other is a 1-variable model wearing k clothes, and `deps` is what catches it
  bits     recovery of the ground-truth bit b_j from the learned variable tau_i, via the best map from tau_i's values
           to {0,1}. IIA says the abstraction is valid; this says it found the task's own variables

Usage: python hierarchical_equality.py [--pairs 2] [--dim 4] [--width 64] [--cut 2] [--specs 1:2:1,2:2:1,...]
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.algos.sigmoid_abstract import (degeneracy, evaluate, fit, graded_baseline, split_sequential,
                                        truth_table)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "..", "data")


def sample_batch(n, pairs, dim, generator):
    """(X, y, bits): 2*pairs objects in R^dim, each pair equal with probability 1/2, label = parity of the bits."""
    objs = torch.rand(n, 2 * pairs, dim, generator=generator) * 2 - 1
    eq = torch.rand(n, pairs, generator=generator) < 0.5
    for i in range(pairs):
        objs[:, 2 * i + 1] = torch.where(eq[:, i, None], objs[:, 2 * i], objs[:, 2 * i + 1])
    bits = eq.long()
    return objs.reshape(n, -1), ((bits.sum(-1) % 2) == 0).long(), bits


def train_mlp(pairs, dim, width, depth=2, steps=6000, batch=512, lr=1e-3, seed=0, verbose=True):
    """2*pairs*dim -> (width -> ReLU) * depth -> 2. Cut 2*l is after hidden layer l.

    Depth is a knob on the result, not a detail. The bits have to be computed before the parity, so with depth 2 the
    only cut that leaves the parity ahead of us is after layer 1 -- where the equality comparisons may not be finished
    yet. Depth 3 gives the net two layers to form the bits and still leaves the parity behind cut 4.

    Trained on fresh batches, so there is no training set to memorise and the test accuracy below is the task.
    """
    torch.manual_seed(seed)
    layers = []
    for d_in in [2 * pairs * dim] + [width] * (depth - 1):
        layers += [nn.Linear(d_in, width), nn.ReLU()]
    mlp = nn.Sequential(*layers, nn.Linear(width, 2))
    optimiser = torch.optim.Adam(mlp.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed + 1)
    for step in range(steps):
        X, y, _ = sample_batch(batch, pairs, dim, generator)
        loss = nn.functional.cross_entropy(mlp(X), y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if verbose and (step % 1000 == 0 or step == steps - 1):
            print(f"  step {step:5d}  loss {loss.item():.4f}")
    return mlp.eval()


def get_net(pairs, dim, width, depth, args):
    path = os.path.join(CACHE, f"heq_p{pairs}_d{dim}_w{width}_L{depth}.pt")
    if os.path.exists(path) and not args.retrain:
        mlp = train_mlp(pairs, dim, width, depth, steps=0, verbose=False)
        mlp.load_state_dict(torch.load(path, weights_only=True))
        mlp.eval()
    else:
        print(f"training pairs={pairs} dim={dim} width={width} depth={depth}")
        mlp = train_mlp(pairs, dim, width, depth, steps=args.train_steps, verbose=args.verbose)
        os.makedirs(CACHE, exist_ok=True)
        torch.save(mlp.state_dict(), path)
    return mlp


def parse_specs(text):
    """"k:m:d" -> [(m, d)] * k: k high-level variables, each taking m values, each reading a d-dimensional block.

    So `2:2:1` is two binary variables on one dimension each (D = 2 dimensions of the layer in total), and `1:4:3`
    is one four-valued variable on a three-dimensional block. `1:2:1` is the trivial baseline: a single halfspace.
    """
    out = []
    for item in text.split(","):
        k, m, d = (int(v) for v in item.strip().split(":"))
        out.append((item.strip(), [(m, d)] * k))
    return out


@torch.no_grad()
def dependence(model, h):
    """For each variable, the share of realised cells whose predicted class changes when that variable alone is
    changed to some other value. 0 means the mechanism ignores the variable, so the abstraction has fewer effective
    variables than it claims."""
    cells, pred = model.cells.numpy(), model.M.argmax(-1).numpy()
    radix = model.radix.numpy()
    realised = set(model.cell_index(h).tolist())
    out = []
    for i in range(model.k):
        hits, seen = 0, 0
        for c in sorted(realised):
            tup = cells[c]
            # only alternatives that are themselves realised: M is undefined on cells nothing ever produced, so a
            # change into one of those says nothing about whether the mechanism uses the variable
            alts = [pred[c + (v - tup[i]) * radix[i]] for v in range(model.sizes[i])
                    if v != tup[i] and c + (v - tup[i]) * radix[i] in realised]
            if alts:
                seen += 1
                hits += any(a != pred[c] for a in alts)
        out.append(hits / max(seen, 1))
    return out


@torch.no_grad()
def bit_recovery(model, h, bits):
    """(k, pairs) accuracy of predicting ground-truth bit j from learned variable i, using the best map from the
    variable's m_i values to {0, 1}: for each value, the majority bit among the activations taking it."""
    values = model.values(h).numpy()
    bits = bits.numpy()
    acc = np.zeros((model.k, bits.shape[1]))
    for i in range(model.k):
        for j in range(bits.shape[1]):
            correct = 0
            for v in range(model.sizes[i]):
                sel = values[:, i] == v
                if sel.any():
                    correct += max(int((bits[sel, j] == 0).sum()), int((bits[sel, j] == 1).sum()))
            acc[i, j] = correct / len(bits)
    return acc


@torch.no_grad()
def joint_recovery(model, h, bits):
    """Accuracy of predicting the WHOLE bit vector (b_1, ..., b_k) from the joint high-level state tau(h), using the
    best map from realised cells to bit patterns (the majority pattern in each cell).

    This is the comparison that is fair across specs. `bit_recovery` is per-variable, so for a single multi-valued
    variable it only reports the best single bit and understates what that variable carries; this asks whether the
    high-level state determines the task's full state, which is what "it found the variables" should mean.
    """
    idx = model.cell_index(h).numpy()
    pattern = (bits.numpy() * (2 ** np.arange(bits.shape[1]))).sum(1)
    correct = 0
    for c in np.unique(idx):
        sel = idx == c
        correct += int(np.bincount(pattern[sel]).max())
    return correct / len(idx)


def best_matching(acc):
    """Mean recovery under the best one-to-one assignment of learned variables to ground-truth bits (the smaller
    side is matched in full, so with k < pairs some bit is left unclaimed and with k > pairs some variable is)."""
    k, n = acc.shape
    if k <= n:
        return max(np.mean([acc[i, j] for i, j in zip(range(k), p)]) for p in itertools.permutations(range(n), k))
    return max(np.mean([acc[i, j] for i, j in zip(p, range(n))]) for p in itertools.permutations(range(k), n))


def run(mlp, cut, X_fit, X_eval, bits_eval, specs, args):
    to_layer, from_layer = split_sequential(mlp, cut)
    with torch.no_grad():
        H_fit = to_layer(X_fit).detach()
        H_eval = to_layer(X_eval).detach()
    p = H_fit.shape[1]
    n_distinct, _, shatterable = degeneracy(H_fit, 1)
    print(f"cut after module {cut}: p = {p}, {n_distinct} distinct activations in the fitting pool"
          f"{'  -- SHATTERABLE' if shatterable else ''}")
    print(f"fitted on `{args.family}`, {args.n_interventions} frozen triples, {args.steps} steps\n")
    print("legend:  k:m:d = k high-level variables, each with m values, each on a d-dimensional block; D = k*d")
    print("         dep(V) = share of realised cells whose predicted class changes when that variable alone is")
    print("                  changed -- 0 means the mechanism ignores it, so the model has fewer real variables")
    print("         bits   = recovery of the task's own bits b_j from the learned variables (0.5 = chance)")
    print("         ctrl   = best skill of a FROZEN RANDOM subspace of the same size\n")
    print(f"  {'k:m:d':10s} {'D':>4s} {'IIA(obs)':>9s} {'IIA(sing)':>10s} {'IIA(all)':>9s} {'skill':>7s} "
          f"{'purity':>7s} {'graded':>7s} {'cells':>8s} {'unseen':>7s} {'sharp':>6s} {'dep(V)':>14s} {'bits':>6s} "
          f"{'ctrl':>7s}")
    rows = []
    for label, spec in specs:
        t0 = time.time()
        model, _ = fit(to_layer, from_layer, X_fit, spec, family=args.family, steps=args.steps,
                       n_interventions=args.n_interventions, kappa=args.kappa, seed=args.seed, verbose=False)
        scores = evaluate(model, to_layer, from_layer, X_eval, n=args.n_eval, seed=args.seed + 1)
        graded = graded_baseline(model, to_layer, from_layer, X_eval, family="all", n=args.n_eval)
        deps = dependence(model, H_eval)
        acc = bit_recovery(model, H_eval, bits_eval)
        ctrl = max(evaluate(fit(to_layer, from_layer, X_fit, spec, family=args.family, steps=args.steps,
                                n_interventions=args.n_interventions, kappa=args.kappa, freeze_directions=True,
                                seed=100 + r, verbose=False)[0],
                            to_layer, from_layer, X_eval, n=args.n_eval, seed=args.seed + 1)["all"]["skill"]
                   for r in range(args.control_draws))
        s = scores["all"]
        print(f"  {label:10s} {model.D:>4d} {scores['obs']['iia']:>9.3f} {scores['single']['iia']:>10.3f} "
              f"{s['iia']:>9.3f} {s['skill']:>+7.3f} {s['purity']:>7.3f} {graded:>7.3f} "
              f"{str(s['cells']) + '/' + str(s['cells_total']):>8s} {s['unseen']:>7.2f} {s['sharp']:>6.1f} "
              f"{'[' + ' '.join(f'{v:.2f}' for v in deps) + ']':>14s} {best_matching(acc):>6.3f} {ctrl:>+7.3f} "
              f" [{time.time() - t0:.0f}s]")
        rows.append((label, model, scores, deps, acc))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=2, help="k high-level bits; the label is their parity")
    parser.add_argument("--dim", type=int, default=4, help="dimension of each object vector")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2, help="hidden layers in the net")
    parser.add_argument("--cut", type=int, default=2, help="module index of the cut; 2*l is after hidden layer l")
    parser.add_argument("--specs", default=None, help="comma-separated k:m:d; default depends on --pairs")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--n-interventions", type=int, default=4096)
    parser.add_argument("--n-eval", type=int, default=4096)
    parser.add_argument("--pool", type=int, default=8192)
    parser.add_argument("--kappa", type=float, default=8.0)
    parser.add_argument("--family", default="all")
    parser.add_argument("--control-draws", type=int, default=3)
    parser.add_argument("--train-steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--truth-table", action="store_true", help="print the learned mechanism of the last spec")
    args = parser.parse_args()

    k = args.pairs
    specs = parse_specs(args.specs if args.specs else
                        ",".join([f"1:2:1", f"1:{2 ** k}:{2 ** k - 1}"] + [f"{k}:2:{d}" for d in (1, 2, 4)]))

    mlp = get_net(k, args.dim, args.width, args.depth, args)
    generator = torch.Generator().manual_seed(1234)
    X_fit, y_fit, _ = sample_batch(args.pool, k, args.dim, generator)
    X_eval, y_eval, bits_eval = sample_batch(args.pool, k, args.dim, generator)
    with torch.no_grad():
        acc_fit = (mlp(X_fit).argmax(-1) == y_fit).double().mean().item()
        acc = (mlp(X_eval).argmax(-1) == y_eval).double().mean().item()
        # per-bit accuracy of a linear probe is not the question here; what matters is that the net solves the task
    widths = "-".join([str(2 * k * args.dim)] + [str(args.width)] * args.depth + ["2"])
    print(f"\nhierarchical equality: {k} pairs, dim {args.dim}, net {widths}, cut {args.cut}")
    print(f"test accuracy {acc:.4f} (fit pool {acc_fit:.4f}), label base rate "
          f"{max(y_eval.double().mean().item(), 1 - y_eval.double().mean().item()):.3f}\n")

    rows = run(mlp, args.cut, X_fit, X_eval, bits_eval, specs, args)
    if args.truth_table:
        label, model, _, deps, acc = rows[-1]
        print(f"\nmechanism for {label} (variable values -> class):")
        for row in truth_table(model):
            print("   ", " ".join(str(v) for v in row[:-1]), "->", row[-1])
        print(f"  bit recovery matrix (learned variable x ground-truth bit):\n{np.round(acc, 3)}")


if __name__ == "__main__":
    main()
