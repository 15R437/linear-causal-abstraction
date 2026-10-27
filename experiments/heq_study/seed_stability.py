"""Seed spread of the headline fit, and how often it collapses.

The fit is a single non-convex run with no restarts, and it sometimes ends with every activation in one cell -- a
degenerate optimum of the same objective (one cell, the majority class, IIA = base rate). That is visible in `cells`
without looking at anything else, so the right protocol is to fit a few seeds and drop the collapsed ones. This
script measures how often that happens and what the spread is among the runs that do not.

Usage: python seed_stability.py [--dim 8] [--seeds 8] [--specs 2:2:1,2:2:2]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.algos.sigmoid_abstract import evaluate, fit, split_sequential
from hierarchical_equality import bit_recovery, best_matching, dependence, get_net, parse_specs, sample_batch


class Args:
    retrain = False
    train_steps = 6000
    verbose = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--cut", type=int, default=2)
    parser.add_argument("--specs", default="2:2:1,2:2:2")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    mlp = get_net(args.pairs, args.dim, args.width, args.depth, Args())
    to_layer, from_layer = split_sequential(mlp, args.cut)
    g = torch.Generator().manual_seed(1234)
    X_fit, _, _ = sample_batch(8192, args.pairs, args.dim, g)
    X_eval, _, bits = sample_batch(8192, args.pairs, args.dim, g)
    with torch.no_grad():
        H_eval = to_layer(X_eval).detach()

    print(f"pairs={args.pairs} dim={args.dim} depth={args.depth} cut={args.cut}, {args.seeds} seeds\n")
    for label, spec in parse_specs(args.specs):
        rows = []
        for seed in range(args.seeds):
            model, _ = fit(to_layer, from_layer, X_fit, spec, family="all", steps=args.steps, seed=seed,
                           verbose=False)
            s = evaluate(model, to_layer, from_layer, X_eval, seed=1000)["all"]
            rows.append((seed, s["cells"], s["iia"], s["purity"], min(dependence(model, H_eval)),
                         best_matching(bit_recovery(model, H_eval, bits))))
        good = [r for r in rows if r[1] > 1]
        print(f"  {label}:  {len(good)}/{len(rows)} seeds did not collapse to one cell")
        print(f"    {'seed':>4s} {'cells':>5s} {'IIA(all)':>8s} {'purity':>7s} {'min dep(V)':>10s} {'bits':>6s}")
        for seed, cells, iia, purity, dep, b in rows:
            flag = "" if cells > 1 else "   <- collapsed"
            print(f"    {seed:>4d} {cells:>5d} {iia:>8.3f} {purity:>7.3f} {dep:>10.2f} {b:>6.3f}{flag}")
        if good:
            arr = np.array([[r[2], r[3], r[5]] for r in good])
            m, sd = arr.mean(0), arr.std(0)
            print(f"    non-collapsed mean +- sd:  IIA {m[0]:.3f} +- {sd[0]:.3f}   purity {m[1]:.3f} +- {sd[1]:.3f}"
                  f"   bits {m[2]:.3f} +- {sd[2]:.3f}\n")


if __name__ == "__main__":
    main()
