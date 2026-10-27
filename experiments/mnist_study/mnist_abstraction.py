"""Boolean abstraction of an MNIST MLP's hidden layer: how few thresholded directions reproduce the net's decision?

Unlike the XOR network, here the condition is evaluated on thousands of distinct activations rather than 4, so it can
actually fail: the number of labellings of n points in general position in R^p that one halfspace realises is
2 * sum_{i<=p} C(n-1, i), which is every labelling only when n <= p+1. With p = 128 and n in the thousands, an IIA of 1
at small k is a real claim about the network rather than a property of the geometry.

Modes:
  binary    3 vs 8, so 2 classes and k = 1 is already expressive enough in principle
  multi     all ten digits. Note that argmax over K classes is decided by the ranking of K linear forms, i.e. by the
            signs of all C(K,2) pairwise differences, so a handful of halfspaces cannot express it however many bits
            they carry -- the counting bound 2^k >= K is far too optimistic
  lt5       a net trained directly on "is the digit < 5": does it represent that predicate in its own right?
  lt5pool   the SAME ten-class net, with its decision pooled to 1[argmax < 5]. Two outcomes, so the argmax obstruction
            is gone, but the ten-way computation inside is untouched. Comparing lt5 with lt5pool separates "this task
            needs k directions" from "this network happens to represent the task that way".

The sweep over k is the result, not a hyperparameter search -- k is the compression being claimed. Watch purity (is the
net's decision really constant on each cell of the arrangement?) and the graded baseline (if a linear readout of the
same directions wins by a wide margin, the layers after the cut use magnitude rather than sign and no number of bits
will do).

Usage: python mnist_abstraction.py [--mode binary|multi] [--cut 2|4] [--ks 1,2,3,4] [--steps 1500]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.algos.sigmoid_abstract import degeneracy, evaluate, fit, graded_baseline, split_sequential

MNIST_ROOT = "/Users/issy/datasets/mnist"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "..", "data")


def load_mnist(mode):
    """Flattened, normalised MNIST as (X_train, y_train, X_test, y_test)."""
    from torchvision import datasets

    out = []
    for train in (True, False):
        ds = datasets.MNIST(root=MNIST_ROOT, train=train, download=False)
        X = ds.data.reshape(len(ds), -1).float() / 255.0
        y = ds.targets
        if mode == "binary":
            keep = (y == 3) | (y == 8)
            X, y = X[keep], (y[keep] == 8).long()
        elif mode == "lt5":
            y = (y < 5).long()
        out += [X, y]
    return out


def train_mlp(X, y, n_classes, widths=(128, 128), epochs=4, lr=1e-3, seed=0, verbose=True):
    """784 -> 128 -> 128 -> n_classes with ReLU, so the cut points are module index 2 (h1) and 4 (h2)."""
    torch.manual_seed(seed)
    sizes = [X.shape[1]] + list(widths)
    layers = []
    for d_in, d_out in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(d_in, d_out), nn.ReLU()]
    layers += [nn.Linear(sizes[-1], n_classes)]
    mlp = nn.Sequential(*layers)
    optimiser = torch.optim.Adam(mlp.parameters(), lr=lr)
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            idx = perm[i:i + 256]
            loss = nn.functional.cross_entropy(mlp(X[idx]), y[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        if verbose:
            with torch.no_grad():
                acc = (mlp(X).argmax(-1) == y).double().mean().item()
            print(f"  epoch {epoch}: train accuracy {acc:.4f}")
    return mlp.eval()


def get_net(mode, args):
    # lt5pool reuses the ten-class network and only changes how its output is read
    train_mode = "multi" if mode == "lt5pool" else mode
    X_tr, y_tr, X_te, y_te = load_mnist(train_mode)
    n_classes = 10 if train_mode == "multi" else 2
    path = os.path.join(CACHE, f"mnist_mlp_{train_mode}.pt")
    if os.path.exists(path) and not args.retrain:
        mlp = train_mlp(X_tr[:1], y_tr[:1], n_classes, epochs=0, verbose=False)
        mlp.load_state_dict(torch.load(path))
        mlp.eval()
    else:
        print(f"training the {train_mode} MLP ({len(X_tr)} examples, {n_classes} classes)")
        mlp = train_mlp(X_tr, y_tr, n_classes, epochs=args.epochs)
        torch.save(mlp.state_dict(), path)
    with torch.no_grad():
        acc = (mlp(X_te).argmax(-1) == y_te).double().mean().item()
    print(f"{train_mode} MLP test accuracy {acc:.4f}\n")
    return mlp, X_tr, X_te, y_te


def visualise(mlp, to_layer, from_layer, X, y, mode, out_dir, steps, n_interventions, class_names, k=1,
              detail_y=None, detail_name="digit"):
    """The learned directions, drawn so they can be read against the digits themselves.

    Each t_j lives in activation space, so its pixel-space picture is the gradient of t_j . h with respect to the
    input, W_1^T diag(1[z > 0]) t_j, averaged over the pool using each unit's ReLU firing rate (the gate makes it
    input-dependent, so this is a linearisation, exact only where the active set does not change).

    `detail_y` is an optional finer label (the true digit, when the task is a pooling of the digits such as "< 5").
    Where it is given, the projection panel shows the distribution per fine label instead of per class, which is what
    reveals whether the net has one feature for the predicate or is pooling several digit detectors.

    For k = 1 the companion panel is the difference of class means -- the naive discriminative feature, computed from
    the data and labels alone with no reference to the network, so agreement means the net found something close to
    the trivial answer. For k = 2 it is the joint scatter of the two projections, which shows which of the four cells
    of the arrangement are actually populated. The bottom row shows real digits drawn from each realised cell.
    """
    import matplotlib.pyplot as plt

    model, _ = fit(to_layer, from_layer, X, k, steps=steps, n_interventions=n_interventions, verbose=False)
    W1 = mlp[0].weight.detach()
    with torch.no_grad():
        H = to_layer(X).detach()
        gate = (H > 0).double().mean(0)                        # per-unit firing rate over the pool
        # each variable here is binary on a one-dimensional block, so its two affine forms differ by a single unit
        # vector: that difference is the direction, and the signed score gap is the distance to the hyperplane
        cols, Z = [], []
        for i, z in enumerate(model.blocks(H)):
            A, a = model.readout(i)
            cols.append(model.T[:, i] * (A[1, 0] - A[0, 0]))
            Z.append(z[:, 0] * (A[1, 0] - A[0, 0]) + (a[1] - a[0]))
        T = torch.stack(cols, dim=1)                           # (p, k) unit directions
        Z = torch.stack(Z, dim=1)                              # (N, k) signed distances
    classes = sorted(y.unique().tolist())
    names = [class_names[c] for c in classes]

    # orient each direction so that a higher projection means "more like the second class listed", which makes
    # "red = towards that class" true in every panel regardless of the polarity the fit happened to land on
    flip = torch.tensor([-1.0 if Z[y == classes[0], j].mean() > Z[y == classes[1], j].mean() else 1.0
                         for j in range(k)])
    Z, T = Z * flip, T * flip
    pixels = [(W1 * (gate * T[:, j]).unsqueeze(1)).sum(0) for j in range(k)]
    bits = (Z > 0).long()

    # Pixel-wise average of every image of the second class minus that of the first: at each pixel, how much more ink
    # that class carries on average. It depends only on the data and the labels, never on the network, and is (up to
    # scale) what a nearest-centroid classifier would use in pixel space. Note it is a different kind of object from
    # the gradient maps -- pixel space versus a linearisation through the first layer -- so only broad sign agreement
    # is meaningful.
    mean_diff = X[y == classes[1]].mean(0) - X[y == classes[0]].mean(0)

    # k = 1 shows direction + class-mean difference + projection; k > 1 shows one map per direction + joint
    n_ex, n_top = 3, (k + 2 if k == 1 else k + 1)
    fig = plt.figure(figsize=(3.7 * n_top, 6.0))
    fig.suptitle("background = 0, ink $\\approx$ 1", fontsize=9)
    grid = fig.add_gridspec(2, n_top, height_ratios=[2.6, 1.1], hspace=0.3, wspace=0.45)

    maps = [(pixels[j], f"$\\nabla_x\\, t_{j + 1}^\\top h$   (red $\\to$ '{names[1]}')")
            for j in range(k)]
    if k == 1:
        maps.append((mean_diff, f"mean('{names[1]}') $-$ mean('{names[0]}')"))
    for col, (img, title) in enumerate(maps):
        ax = fig.add_subplot(grid[0, col])
        lim = img.abs().max()
        im = ax.imshow(img.reshape(28, 28), cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(grid[0, n_top - 1])
    if k == 1 and detail_y is not None:
        fine = sorted(detail_y.unique().tolist())
        data = [Z[detail_y == f, 0].numpy() for f in fine]
        colours = ["tab:orange" if (y[detail_y == f][0] == classes[1]) else "tab:blue" for f in fine]
        bp = ax.boxplot(data, vert=False, labels=[str(f) for f in fine], patch_artist=True, widths=0.6,
                        flierprops=dict(markersize=2, alpha=0.3))
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.6)
        for median in bp["medians"]:
            median.set_color("k")
        ax.axvline(0.0, color="k", lw=1, ls="--")
        ax.set_xlabel("$t_1^\\top h + \\beta_1$")
        ax.set_ylabel(detail_name)
        ax.invert_yaxis()
    elif k == 1:
        for cls, name in zip(classes, names):
            ax.hist(Z[y == cls, 0].numpy(), bins=60, alpha=0.6, label=f"digit '{name}'")
        ax.axvline(0.0, color="k", lw=1, ls="--")
        ax.set_xlabel("$t_1^\\top h + \\beta_1$")
        ax.set_ylabel("count")
    else:
        sub = torch.randperm(len(Z))[:2500]
        for cls, name in zip(classes, names):
            m = sub[(y[sub] == cls)]
            ax.scatter(Z[m, 0].numpy(), Z[m, 1].numpy(), s=4, alpha=0.4, label=f"digit '{name}'")
        ax.axvline(0.0, color="k", lw=1, ls="--")
        ax.axhline(0.0, color="k", lw=1, ls="--")
        ax.set_xlabel("$t_1^\\top h + \\beta_1$")
        ax.set_ylabel("$t_2^\\top h + \\beta_2$")
    ax.set_title(f"projection by true {detail_name if detail_y is not None else 'digit'}", fontsize=10)
    if detail_y is None:
        ax.legend(fontsize=9, markerscale=2)

    # one block of real examples per realised cell, labelled with its bit pattern and what it contains
    cell_id = (bits * (2 ** torch.arange(k - 1, -1, -1))).sum(-1)
    realised = [int(c) for c in cell_id.unique()]
    inner = grid[1, :].subgridspec(1, len(realised) * n_ex + (len(realised) - 1),
                                   width_ratios=sum([[1] * n_ex + [0.5] for _ in realised], [])[:-1], wspace=0.1)
    for b, cell in enumerate(realised):
        idx = (cell_id == cell).nonzero().reshape(-1)
        share = (y[idx] == classes[1]).double().mean().item()
        pattern = format(cell, f"0{k}b")
        if detail_y is not None:
            counts = detail_y[idx].bincount()
            top = counts.argsort(descending=True)[:3].tolist()
            extra = "  top: " + ",".join(f"{t}({counts[t] / len(idx):.0%})" for t in top)
        else:
            extra = ""
        for j, i in enumerate(idx[:n_ex].tolist()):
            ax = fig.add_subplot(inner[0, b * (n_ex + 1) + j])
            ax.imshow(X[i].reshape(28, 28), cmap="gray_r")
            ax.axis("off")
            if j == 0:
                ax.set_title(f"cell {pattern}: n={len(idx)}, {share:.0%} '{names[1]}'{extra}",
                             fontsize=8.5, loc="left")

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"mnist_{mode}_k{k}_directions.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    scores = evaluate(model, to_layer, from_layer, X)
    print(f"  saved {out_dir}/mnist_{mode}_k{k}_directions.png   (IIA(all) {scores['all']['iia']:.3f}, "
          f"purity {scores['all']['purity']:.3f}, {len(realised)} of {2 ** k} cells realised)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="binary", choices=("binary", "multi", "lt5", "lt5pool"))
    parser.add_argument("--cut", type=int, default=2, help="2 = after the first hidden layer, 4 = after the second")
    parser.add_argument("--ks", default="1,2,3,4")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--n-interventions", type=int, default=2048)
    parser.add_argument("--pool", type=int, default=8192, help="activations the condition is evaluated on")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--family", default="single")
    parser.add_argument("--control-draws", type=int, default=3)
    parser.add_argument("--visualise", action="store_true", help="draw the learned directions and exit")
    parser.add_argument("--vis-k", type=int, default=1, help="how many directions to draw (1 or 2)")
    args = parser.parse_args()

    mlp, X_tr, X_te, y_te = get_net(args.mode, args)
    to_layer, from_layer = split_sequential(mlp, args.cut)
    X_fit, X_eval = X_tr[:args.pool], X_te[:args.pool]
    # lt5pool reads the ten-class net's output as a single bit; every other mode uses the net's own argmax
    decision = (lambda out: (out.argmax(-1) < 5).long()) if args.mode == "lt5pool" else None
    n_cls = 2 if args.mode == "lt5pool" else None
    with torch.no_grad():
        p = to_layer(X_fit[:1]).shape[1]

    n_distinct, _, shatterable = degeneracy(to_layer(X_fit).detach(), 1)
    print(f"cut after module {args.cut}: p = {p}, {n_distinct} distinct activations in the fitting pool"
          f"{'  -- SHATTERABLE' if shatterable else ''}")
    print(f"fitted on `{args.family}`, {args.n_interventions} frozen triples, {args.steps} steps\n")

    if args.visualise:
        # in binary mode the labels were remapped, so name the classes by the digit they came from
        if args.mode == "binary":
            class_names, vis_y, detail = {0: "3", 1: "8"}, y_te[:args.pool], None
        elif args.mode in ("lt5", "lt5pool"):
            # colour by the true digit, which shows whether one feature carries the predicate or several are pooled
            class_names = {0: "$\\geq$5", 1: "<5"}
            detail = load_mnist("multi")[3][:args.pool]
            vis_y = (detail < 5).long()
        else:
            class_names, vis_y, detail = {d: str(d) for d in range(10)}, y_te[:args.pool], None
        visualise(mlp, to_layer, from_layer, X_eval, vis_y, args.mode,
                  os.path.join(HERE, "figures"), args.steps, args.n_interventions,
                  class_names, k=args.vis_k, detail_y=detail)
        return

    for k in [int(v) for v in args.ks.split(",")]:
        t0 = time.time()
        model, _ = fit(to_layer, from_layer, X_fit, k, family=args.family, steps=args.steps,
                       n_interventions=args.n_interventions, decision=decision, n_classes=n_cls, verbose=False)
        scores = evaluate(model, to_layer, from_layer, X_eval, decision=decision)
        graded = graded_baseline(model, to_layer, from_layer, X_eval, family="all",  # same family as the IIA below
                                 decision=decision)
        # Control: the same network and the same decision distribution, but random orthonormal directions, with only
        # the thresholds and the table fitted. It asks whether the fitted directions carry the result or whether any k
        # of them would do. A control built from a *different* network cannot answer that, because its decision
        # distribution differs and a network that fails the task can still be validly abstracted.
        ctrl = max(evaluate(fit(to_layer, from_layer, X_fit, k, family=args.family, steps=args.steps,
                                n_interventions=args.n_interventions, freeze_directions=True, seed=100 + r,
                                decision=decision, n_classes=n_cls, verbose=False)[0],
                            to_layer, from_layer, X_eval, decision=decision)["all"]["skill"]
                   for r in range(args.control_draws))
        s = scores["all"]
        cols = " ".join(f"{f} {v['iia']:.3f}" for f, v in scores.items())
        print(f"  k = {k}  IIA[{cols}]  purity {s['purity']:.3f}  graded {graded:.3f}  "
              f"margin {s['margin_min']:.3f}/{s['margin_mean']:.3f} sharp {s['sharp']:.1f}  "
              f"cells {s['cells']}/{s['cells_total']} unseen {s['unseen']:.2f}  [{time.time() - t0:.0f}s]")
        print(f"         skill over a constant model ({s['base_rate']:.3f}): fitted T {s['skill']:+.3f}  |  "
              f"random T, best of {args.control_draws}: {ctrl:+.3f}")


if __name__ == "__main__":
    main()
