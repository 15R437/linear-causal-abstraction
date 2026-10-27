"""The ten-class MNIST network's decision code: fit it, draw it, and read off what each region means.

The abstraction is one high-level variable with 16 values on a 9-dimensional block of the first hidden layer, with the
block learned by interchange-intervention training. Passing --weights-subspace instead freezes it at the row space of
the centred product of the tail's weight matrices: a control that works only because this tail is near-linear, which
is not general.

Produces figures/mnist_multi_9d_code.{png,pdf} and figures/mnist_multi_split_digit.{png,pdf}.

Usage: python multiclass_code.py [--values 16] [--steps 1500]
"""
import argparse
import collections
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mnist_abstraction import load_mnist, train_mlp
from src.algos.sigmoid_abstract import decision_subspace, evaluate, fit, split_sequential

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(HERE, "figures")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--values", type=int, default=16)
    parser.add_argument("--dims", type=int, default=9)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--pool", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights-subspace", action="store_true",
                        help="freeze T at the closed-form decision subspace instead of learning it (a control)")
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    X_tr, y_tr, X_te, _ = load_mnist("multi")
    mlp = train_mlp(X_tr[:1], y_tr[:1], 10, epochs=0, verbose=False)
    mlp.load_state_dict(torch.load(os.path.join(HERE, "..", "..", "data", "mnist_mlp_multi.pt")))
    mlp.eval()
    to_layer, from_layer = split_sequential(mlp, 2)
    X_fit, X_eval = X_tr[:args.pool], X_te[:args.pool]

    T0, spectrum = decision_subspace(to_layer, from_layer, X_fit, args.dims, method="weights")
    print(f"decision subspace from the weights: spectrum {np.round(spectrum[:11] / spectrum[0], 3).tolist()}")
    print(f"T is {'frozen at that subspace' if args.weights_subspace else 'learned by interchange training'}")
    model, _ = fit(to_layer, from_layer, X_fit, [(args.values, args.dims)], steps=args.steps,
                   init_T=T0 if args.weights_subspace else None,
                   freeze_directions=args.weights_subspace, seed=args.seed, verbose=False)
    if not args.weights_subspace:
        import numpy.linalg as la
        cos = la.svd(T0 @ model.T.detach().numpy(), compute_uv=False).mean()
        print(f"alignment of the learned subspace with the closed-form one: {cos:.3f}")
    scores = evaluate(model, to_layer, from_layer, X_eval)["all"]
    print(f"IIA(all) {scores['iia']:.3f}  purity {scores['purity']:.3f}  "
          f"{scores['cells']} of {args.values} regions realised\n")

    with torch.no_grad():
        He = to_layer(X_eval).detach()
        Z = (He @ model.T).numpy()
        vals = model.values(He).reshape(-1).numpy()
        dec = from_layer(He).argmax(-1).numpy()
        table = model.M.argmax(-1).numpy()
        A, _ = model.readout(0)
        gate = (He > 0).to(torch.get_default_dtype()).mean(0)
        W1 = mlp[0].weight.detach()
        # what raises a region's score, pushed back to pixels: the gradient of that region's affine form, averaged
        # over the pool using each unit's firing rate (a linearisation, exact only where the active set is fixed)
        maps = {v: (W1 * (gate * (model.T @ A[v])).unsqueeze(1)).sum(0).numpy() for v in np.unique(vals)}
    used = list(np.unique(vals))
    counts = {v: collections.Counter(dec[vals == v].tolist()) for v in used}

    print(f"  {'region':>7s} {'n':>6s} {'names':>7s}   composition of the network's decisions")
    for v in used:
        mix = " ".join(f"{d}:{c / (vals == v).sum():.0%}" for d, c in counts[v].most_common(3))
        print(f"  {v:>7d} {int((vals == v).sum()):>6d} {int(table[v]):>7d}   {mix}")

    # ---- the code and what each region responds to
    Zc = Z - Z.mean(0)
    P = Zc @ np.linalg.svd(Zc, full_matrices=False)[2][:2].T
    fig = plt.figure(figsize=(13, 5.4))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.5], wspace=0.2)
    ax = fig.add_subplot(grid[0, 0])
    cmap = plt.get_cmap("tab20")
    for i, v in enumerate(used):
        mask = vals == v
        ax.scatter(P[mask, 0], P[mask, 1], s=3, alpha=0.35, color=cmap(i % 20))
        ax.text(P[mask, 0].mean(), P[mask, 1].mean(), str(table[v]), fontsize=13, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="k", lw=0.6, alpha=0.85))
    ax.set_xlabel(f"PC 1 of the {args.dims}-D code")
    ax.set_ylabel("PC 2")
    ax.set_title(f"the {args.dims}-dimensional decision code, projected to 2-D\n"
                 "(colour = region, label = digit the region names)", fontsize=10)
    inner = grid[0, 1].subgridspec(3, 4, hspace=0.35, wspace=0.1)
    for i, v in enumerate(used[:12]):
        a = fig.add_subplot(inner[i // 4, i % 4])
        lim = np.abs(maps[v]).max()
        a.imshow(maps[v].reshape(28, 28), cmap="RdBu_r", vmin=-lim, vmax=lim)
        a.set_title(f"region {v} $\\to$ '{table[v]}'  (n={int((vals == v).sum())})", fontsize=8)
        a.axis("off")
    fig.suptitle("what raises each region's score: $\\nabla_x$ of that region's affine form "
                 "(red = more ink here favours this region)", fontsize=9)
    os.makedirs(FIGURES, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES, f"mnist_multi_{args.dims}d_code.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- any digit needing more than one region. argmax regions are convex, so a digit whose set in the code is
    #      not convex needs several of them; here that is digit 0, split by slant.
    by_digit = collections.defaultdict(list)
    for v in used:
        by_digit[counts[v].most_common(1)[0][0]].append(v)
    # only worth drawing when a digit genuinely occupies two populated regions; single-example slivers are
    # fragmentation, not structure
    split = [(d, vs) for d, vs in by_digit.items()
             if len([v for v in vs if (vals == v).sum() >= 20]) > 1]
    split = [(d, [v for v in vs if (vals == v).sum() >= 20]) for d, vs in split]
    print("\ndigits covered by more than one populated region:", split if split else "none")
    if not split:
        print("(no split-digit figure drawn)")
        return
    digit, regions = split[0]
    regions = sorted(regions, key=lambda v: (vals == v).sum(), reverse=True)
    n_show = 10
    fig, axes = plt.subplots(len(regions) + 1, n_show, figsize=(n_show, (len(regions) + 1) * 1.15))
    for r, v in enumerate(regions):
        idx = np.where(vals == v)[0][:n_show]
        for j in range(n_show):
            axes[r, j].axis("off")
            if j < len(idx):
                axes[r, j].imshow(X_eval[idx[j]].reshape(28, 28), cmap="gray_r")
        axes[r, 0].set_title(f"region {v}: n={int((vals == v).sum())}, "
                             f"{counts[v].most_common(1)[0][1] / (vals == v).sum():.0%} '{digit}'",
                             fontsize=8.5, loc="left")
    for j in range(n_show):
        axes[-1, j].axis("off")
    means = [X_eval[vals == v].mean(0).numpy() for v in regions]
    axes[-1, 0].imshow(means[0].reshape(28, 28), cmap="gray_r")
    axes[-1, 0].set_title("mean of big", fontsize=8)
    axes[-1, 1].imshow(means[1].reshape(28, 28), cmap="gray_r")
    axes[-1, 1].set_title("mean of small", fontsize=8)
    diff = means[1] - means[0]
    lim = np.abs(diff).max()
    axes[-1, 2].imshow(diff.reshape(28, 28), cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[-1, 2].set_title("small $-$ big", fontsize=8)
    fig.suptitle(f"digit '{digit}' occupies two convex regions of the code", fontsize=10)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES, f"mnist_multi_split_digit.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures to {FIGURES}")


if __name__ == "__main__":
    main()
