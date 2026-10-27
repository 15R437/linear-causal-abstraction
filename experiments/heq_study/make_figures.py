"""Figures for REPORT.md: the learned abstraction of a hierarchical-equality MLP, and the sweeps around it.

Every number drawn here is produced by this script, not copied from a log. Fits are cached in figures/cache.json
(scalars) and figures/cache_<tag>.npz (the projections a scatter needs), so reruns are instant; pass --refit to
discard the cache.

Usage: python make_figures.py [--refit] [--steps 2000]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.algos.sigmoid_abstract import evaluate, fit, graded_baseline, split_sequential
from hierarchical_equality import (best_matching, bit_recovery, dependence, get_net, joint_recovery,
                                   parse_specs, sample_batch)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
POOL = 8192

# (tag, pairs, dim, width, depth, cut, spec)
CONFIGS = [
    *[(f"d{d}_2x2", 2, d, 64, 2, 2, "2:2:1") for d in (2, 4, 8, 16)],
    *[(f"d{d}_1x4", 2, d, 64, 2, 2, "1:4:3") for d in (2, 4, 8, 16)],
    ("d4_1x2", 2, 4, 64, 2, 2, "1:2:1"),
    ("d4_2x2d2", 2, 4, 64, 2, 2, "2:2:2"),
    ("d8_2x2d2", 2, 8, 64, 2, 2, "2:2:2"),
    ("cut4_2x2", 2, 4, 64, 2, 4, "2:2:1"),
    ("cut4_1x2", 2, 4, 64, 2, 4, "1:2:1"),
    ("L3cut4_2x2", 2, 4, 64, 3, 4, "2:2:1"),
    ("p3_3x2", 3, 4, 96, 2, 2, "3:2:1"),
    ("p3_1x8", 3, 4, 96, 2, 2, "1:8:7"),
]

# colour per ground-truth bit pattern, and per predicted class
BITCOL = {(0, 0): "#4C72B0", (0, 1): "#DD8452", (1, 0): "#55A868", (1, 1): "#C44E52"}
CLSCOL = {0: "#E8EDF4", 1: "#FBEDE4"}


class Args:  # get_net only reads these
    retrain = False
    train_steps = 6000
    verbose = False


def fit_once(to_layer, from_layer, X_fit, spec, steps, seed):
    return fit(to_layer, from_layer, X_fit, spec, family="all", steps=steps, seed=seed, verbose=False)[0]


def run_one(tag, pairs, dim, width, depth, cut, spec_text, steps, seeds=3, controls=2):
    """Fit one configuration and return (scalars, arrays-for-plots).

    The fit is a single non-convex run and its quality varies a lot with the seed -- it sometimes puts every
    activation in one cell, which is a degenerate optimum of the same objective. So fit `seeds` of them and keep the
    one with the highest IIA. IIA is the fitting objective and uses no ground-truth information, so this is model
    selection inside the method; `bits` is then measured on the selected model and is not what selected it.
    """
    (label, spec), = parse_specs(spec_text)
    mlp = get_net(pairs, dim, width, depth, Args())
    to_layer, from_layer = split_sequential(mlp, cut)
    g = torch.Generator().manual_seed(1234)
    X_fit, y_fit, _ = sample_batch(POOL, pairs, dim, g)
    X_eval, y_eval, bits = sample_batch(POOL, pairs, dim, g)
    with torch.no_grad():
        acc = (mlp(X_eval).argmax(-1) == y_eval).double().mean().item()
        H_eval = to_layer(X_eval).detach()

    runs = []
    for seed in range(seeds):
        m = fit_once(to_layer, from_layer, X_fit, spec, steps, seed)
        runs.append((evaluate(m, to_layer, from_layer, X_eval, seed=1), m, seed))
    iia_seeds = [r[0]["all"]["iia"] for r in runs]
    scores, model, best_seed = max(runs, key=lambda r: r[0]["all"]["iia"])

    ctrl = max(evaluate(fit(to_layer, from_layer, X_fit, spec, family="all", steps=steps,
                            freeze_directions=True, seed=100 + r, verbose=False)[0],
                        to_layer, from_layer, X_eval, seed=1)["all"]["skill"] for r in range(controls))
    acc_bits = bit_recovery(model, H_eval, bits)
    with torch.no_grad():
        values = model.values(H_eval).numpy()           # (N, k) the high-level state
        table = model.M.argmax(-1).numpy()              # the mechanism, one class per cell
        # Per-variable score margin g_i = s_1 - s_0, in raw projection units. For a binary variable the value is 1
        # exactly where g_i > 0, whatever the sign of A_1 - A_0 and whatever d_i is, so the plane below is the right
        # picture for d_i > 1 too and carries no sign ambiguity.
        gm = np.stack([((sc[:, 1] - sc[:, 0]) / model.sharpness).numpy() for sc in model.scores(H_eval)], axis=1) \
            if all(m == 2 for m in model.sizes) else np.zeros((len(H_eval), model.k))
    s = scores["all"]
    scalars = dict(tag=tag, label=label, pairs=pairs, dim=dim, width=width, depth=depth, cut=cut,
                   net_acc=acc, D=model.D, k=model.k, sizes=model.sizes, best_seed=best_seed,
                   iia_seeds=iia_seeds, iia_seed_mean=float(np.mean(iia_seeds)),
                   iia_seed_sd=float(np.std(iia_seeds)),
                   iia_obs=scores["obs"]["iia"], iia_single=scores["single"]["iia"], iia_all=s["iia"],
                   skill=s["skill"], purity=s["purity"], base_rate=s["base_rate"], sharp=s["sharp"],
                   cells=s["cells"], cells_total=s["cells_total"],
                   graded=graded_baseline(model, to_layer, from_layer, X_eval, family="all"),
                   deps=dependence(model, H_eval), bits=best_matching(acc_bits),
                   joint=joint_recovery(model, H_eval, bits), ctrl_skill=ctrl)
    arrays = dict(gm=gm, values=values, bits=bits.numpy(), table=table, acc_bits=acc_bits,
                  cells=model.cells.numpy())
    return scalars, arrays


def gather(steps, refit, seeds=3):
    path = os.path.join(FIG, "cache.json")
    cache = {} if refit or not os.path.exists(path) else json.load(open(path))
    arrays = {}
    for tag, *cfg in CONFIGS:
        npz = os.path.join(FIG, f"cache_{tag}.npz")
        if tag in cache and os.path.exists(npz):
            arrays[tag] = dict(np.load(npz))
            continue
        print(f"fitting {tag} ...", flush=True)
        cache[tag], arrays[tag] = run_one(tag, *cfg, steps=steps, seeds=seeds)
        np.savez(npz, **arrays[tag])
        json.dump(cache, open(path, "w"), indent=1)
    return cache, arrays


# ----------------------------------------------------------------------------------------------- figure 1


def fig_abstraction(cache, arrays, tag="d4_2x2"):
    """The abstraction itself: the arrangement the two variables cut out, the mechanism it induces, and what the
    variables turn out to mean.

    The axes are the per-variable score margins g_i = s_1 - s_0, so variable i reads 1 exactly where g_i > 0 and the
    two boundaries are the axes themselves. For d_i = 1 this is an affine image of t_i^T h; for d_i > 1 it is the
    right projection to look at, which a plot of raw coordinates would not be.
    """
    s, a = cache[tag], arrays[tag]
    gm, bits, table = a["gm"], a["bits"], a["table"]
    fig = plt.figure(figsize=(13.2, 4.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1, 1], height_ratios=[1, 4.2], hspace=0.06, wspace=0.34)
    ax = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=ax)

    for b1, b2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        sel = (bits[:, 0] == b1) & (bits[:, 1] == b2)
        ax.scatter(gm[sel, 0], gm[sel, 1], s=3.0, alpha=0.45, c=BITCOL[(b1, b2)], linewidths=0,
                   label=f"$b_1={b1},\\ b_2={b2}$")
    lim = np.abs(gm[:, :2]).max(0) * 1.35
    # shade each cell of the arrangement by the class the mechanism assigns it; value 1 is exactly g_i > 0
    for v1, xr in [(0, (-lim[0], 0.0)), (1, (0.0, lim[0]))]:
        for v2, yr in [(0, (-lim[1], 0.0)), (1, (0.0, lim[1]))]:
            cls = table[v1 * s["sizes"][1] + v2]
            ax.fill_between(xr, yr[0], yr[1], color=CLSCOL[int(cls)], zorder=0)
            # anchor each label in the quadrant's OUTER corner, away from the origin where the data sits
            xo, yo = (xr[0] if v1 == 0 else xr[1]), (yr[0] if v2 == 0 else yr[1])
            ax.text(xo * 0.93 if v1 == 0 else xo + 0.07 * lim[0],
                    yo * 0.93 if v2 == 0 else yo - 0.07 * lim[1],
                    f"$\\tau{{=}}({v1},{v2})\\to y{{=}}{cls}$",
                    ha="left" if v1 == 0 else "right", va="bottom" if v2 == 0 else "top",
                    fontsize=8.5, color="#333333", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#BBBBBB", alpha=0.85, lw=0.6))
    ax.axvline(0, color="k", lw=1.3)
    ax.axhline(0, color="k", lw=1.3)
    ax.set_xlim(-lim[0], lim[0]); ax.set_ylim(-lim[1], lim[1])
    ax.set_xlabel("score margin $g_1$")
    ax.set_ylabel("score margin $g_2$")
    ax.legend(markerscale=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4,
              columnspacing=0.8, handletextpad=0.1, frameon=False)

    for b in (0, 1):
        axt.hist(gm[bits[:, 1] == b, 0], bins=70, alpha=0.65,
                 color=BITCOL[(0, 0)] if b == 0 else BITCOL[(1, 1)], label=f"$b_2={b}$")
    axt.axvline(0, color="k", lw=1.3)
    axt.set_yticks([]); axt.tick_params(labelbottom=False)
    axt.legend(fontsize=7.5, loc="upper right", ncol=2, frameon=False)
    for side in ("top", "right", "left"):
        axt.spines[side].set_visible(False)
    axt.set_title(f"(a) Learned plane, coloured by true bits  ({s['D']}/{s['width']} dims)",
                  fontsize=9.5, loc="left", pad=6)

    # (b) IIA by intervention family, against the constant model and a random subspace of the same size
    ax = fig.add_subplot(gs[:, 1])
    vals = [s["iia_obs"], s["iia_single"], s["iia_all"]]
    bars = ax.bar(range(3), vals, color=["#9CB4D1", "#6A8CB8", "#2F5C94"], width=0.62)
    ax.bar_label(bars, fmt="%.3f", fontsize=8.5, padding=2)
    ax.axhline(s["base_rate"], color="#888888", ls="--", lw=1.1)
    ax.text(-0.42, s["base_rate"] - 0.012, "constant", fontsize=7.5, color="#666666", va="top", ha="left")
    ctrl_iia = s["base_rate"] + s["ctrl_skill"] * (1 - s["base_rate"])
    ax.axhline(ctrl_iia, color="#C44E52", ls=":", lw=1.3)
    ax.text(-0.42, ctrl_iia + 0.012, "random $T$", fontsize=7.5, color="#C44E52", va="bottom", ha="left")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["none", "one var", "any subset"], fontsize=9)
    ax.set_xlabel("intervened set $\\Theta$", fontsize=9)
    ax.set_ylim(0, 1.09); ax.set_ylabel("IIA")
    ax.set_title("(b) IIA by intervention family", fontsize=9.5, loc="left")

    # (c) what the variables mean
    ax = fig.add_subplot(gs[:, 2])
    acc = a["acc_bits"]
    im = ax.imshow(acc, cmap="Blues", vmin=0.5, vmax=1.0)
    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            ax.text(j, i, f"{acc[i, j]:.3f}", ha="center", va="center", fontsize=10,
                    color="white" if acc[i, j] > 0.78 else "#222222")
    ax.set_xticks(range(acc.shape[1]), [f"$b_{j + 1}$" for j in range(acc.shape[1])])
    ax.set_yticks(range(acc.shape[0]), [f"$\\tau_{i + 1}$" for i in range(acc.shape[0])])
    ax.set_xlabel("ground-truth bit"); ax.set_ylabel("learned variable")
    ax.set_title("(c) Bit recovery  (0.5 = chance)", fontsize=9.5, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Hierarchical equality, $P={s['pairs']}$ pairs, objects in "
                 f"$\\mathbb{{R}}^{{{s['dim']}}}$  ·  net {2 * s['pairs'] * s['dim']}-{s['width']}-"
                 f"{s['width']}-2, cut at $h_1$  ·  IIA(all) {s['iia_all']:.3f}", fontsize=10.5, y=1.0)
    out = os.path.join(FIG, "01_abstraction.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


# ----------------------------------------------------------------------------------------------- figure 2


def fig_sweeps(cache, arrays):
    """What the result depends on: the object dimension, and where the cut is taken."""
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    dims = [2, 4, 8, 16]

    # (a) object dimension
    ax = axes[0]
    for key, name, col, mk in [("2x2", "$k{=}2$ binary, $d_i{=}1$", "#2F5C94", "o"),
                               ("1x4", "$k{=}1$, four-valued, $d{=}3$", "#C44E52", "s")]:
        ax.plot(dims, [cache[f"d{d}_{key}"]["iia_all"] for d in dims], marker=mk, color=col, label=name, lw=1.8)
    ctrl = [cache[f"d{d}_2x2"]["base_rate"] + cache[f"d{d}_2x2"]["ctrl_skill"] *
            (1 - cache[f"d{d}_2x2"]["base_rate"]) for d in dims]
    ax.plot(dims, ctrl, color="#999999", ls=":", lw=1.5, marker="x", ms=5, label="random subspace ($k{=}2$)")
    ax.set_xscale("log", base=2); ax.set_xticks(dims, [str(d) for d in dims])
    ax.set_xlabel("object dimension $n$")
    ax.set_ylabel("IIA(all)")
    ax.set_ylim(0.45, 1.02); ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower center")
    ax.set_title("(a) Object dimension", fontsize=9.5, loc="left")

    # (b) where the cut is: IIA rises as the bits disappear
    ax = axes[1]
    # label each bar group by the architecture and the cut, since "depth 2/3" alone says nothing. The third group
    # is the control: cutting at h2 of a 3-hidden-layer net leaves a NONLINEAR tail, so the collapse cannot be
    # blamed on the tail being a single linear layer.
    tags = [("d4_2x2", "16-64-64-2\ncut at $h_1$"), ("cut4_2x2", "16-64-64-2\ncut at $h_2$\n(linear tail)"),
            ("L3cut4_2x2", "16-64-64-64-2\ncut at $h_2$\n(nonlinear tail)")]
    x = np.arange(len(tags))
    w = 0.27
    for off, key, col, name in [(-w, "iia", "#2F5C94", "IIA(all)"), (0, "bits", "#55A868", "bit recovery"),
                                (w, "dep", "#DD8452", "min. $\\mathrm{dep}(V_i)$")]:
        vals = [cache[t]["iia_all"] if key == "iia" else cache[t]["bits"] if key == "bits"
                else min(cache[t]["deps"]) for t, _ in tags]
        ax.bar_label(ax.bar(x + off, vals, w, color=col, label=name), fmt="%.2f", fontsize=7.5, padding=2)
    ax.axhline(0.5, color="#999999", ls=":", lw=1.2)
    ax.text(-0.46, 0.55, "chance", fontsize=7.5, color="#777777", va="bottom")
    ax.set_xticks(x, [n for _, n in tags], fontsize=8)
    ax.set_ylim(0, 1.38); ax.set_ylabel("score")
    ax.legend(fontsize=8, loc="upper center", ncol=3, framealpha=0.95, handlelength=1.1, columnspacing=0.8)
    ax.set_title("(b) Deeper cuts: IIA rises as the variables vanish", fontsize=9.5, loc="left")

    # (c) three pairs
    ax = axes[2]
    acc = arrays["p3_3x2"]["acc_bits"]
    im = ax.imshow(acc, cmap="Blues", vmin=0.5, vmax=1.0)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{acc[i, j]:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if acc[i, j] > 0.78 else "#222222")
    ax.set_xticks(range(3), [f"$b_{j + 1}$" for j in range(3)])
    ax.set_yticks(range(3), [f"$\\tau_{i + 1}$" for i in range(3)])
    ax.set_xlabel("ground-truth bit"); ax.set_ylabel("learned variable")
    s = cache["p3_3x2"]
    ax.set_title(f"(c) $P=3$: three variables, 3/96 dims", fontsize=9.5, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out = os.path.join(FIG, "02_sweeps.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


# ----------------------------------------------------------------------------------------------- figure 3


def fig_mechanism(cache, arrays):
    """The learned mechanism against the task's own function, at k = 2 and k = 3."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5), gridspec_kw=dict(width_ratios=[1, 2.1]))
    for ax, tag, title in [(axes[0], "d4_2x2", "$k=2$:  XNOR"), (axes[1], "p3_3x2", "$k=3$:  parity")]:
        s, a = cache[tag], arrays[tag]
        cells, table = a["cells"], a["table"]
        k = s["k"]
        truth = ((cells.sum(1) % 2) == 0).astype(int)
        # the learned variables are only identified up to relabelling each one's values, so compare against the
        # parity of the best bit assignment: flip each variable whose recovery says it is inverted
        best = None
        for flips in range(2 ** k):
            f = np.array([(flips >> i) & 1 for i in range(k)])
            t = (((cells ^ f).sum(1) % 2) == 0).astype(int)
            agree = (t == table).mean()
            if best is None or agree > best[0]:
                best = (agree, t)
        agree, truth = best
        grid = np.stack([table, truth])
        ax.imshow(grid, cmap="Blues", vmin=-0.4, vmax=1.5, aspect="auto")
        for j in range(len(table)):
            for i, row in enumerate(grid):
                ax.text(j, i, str(row[j]), ha="center", va="center", fontsize=10,
                        color="white" if row[j] == 1 else "#222222")
            if table[j] != truth[j]:
                ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, 2, fill=False, ec="#C44E52", lw=2.2))
        ax.set_xticks(range(len(table)), ["".join(str(v) for v in c) for c in cells], fontsize=8.5)
        ax.set_yticks([0, 1], ["learned $M$", "task's own\nfunction"], fontsize=9)
        ax.set_xlabel("high-level state $\\tau(h)$")
        ax.set_title(f"{title}   (agreement {agree:.3f})", fontsize=9.5, loc="left")
    fig.suptitle("The learned mechanism is the task's own function", fontsize=10.5)
    fig.tight_layout()
    out = os.path.join(FIG, "03_mechanism.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refit", action="store_true")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=3, help="fits per configuration; the best IIA is kept")
    args = parser.parse_args()
    # A short run produces untrained models, and with --refit it would overwrite the cached real fits with them.
    # The figures are built from whatever is in the cache, so that silently replaces the results with noise.
    if args.refit and (args.steps < 500 or args.seeds < 2):
        parser.error(f"--refit with steps={args.steps}, seeds={args.seeds} would overwrite the cache with an "
                     f"undertrained fit; use the defaults, or point FIG elsewhere to experiment")
    os.makedirs(FIG, exist_ok=True)
    cache, arrays = gather(args.steps, args.refit, args.seeds)
    fig_abstraction(cache, arrays)
    fig_sweeps(cache, arrays)
    fig_mechanism(cache, arrays)
    print("\nsummary")
    print(f"{'tag':12s} {'k:m:d':8s} {'cells':>7s} {'IIA(all)':>8s} {'purity':>7s} {'bits':>6s} {'joint':>6s} "
          f"{'dep(V)':>18s} {'ctrl':>7s}")
    for tag in cache:
        s = cache[tag]
        print(f"{tag:12s} {s['label']:8s} {str(s['cells']) + '/' + str(s['cells_total']):>7s} "
              f"{s['iia_all']:>8.3f} {s['purity']:>7.3f} {s['bits']:>6.3f} {s.get('joint', float('nan')):>6.3f} "
              f"{'[' + ' '.join(f'{v:.2f}' for v in s['deps']) + ']':>18s} {s['ctrl_skill']:>+7.3f}")
    print("\n bits  = per-variable recovery of a ground-truth bit (0.5 = chance); for k = 1 it is the best SINGLE bit")
    print(" joint = recovery of the whole bit vector from the joint high-level state -- comparable across specs")
    print(" dep(V)= share of realised cells whose class changes when that variable alone changes (0 = unused)")


if __name__ == "__main__":
    main()
