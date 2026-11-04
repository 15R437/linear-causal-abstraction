"""
Visualises exactly which EXIOBASE (GB) sectors GradCAP groups together on the
real Leontief cyclic graph, and the resulting high-level causal structure.

Requires data/A_low.npy and data/A_low_labels.pkl (sector/factor names in the
same row order as A_low), both produced by experiments/leontief.py::parse_data.
"""

import pickle
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from pathlib import Path

from src.algos.gradcap import gradcap
from src.utils.partition import partition_to_matrix, partition_from_matrix
from src.algos.T_fitter import fit_abstraction_eigen

# Validated 2-slot categorical palette (dataviz skill, references/palette.md)
COLOR_CORE = "#2a78d6"
COLOR_ISOLATED = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"


def load_data():
    A_low = np.load("data/A_low.npy")
    with open("data/A_low_labels.pkl", "rb") as f:
        labels = pickle.load(f)
    names = [s for (_, s) in labels["sectors"]] + labels["factors"]
    return A_low, names


def fit_gradcap(A_low, k, seed_note=""):
    partition = gradcap(A_low=A_low, k=k)
    T_hard = partition_to_matrix(partition)
    A_high_pred, T_pred, S_pred = fit_abstraction_eigen(A_low=A_low, partition=partition)
    return T_hard, T_pred, A_high_pred


def draw_bulleted_list(ax, x0, y0, width_chars, entries, fontsize, line_dy, color=TEXT_PRIMARY):
    """Draws a bulleted, word-wrapped list starting at (x0, y0) and returns the
    final y cursor. Wrapping is done in characters (not pixels) so column width
    stays predictable regardless of font metrics."""
    y = y0
    for name in entries:
        wrapped = textwrap.wrap(name, width=width_chars) or [""]
        ax.text(x0, y, f"• {wrapped[0]}", fontsize=fontsize, color=color, va="top")
        y -= line_dy
        for cont in wrapped[1:]:
            ax.text(x0, y, f"  {cont}", fontsize=fontsize, color=color, va="top")
            y -= line_dy
    return y


def draw_membership_panel(ax, names, core_idx, isolated_idx):
    ax.axis("off")
    ax.set_xlim(0, 1)

    ax.text(0.0, 0.99, f"Structurally isolated sectors  (n = {len(isolated_idx)})",
             fontsize=12, fontweight="bold", color=COLOR_ISOLATED, va="top")
    ax.text(0.0, 0.955, "zero rows AND zero columns in $A_{low}$ — no measured GB linkage in or out",
             fontsize=8.5, color=TEXT_SECONDARY, va="top", style="italic")

    iso_names = sorted(names[i] for i in isolated_idx)
    n = len(iso_names)
    col1, col2 = iso_names[: (n + 1) // 2], iso_names[(n + 1) // 2:]

    top = 0.905
    y1_end = draw_bulleted_list(ax, 0.0, top, width_chars=33, entries=col1, fontsize=7.8, line_dy=0.0225)
    y2_end = draw_bulleted_list(ax, 0.505, top, width_chars=33, entries=col2, fontsize=7.8, line_dy=0.0225)

    box_y = min(y1_end, y2_end) - 0.04
    ax.text(0.0, box_y, f"Connected core economy  (n = {len(core_idx)})",
             fontsize=12, fontweight="bold", color=COLOR_CORE, va="top")
    ax.text(0.0, box_y - 0.035,
             "densely interlinked; GradCAP does not further subdivide it at k = 6–20",
             fontsize=8.5, color=TEXT_SECONDARY, va="top", style="italic")

    core_names = sorted(names[i] for i in core_idx)
    sample = [core_names[i] for i in np.linspace(0, len(core_names) - 1, 8).astype(int)]
    y_end = draw_bulleted_list(ax, 0.0, box_y - 0.08, width_chars=72, entries=sample,
                                fontsize=8, line_dy=0.028)
    ax.text(0.0, y_end, f"• ... and {len(core_idx) - len(sample)} more",
             fontsize=8, color=TEXT_SECONDARY, va="top", style="italic")

    ax.set_ylim(y_end - 0.06, 1.02)


def draw_network_panel(ax, n_core, n_isolated, self_weight_core, self_weight_isolated, cross_pairs_checked):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    r_core = 0.15
    r_iso = r_core * np.sqrt(n_isolated / n_core)

    x_core, y_core = 0.32, 0.42
    x_iso, y_iso = 0.75, 0.42

    # no cross edge -- state it explicitly, drawn first so it sits behind the nodes
    ax.annotate("", xy=(x_iso - r_iso - 0.02, y_iso), xytext=(x_core + r_core + 0.02, y_core),
                arrowprops=dict(arrowstyle="-", linestyle=(0, (2, 3)), color=TEXT_SECONDARY, linewidth=1.2),
                zorder=1)
    ax.text((x_core + x_iso) / 2, y_core + 0.12,
             f"0 of {cross_pairs_checked:,} possible\ncross edges",
             fontsize=8.5, ha="center", color=TEXT_SECONDARY, style="italic")

    ax.add_patch(Circle((x_core, y_core), r_core, facecolor=COLOR_CORE, edgecolor="white", linewidth=1.5, zorder=3))
    ax.add_patch(Circle((x_iso, y_iso), r_iso, facecolor=COLOR_ISOLATED, edgecolor="white", linewidth=1.5, zorder=3))

    ax.text(x_core, y_core - r_core - 0.05, f"Core economy\nn = {n_core}",
             fontsize=10, ha="center", va="top", color=TEXT_PRIMARY, fontweight="bold")
    ax.text(x_iso, y_iso - r_iso - 0.05, f"Isolated sectors\nn = {n_isolated}",
             fontsize=10, ha="center", va="top", color=TEXT_PRIMARY, fontweight="bold")

    # self-loop on the core node
    loop = FancyArrowPatch((x_core - 0.035, y_core + r_core), (x_core + 0.035, y_core + r_core),
                            connectionstyle="arc3,rad=2.2", arrowstyle="-|>",
                            mutation_scale=14, linewidth=1.6, color=COLOR_CORE, zorder=2)
    ax.add_patch(loop)
    ax.text(x_core, y_core + r_core + 0.155, f"self-influence\n$\\beta$ = {self_weight_core:.3f}",
             fontsize=8.5, ha="center", color=TEXT_SECONDARY)


def main():
    A_low, names = load_data()
    p = A_low.shape[0]
    k = 8

    T_hard, T_pred, A_high_pred = fit_gradcap(A_low, k)
    idx = partition_from_matrix(T_hard)
    sizes = [len(m) for m in idx]
    active = [c for c, s in enumerate(sizes) if s > 0]
    assert len(active) == 2, f"Expected 2 active clusters, found {len(active)}: {sizes}"
    core_c, iso_c = (active[0], active[1]) if sizes[active[0]] > sizes[active[1]] else (active[1], active[0])
    core_idx, iso_idx = idx[core_c], idx[iso_c]

    # un-masked self-influence (get_A_high zeroes the diagonal by convention, so
    # recompute the same T.T @ (A_low @ S) product without the (1 - eye) mask)
    from src.utils.helper import get_S
    E_low = np.linalg.inv(np.eye(A_low.shape[0]) - A_low)
    S = get_S(T=T_pred, E_low=E_low)
    AS = A_low @ S
    A_high_full = T_pred.T @ AS
    self_core = A_high_full[core_c, core_c]
    self_iso = A_high_full[iso_c, iso_c]

    cross_pairs = len(core_idx) * len(iso_idx) * 2

    fig = plt.figure(figsize=(13, 9.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.05,
                           top=0.93, bottom=0.02, left=0.02, right=0.98)
    ax_text = fig.add_subplot(gs[0])
    ax_net = fig.add_subplot(gs[1])

    draw_membership_panel(ax_text, names, core_idx, iso_idx)
    draw_network_panel(ax_net, len(core_idx), len(iso_idx), self_core, self_iso, cross_pairs)

    fig.suptitle(f"GradCAP abstraction of the UK Leontief input–output graph  (p = {p}, k = {k})",
                 fontsize=13, fontweight="bold", y=0.995)

    out_dir = Path("results/leontief")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "gradcap_sector_groups.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "gradcap_sector_groups.pdf", bbox_inches="tight")
    print(f"Saved to {out_dir / 'gradcap_sector_groups.png'} and .pdf")
    print(f"core self-influence beta={self_core:.4f}, isolated self-influence beta={self_iso:.4f}")


if __name__ == "__main__":
    main()
