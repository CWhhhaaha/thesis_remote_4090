import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

COLORS = {
    "B0": "#4D4D4D",
    "B2": "#E45756",
}


def load_metrics(run_dir: Path):
    path = run_dir / "metrics.csv"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    out = {}
    for key in rows[0].keys():
        if key == "epoch":
            out[key] = np.array([int(r[key]) for r in rows], dtype=int)
        else:
            out[key] = np.array([float(r[key]) for r in rows], dtype=float)
    return out


def load_json(run_dir: Path, filename: str):
    return json.loads((run_dir / filename).read_text())


def try_load_json(run_dir: Path, filename: str):
    path = run_dir / filename
    if not path.exists():
        return None
    return json.loads(path.read_text())


def style():
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.5,
        }
    )


def plot_main_figure(b0_dir: Path, b2_dir: Path, figs_dir: Path):
    style()
    b0 = load_metrics(b0_dir)
    b2 = load_metrics(b2_dir)
    b0_final = load_json(b0_dir, "final_structure.json")
    b2_final = load_json(b2_dir, "final_structure.json")

    epochs = b0["epoch"]
    layers = np.arange(1, 7)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax = axes[0, 0]
    ax.plot(epochs, b0["top1_acc"], color=COLORS["B0"], lw=2.2, label="Standard")
    ax.plot(epochs, b2["top1_acc"], color=COLORS["B2"], lw=2.2, label="Layerwise Prior")
    ax.set_title("Validation Top-1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Top-1 (%)")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[0, 1]
    ax.plot(epochs, b0["val_loss"], color=COLORS["B0"], lw=2.2, label="Standard")
    ax.plot(epochs, b2["val_loss"], color=COLORS["B2"], lw=2.2, label="Layerwise Prior")
    ax.set_title("Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy")
    ax.grid(True, axis="y")

    ax = axes[1, 0]
    b0_sym = [b0_final["final_layer_metrics"][f"sym_l{i}"] for i in layers]
    b2_sym = [b2_final["final_layer_metrics"][f"sym_l{i}"] for i in layers]
    ax.plot(layers, b0_sym, color=COLORS["B0"], lw=2.2, marker="o", ms=5, label="Standard")
    ax.plot(layers, b2_sym, color=COLORS["B2"], lw=2.2, marker="o", ms=5, label="Layerwise Prior")
    ax.set_title("Final Layerwise Symmetry")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Symmetry Ratio")
    ax.set_xticks(layers)
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, axis="y")

    ax = axes[1, 1]
    b0_uv = [b0_final["final_layer_metrics"][f"uvcos_l{i}"] for i in layers]
    b2_uv = [b2_final["final_layer_metrics"][f"uvcos_l{i}"] for i in layers]
    ax.plot(layers, b0_uv, color=COLORS["B0"], lw=2.2, marker="o", ms=5, label="Standard")
    ax.plot(layers, b2_uv, color=COLORS["B2"], lw=2.2, marker="o", ms=5, label="Layerwise Prior")
    ax.set_title("Final Weighted U-V Alignment")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Weighted Cosine")
    ax.set_xticks(layers)
    ax.set_ylim(0.1, 1.02)
    ax.grid(True, axis="y")

    fig.suptitle("CIFAR-10 ViT: Standard Initialization vs Layerwise Structural Prior", y=0.98, fontsize=12)
    fig.tight_layout()
    fig.savefig(figs_dir / "main_comparison.png", bbox_inches="tight")
    fig.savefig(figs_dir / "main_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_structure_figure(b2_dir: Path, figs_dir: Path):
    style()
    b2_init = try_load_json(b2_dir, "init_stats.json")
    b2_init = b2_init["layers"] if b2_init is not None else []
    b2_init_struct = try_load_json(b2_dir, "init_structure.json")
    b2_final = load_json(b2_dir, "final_structure.json")["final_layer_metrics"]
    layers = np.arange(1, 7)

    final_sym = [b2_final[f"sym_l{i}"] for i in layers]
    final_uv = [b2_final[f"uvcos_l{i}"] for i in layers]

    if b2_init_struct is not None:
        init_layer_metrics = b2_init_struct["initial_layer_metrics"]
        init_sym = [init_layer_metrics[f"sym_l{i}"] for i in layers]
        init_uv = [init_layer_metrics[f"uvcos_l{i}"] for i in layers]
    elif len(b2_init) == len(layers):
        init_sym = [x["symmetry_ratio"] for x in b2_init]
        init_uv = [x["weighted_uv_alignment"] for x in b2_init]
    else:
        init_sym = None
        init_uv = None

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

    ax = axes[0]
    if init_sym is not None:
        ax.plot(layers, init_sym, color="#4C78A8", lw=2.2, marker="o", ms=5, label="Initialization")
    ax.plot(layers, final_sym, color=COLORS["B2"], lw=2.2, marker="o", ms=5, label="Final")
    ax.set_title("B2 Symmetry Profile")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Symmetry Ratio")
    ax.set_xticks(layers)
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, axis="y")
    if init_sym is not None:
        ax.legend(frameon=False, loc="best")

    ax = axes[1]
    if init_uv is not None:
        ax.plot(layers, init_uv, color="#4C78A8", lw=2.2, marker="o", ms=5, label="Initialization")
    ax.plot(layers, final_uv, color=COLORS["B2"], lw=2.2, marker="o", ms=5, label="Final")
    ax.set_title("B2 U-V Alignment Profile")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Weighted Cosine")
    ax.set_xticks(layers)
    ax.set_ylim(0.1, 1.02)
    ax.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(figs_dir / "b2_structure_shift.png", bbox_inches="tight")
    fig.savefig(figs_dir / "b2_structure_shift.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_loss_curve(b0_dir: Path, b2_dir: Path, figs_dir: Path):
    style()
    b0 = load_metrics(b0_dir)
    b2 = load_metrics(b2_dir)
    epochs = b0["epoch"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(epochs, b0["train_loss"], color=COLORS["B0"], lw=2.2, label="Standard Train")
    ax.plot(epochs, b0["val_loss"], color=COLORS["B0"], lw=2.2, ls="--", label="Standard Val")
    ax.plot(epochs, b2["train_loss"], color=COLORS["B2"], lw=2.2, label="Layerwise Prior Train")
    ax.plot(epochs, b2["val_loss"], color=COLORS["B2"], lw=2.2, ls="--", label="Layerwise Prior Val")
    ax.set_title("Loss Curves Across Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(figs_dir / "loss_curves.png", bbox_inches="tight")
    fig.savefig(figs_dir / "loss_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot comparison figures for two completed runs.")
    parser.add_argument("--b0-run", type=str, required=True)
    parser.add_argument("--b2-run", type=str, required=True)
    parser.add_argument("--fig-dir", type=str, default="outputs/figures")
    args = parser.parse_args()

    b0_dir = Path(args.b0_run)
    b2_dir = Path(args.b2_run)
    if not b0_dir.is_absolute():
        b0_dir = EXPERIMENTS_ROOT / b0_dir
    if not b2_dir.is_absolute():
        b2_dir = EXPERIMENTS_ROOT / b2_dir
    figs_dir = Path(args.fig_dir)
    if not figs_dir.is_absolute():
        figs_dir = EXPERIMENTS_ROOT / figs_dir
    figs_dir.mkdir(parents=True, exist_ok=True)

    plot_main_figure(b0_dir, b2_dir, figs_dir)
    plot_structure_figure(b2_dir, figs_dir)
    plot_loss_curve(b0_dir, b2_dir, figs_dir)
    print("Saved figures to", figs_dir)


if __name__ == "__main__":
    main()
