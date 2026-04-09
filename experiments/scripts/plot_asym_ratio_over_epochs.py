import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]


def load_metrics(run_dir: Path):
    path = run_dir / "metrics.csv"
    with path.open() as f:
        raw_rows = list(csv.DictReader(f))

    rows = []
    for row in raw_rows:
        try:
            parsed = {}
            for key, value in row.items():
                if key == "epoch":
                    parsed[key] = int(value)
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
        except (TypeError, ValueError):
            continue

    if not rows:
        raise ValueError(f"No valid numeric rows found in {path}")

    out = {}
    for key in rows[0].keys():
        if key == "epoch":
            out[key] = np.array([r[key] for r in rows], dtype=int)
        else:
            out[key] = np.array([r[key] for r in rows], dtype=float)
    return out


def infer_layers(metrics):
    layers = []
    for key in metrics.keys():
        if key.startswith("sym_l"):
            try:
                layers.append(int(key.split("sym_l", 1)[1]))
            except ValueError:
                continue
        elif key.startswith("asym_l"):
            try:
                layers.append(int(key.split("asym_l", 1)[1]))
            except ValueError:
                continue
    layers = sorted(set(layers))
    if not layers:
        raise ValueError("Could not infer any layerwise asymmetry/symmetry columns from metrics.csv")
    return layers


def get_asym_curve(metrics, layer_idx: int):
    asym_key = f"asym_l{layer_idx}"
    if asym_key in metrics:
        return metrics[asym_key]
    sym_key = f"sym_l{layer_idx}"
    if sym_key in metrics:
        return 1.0 - metrics[sym_key]
    raise KeyError(f"Missing both {asym_key} and {sym_key} in metrics")


def style():
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
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


def plot_run(ax, metrics, label, layers):
    epochs = metrics["epoch"]
    cmap = plt.get_cmap("tab10")
    if len(layers) > 10:
        cmap = plt.get_cmap("tab20")
    for idx, layer in enumerate(layers):
        ax.plot(
            epochs,
            get_asym_curve(metrics, layer),
            lw=1.8,
            color=cmap(idx % cmap.N),
            label=f"Layer {layer}",
        )
    ax.set_title(label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Asymmetry Ratio")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y")


def main():
    parser = argparse.ArgumentParser(description="Plot layerwise asymmetry ratio over training epochs for one or two runs.")
    parser.add_argument("--run-a", type=str, required=True, help="First run directory containing metrics.csv")
    parser.add_argument("--label-a", type=str, default="Run A")
    parser.add_argument("--run-b", type=str, default=None, help="Optional second run directory")
    parser.add_argument("--label-b", type=str, default="Run B")
    parser.add_argument("--fig-dir", type=str, default="outputs/figures")
    parser.add_argument("--filename", type=str, default="asym_ratio_over_epochs.png")
    args = parser.parse_args()

    style()

    run_a = Path(args.run_a)
    if not run_a.is_absolute():
        run_a = EXPERIMENTS_ROOT / run_a
    fig_dir = Path(args.fig_dir)
    if not fig_dir.is_absolute():
        fig_dir = EXPERIMENTS_ROOT / fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics_a = load_metrics(run_a)
    layers_a = infer_layers(metrics_a)

    if args.run_b is not None:
        run_b = Path(args.run_b)
        if not run_b.is_absolute():
            run_b = EXPERIMENTS_ROOT / run_b
        metrics_b = load_metrics(run_b)
        layers_b = infer_layers(metrics_b)
        layers = sorted(set(layers_a) | set(layers_b))

        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), sharey=True)
        plot_run(axes[0], metrics_a, args.label_a, layers)
        plot_run(axes[1], metrics_b, args.label_b, layers)
        axes[1].legend(frameon=False, loc="upper right", ncol=1)
        fig.suptitle("Layerwise Asymmetry Ratio Across Training", y=1.02, fontsize=12)
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.2))
        plot_run(ax, metrics_a, args.label_a, layers_a)
        ax.legend(frameon=False, loc="upper right", ncol=1)
        fig.tight_layout()

    out_path = fig_dir / args.filename
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
