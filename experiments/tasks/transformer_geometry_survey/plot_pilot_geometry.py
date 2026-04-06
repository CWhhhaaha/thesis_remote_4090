#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot pilot geometry curves from layers.csv files.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="outputs/pilot_subset",
        help="Directory containing per-model subdirectories with layers.csv.",
    )
    parser.add_argument(
        "--fig-dir",
        type=str,
        default="outputs/pilot_subset_figures",
        help="Directory to save the generated figures.",
    )
    return parser.parse_args()


def load_layers_csv(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_by_stack(rows: List[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[row["stack_name"]].append(row)
    for stack_name, stack_rows in grouped.items():
        stack_rows.sort(key=lambda r: int(r["layer_index"]))
    return grouped


def display_name(model_dir_name: str, stack_name: str) -> str:
    model_name = model_dir_name.replace("__", "/")
    return f"{model_name} | {stack_name}"


def collect_series(input_dir: Path, metric: str) -> List[Dict]:
    series: List[Dict] = []
    for model_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        csv_path = model_dir / "layers.csv"
        if not csv_path.exists():
            continue
        rows = load_layers_csv(csv_path)
        for stack_name, stack_rows in group_by_stack(rows).items():
            xs = [int(row["layer_index"]) for row in stack_rows]
            ys = [float(row[metric]) for row in stack_rows]
            categories = {row["category"] for row in stack_rows}
            category = sorted(categories)[0] if categories else "unknown"
            series.append(
                {
                    "label": display_name(model_dir.name, stack_name),
                    "category": category,
                    "x": xs,
                    "y": ys,
                }
            )
    return series


def plot_metric(series: List[Dict], title: str, ylabel: str, out_path: Path):
    if not series:
        raise ValueError(f"No series available for {title}")

    plt.figure(figsize=(12, 7))
    for item in series:
        plt.plot(item["x"], item["y"], marker="o", linewidth=1.8, markersize=4, label=item["label"])

    plt.title(title)
    plt.xlabel("Layer Index")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = script_dir / input_dir

    fig_dir = Path(args.fig_dir)
    if not fig_dir.is_absolute():
        fig_dir = script_dir / fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    asym_series = collect_series(input_dir, "asym_ratio")
    uvcos_series = collect_series(input_dir, "uvcos")

    plot_metric(
        asym_series,
        title="Pilot Geometry Survey: Asymmetry Ratio vs Layer",
        ylabel="Asymmetry Ratio",
        out_path=fig_dir / "pilot_asym_ratio_vs_layer.png",
    )
    plot_metric(
        uvcos_series,
        title="Pilot Geometry Survey: Weighted UV Alignment vs Layer",
        ylabel="Weighted UV Alignment",
        out_path=fig_dir / "pilot_uvcos_vs_layer.png",
    )

    print(f"Saved figures to {fig_dir}")


if __name__ == "__main__":
    main()
