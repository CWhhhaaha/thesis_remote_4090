#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from transformers import AutoConfig, AutoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a layerwise geometry table for a completed survey subset.")
    parser.add_argument("--inventory", type=str, required=True, help="Inventory JSON used for the subset.")
    parser.add_argument("--input-dir", type=str, required=True, help="Completed output directory for the subset.")
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Output path prefix, without extension; writes .csv and .md.",
    )
    return parser.parse_args()


def load_inventory(path: Path) -> Dict[str, Dict]:
    data = json.loads(path.read_text())
    rows = data["primary"] + data.get("extension", [])
    return {row["model_id"]: row for row in rows}


def infer_task_object(inv_row: Dict) -> str:
    value = inv_row.get("training_object")
    if value:
        return value
    category = inv_row.get("category", "")
    family = inv_row.get("family", "").lower()
    if "vision" in category:
        if "dino" in family or "jepa" in family:
            return "self_supervised_representation"
        return "supervised_classification"
    return "unknown"


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    inventory_path = Path(args.inventory)
    if not inventory_path.is_absolute():
        inventory_path = script_dir / inventory_path

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = script_dir / input_dir

    output_prefix = Path(args.output_prefix)
    if not output_prefix.is_absolute():
        output_prefix = script_dir / output_prefix
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    inventory = load_inventory(inventory_path)
    summary_rows = json.loads((input_dir / "inventory_summary.json").read_text())

    param_cache: Dict[str, int] = {}
    arch_cache: Dict[str, str] = {}
    for row in summary_rows:
        model_id = row["model_id"]
        local_dir = Path(row["local_model_dir"])
        config = AutoConfig.from_pretrained(str(local_dir), local_files_only=True, trust_remote_code=False)
        model = AutoModel.from_pretrained(str(local_dir), local_files_only=True, trust_remote_code=False)
        param_cache[model_id] = int(sum(p.numel() for p in model.parameters()))
        archs = getattr(config, "architectures", None)
        arch_cache[model_id] = archs[0] if archs else config.__class__.__name__
        del model

    layer_table: List[Dict] = []
    for row in summary_rows:
        model_id = row["model_id"]
        inv_row = inventory.get(model_id, {})
        model_slug = model_id.replace("/", "__")
        csv_path = input_dir / model_slug / "layers.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for layer in reader:
                layer_table.append(
                    {
                        "model_name": model_id,
                        "family": row.get("family", inv_row.get("family", "")),
                        "task_object": infer_task_object(inv_row),
                        "architecture": arch_cache[model_id],
                        "parameter_count": param_cache[model_id],
                        "stack_name": layer["stack_name"],
                        "layer_index": int(layer["layer_index"]),
                        "asym_ratio": float(layer["asym_ratio"]),
                        "uvcos": float(layer["uvcos"]),
                    }
                )

    layer_table.sort(key=lambda x: (x["task_object"], x["family"], x["model_name"], x["stack_name"], x["layer_index"]))

    csv_out = output_prefix.with_suffix(".csv")
    md_out = output_prefix.with_suffix(".md")
    fieldnames = [
        "model_name",
        "family",
        "task_object",
        "architecture",
        "parameter_count",
        "stack_name",
        "layer_index",
        "asym_ratio",
        "uvcos",
    ]

    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(layer_table)

    with md_out.open("w", encoding="utf-8") as f:
        f.write("| model_name | family | task_object | architecture | parameter_count | stack_name | layer_index | asym_ratio | uvcos |\n")
        f.write("|---|---|---|---|---:|---|---:|---:|---:|\n")
        for row in layer_table:
            f.write(
                f"| {row['model_name']} | {row['family']} | {row['task_object']} | "
                f"{row['architecture']} | {row['parameter_count']} | {row['stack_name']} | "
                f"{row['layer_index']} | {row['asym_ratio']:.6f} | {row['uvcos']:.6f} |\n"
            )

    print(csv_out)
    print(md_out)
    print(f"rows={len(layer_table)}")


if __name__ == "__main__":
    main()
