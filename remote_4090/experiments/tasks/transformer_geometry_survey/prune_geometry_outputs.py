#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove geometry outputs for models with too few analyzable attention sites."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Run output directory containing inventory_summary.json and per-model subdirectories.",
    )
    parser.add_argument(
        "--min-attention-sites",
        type=int,
        default=2,
        help="Keep only models with at least this many attention sites.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the directories that would be removed without deleting anything.",
    )
    return parser.parse_args()


def slugify_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")


def load_inventory_summary(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_inventory_summary_json(path: Path, rows: List[Dict]):
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def write_inventory_summary_csv(path: Path, rows: List[Dict]):
    fieldnames = [
        "model_id",
        "category",
        "family",
        "config_class",
        "num_attention_sites",
        "elapsed_sec",
        "skipped_existing",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    summary_json = output_dir / "inventory_summary.json"
    summary_csv = output_dir / "inventory_summary.csv"
    if not summary_json.exists():
        raise FileNotFoundError(f"Missing inventory summary: {summary_json}")

    rows = load_inventory_summary(summary_json)
    kept_rows: List[Dict] = []
    pruned_rows: List[Dict] = []
    for row in rows:
        if int(row.get("num_attention_sites", 0)) < args.min_attention_sites:
            pruned_rows.append(row)
        else:
            kept_rows.append(row)

    print(
        f"Keeping {len(kept_rows)} models and pruning {len(pruned_rows)} models "
        f"with num_attention_sites < {args.min_attention_sites}"
    )

    for row in pruned_rows:
        model_dir = output_dir / slugify_model_id(row["model_id"])
        if args.dry_run:
            print(f"[dry-run] would remove {model_dir}")
            continue
        if model_dir.exists():
            shutil.rmtree(model_dir)
            print(f"[removed] {model_dir}")
        else:
            print(f"[missing] {model_dir}")

    if args.dry_run:
        return

    write_inventory_summary_json(summary_json, kept_rows)
    if summary_csv.exists():
        write_inventory_summary_csv(summary_csv, kept_rows)
    print(f"Updated summaries in {output_dir}")


if __name__ == "__main__":
    main()
