#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel


QUERY_KEY_SUFFIXES = [
    ("query", "key"),
    ("q_proj", "k_proj"),
    ("q", "k"),
]

FUSED_QKV_SUFFIXES = [
    "qkv",
    "c_attn",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze layerwise WqWk^T geometry from open Transformer checkpoints.")
    parser.add_argument(
        "--inventory",
        type=str,
        default="model_inventory.json",
        help="Path to model inventory JSON.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="primary",
        choices=["primary", "extension", "all"],
        help="Which inventory split to analyze.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        help="Optional category filter, e.g. vision, text_encoder, decoder_lm, seq2seq, audio, multimodal.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit model ids to analyze.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of models after filtering. 0 means no limit.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/geometry_survey",
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for model loading and matrix computation.",
    )
    parser.add_argument(
        "--load-dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Checkpoint loading dtype.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use already-downloaded checkpoints.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trust_remote_code for exotic models if needed.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models whose output directory already contains summary.json.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="hf_cache",
        help="Local cache/snapshot directory used before loading models.",
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=3,
        help="How many times to retry snapshot download before giving up.",
    )
    parser.add_argument(
        "--retry-sleep-sec",
        type=float,
        default=3.0,
        help="Base sleep time between retries.",
    )
    return parser.parse_args()


def resolve_dtype(name: str):
    if name == "auto":
        return "auto"
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def slugify_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")


def load_inventory(path: Path, split: str) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        inventory = json.load(f)
    if split == "all":
        return inventory["primary"] + inventory.get("extension", [])
    return inventory[split]


def filter_inventory(entries: List[Dict], category: str, models: Optional[List[str]], limit: int) -> List[Dict]:
    out = entries
    if category != "all":
        out = [entry for entry in out if entry["category"] == category]
    if models:
        model_set = set(models)
        out = [entry for entry in out if entry["model_id"] in model_set]
    if limit > 0:
        out = out[:limit]
    return out


def model_cache_dir(cache_root: Path, model_id: str) -> Path:
    return cache_root / slugify_model_id(model_id)


def has_local_snapshot(cache_dir: Path) -> bool:
    return (cache_dir / "config.json").exists()


def is_supported_weight_module(module: nn.Module) -> bool:
    if isinstance(module, nn.Linear):
        return True
    return module.__class__.__name__ == "Conv1D" and hasattr(module, "weight")


def linearized_weight(module: nn.Module) -> torch.Tensor:
    weight = module.weight.detach()
    if isinstance(module, nn.Linear):
        return weight
    if module.__class__.__name__ == "Conv1D":
        return weight.transpose(0, 1)
    raise TypeError(f"Unsupported attention projection module type: {module.__class__.__name__}")


def extract_layer_index(path: str) -> int:
    for token in path.split("."):
        if token.isdigit():
            return int(token)
    return -1


def strip_numeric_tokens(path: str) -> str:
    tokens = [token for token in path.split(".") if not token.isdigit()]
    return ".".join(tokens)


def stack_name_from_prefix(prefix: str) -> str:
    return strip_numeric_tokens(prefix).strip(".")


def weighted_uv_alignment(matrix: torch.Tensor) -> float:
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    v = vh.transpose(-1, -2)
    cos = torch.sum(u * v, dim=0).abs()
    weight_sum = s.sum().clamp_min(1e-12)
    return ((s * cos).sum() / weight_sum).item()


def geometry_metrics(q_weight: torch.Tensor, k_weight: torch.Tensor) -> Dict[str, float]:
    q_weight = q_weight.float()
    k_weight = k_weight.float()
    w_qk = q_weight @ k_weight.transpose(-1, -2)
    sym = 0.5 * (w_qk + w_qk.transpose(-1, -2))
    asym = 0.5 * (w_qk - w_qk.transpose(-1, -2))
    q_norm = torch.linalg.matrix_norm(q_weight, ord="fro")
    k_norm = torch.linalg.matrix_norm(k_weight, ord="fro")
    qk_norm = torch.linalg.matrix_norm(w_qk, ord="fro")
    total_energy = qk_norm.pow(2).clamp_min(1e-12)
    sym_energy = torch.linalg.matrix_norm(sym, ord="fro").pow(2)
    asym_energy = torch.linalg.matrix_norm(asym, ord="fro").pow(2)
    gap = torch.linalg.matrix_norm(q_weight - k_weight, ord="fro") / (q_norm + k_norm + 1e-12)
    return {
        "sym_ratio": (sym_energy / total_energy).item(),
        "asym_ratio": (asym_energy / total_energy).item(),
        "uvcos": weighted_uv_alignment(w_qk),
        "qk_norm": qk_norm.item(),
        "q_norm": q_norm.item(),
        "k_norm": k_norm.item(),
        "gap_ratio": gap.item(),
        "hidden_dim_out": int(w_qk.shape[0]),
        "input_dim": int(q_weight.shape[1]),
    }


def collect_attention_pairs(model: nn.Module) -> List[Dict]:
    modules = {name: module for name, module in model.named_modules() if is_supported_weight_module(module)}
    pairs: List[Dict] = []
    used_names = set()

    for name, module in modules.items():
        for q_suffix, k_suffix in QUERY_KEY_SUFFIXES:
            if not name.endswith(q_suffix):
                continue
            prefix = name[: -len(q_suffix)]
            key_name = prefix + k_suffix
            if key_name not in modules:
                continue
            if name in used_names or key_name in used_names:
                continue
            q_weight = linearized_weight(module)
            k_weight = linearized_weight(modules[key_name])
            if q_weight.ndim != 2 or k_weight.ndim != 2:
                continue
            if q_weight.shape[1] != k_weight.shape[1]:
                continue
            if q_weight.shape[0] != k_weight.shape[0]:
                continue
            pairs.append(
                {
                    "kind": "separate_qk",
                    "prefix": prefix.rstrip("."),
                    "q_name": name,
                    "k_name": key_name,
                    "q_weight": q_weight,
                    "k_weight": k_weight,
                }
            )
            used_names.add(name)
            used_names.add(key_name)
            break

    for name, module in modules.items():
        if name in used_names:
            continue
        if not any(name.endswith(suffix) for suffix in FUSED_QKV_SUFFIXES):
            continue
        weight = linearized_weight(module)
        if weight.ndim != 2 or weight.shape[0] % 3 != 0:
            continue
        hidden = weight.shape[0] // 3
        q_weight = weight[:hidden]
        k_weight = weight[hidden : 2 * hidden]
        pairs.append(
            {
                "kind": "fused_qkv",
                "prefix": name.rstrip("."),
                "q_name": f"{name}[q]",
                "k_name": f"{name}[k]",
                "q_weight": q_weight,
                "k_weight": k_weight,
            }
        )
        used_names.add(name)

    pairs.sort(key=lambda item: (extract_layer_index(item["prefix"]), item["prefix"]))
    return pairs


def summarize_by_stack(layer_rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in layer_rows:
        grouped[row["stack_name"]].append(row)

    summary: Dict[str, Dict[str, float]] = {}
    for stack_name, rows in grouped.items():
        rows = sorted(rows, key=lambda item: item["layer_index"])
        summary[stack_name] = {
            "num_layers": len(rows),
            "mean_sym_ratio": float(sum(row["sym_ratio"] for row in rows) / len(rows)),
            "mean_asym_ratio": float(sum(row["asym_ratio"] for row in rows) / len(rows)),
            "mean_uvcos": float(sum(row["uvcos"] for row in rows) / len(rows)),
            "mean_gap_ratio": float(sum(row["gap_ratio"] for row in rows) / len(rows)),
            "mean_qk_norm": float(sum(row["qk_norm"] for row in rows) / len(rows)),
        }
    return summary


def maybe_download_snapshot(model_id: str, cache_dir: Path, args: argparse.Namespace) -> Path:
    if has_local_snapshot(cache_dir):
        return cache_dir
    if args.local_files_only:
        raise FileNotFoundError(f"No local snapshot for {model_id} in {cache_dir}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    download_kwargs = {
        "repo_id": model_id,
        "local_dir": str(cache_dir),
        "allow_patterns": [
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "*.safetensors",
            "*.safetensors.index.json",
            "*.bin",
            "*.bin.index.json",
            "*.py",
        ],
        "ignore_patterns": [
            "README.md",
            "LICENSE",
            ".gitattributes",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "vocab.json",
            "merges.txt",
            "spiece.model",
            "sentencepiece.bpe.model",
            "*.mlmodel",
            "*.mlpackage",
            "*.h5",
            "*.ot",
            "*.onnx",
            "*.onnx_data",
            "*.tflite",
            "*.msgpack",
            "*.gguf",
            "coreml/*",
            "onnx/*",
            "openvino/*",
            "flax/*",
            "tf_model/*",
        ],
    }

    last_error = None
    for attempt in range(1, args.download_retries + 1):
        try:
            snapshot_download(**download_kwargs)
            if has_local_snapshot(cache_dir):
                return cache_dir
        except Exception as exc:
            last_error = exc
            if attempt < args.download_retries:
                sleep_for = args.retry_sleep_sec * attempt
                print(f"[retry] download {model_id} failed on attempt {attempt}/{args.download_retries}: {exc}")
                print(f"[retry] sleeping {sleep_for:.1f}s before retry")
                time.sleep(sleep_for)
            else:
                break
    raise RuntimeError(f"Snapshot download failed for {model_id}: {last_error}")


def load_model(model_source: str, device: str, load_dtype: str, local_files_only: bool, trust_remote_code: bool):
    dtype = resolve_dtype(load_dtype)
    kwargs = {
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
        "use_safetensors": False,
    }
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype
    config = AutoConfig.from_pretrained(model_source, **kwargs)
    model = AutoModel.from_pretrained(model_source, config=config, **kwargs)
    model.eval()
    if device == "cuda":
        model = model.to("cuda")
    return model, config


def analyze_one_model(entry: Dict, args: argparse.Namespace, output_root: Path) -> Dict:
    model_id = entry["model_id"]
    model_slug = slugify_model_id(model_id)
    model_dir = output_root / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)
    summary_path = model_dir / "summary.json"
    if args.skip_existing and summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            cached = json.load(f)
        cached["skipped_existing"] = True
        return cached

    start = time.time()
    cache_root = Path(args.cache_dir)
    if not cache_root.is_absolute():
        cache_root = Path(__file__).resolve().parent / cache_root
    local_model_dir = maybe_download_snapshot(model_id, model_cache_dir(cache_root, model_id), args)
    model, config = load_model(
        model_source=str(local_model_dir),
        device=args.device,
        load_dtype=args.load_dtype,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )

    layer_rows: List[Dict] = []
    for pair in collect_attention_pairs(model):
        metrics = geometry_metrics(pair["q_weight"], pair["k_weight"])
        row = {
            "model_id": model_id,
            "category": entry["category"],
            "family": entry["family"],
            "kind": pair["kind"],
            "prefix": pair["prefix"],
            "stack_name": stack_name_from_prefix(pair["prefix"]),
            "layer_index": extract_layer_index(pair["prefix"]),
            **metrics,
        }
        layer_rows.append(row)

    stack_summary = summarize_by_stack(layer_rows)
    summary = {
        "model_id": model_id,
        "category": entry["category"],
        "family": entry["family"],
        "url": entry["url"],
        "local_model_dir": str(local_model_dir),
        "config_class": config.__class__.__name__,
        "architectures": getattr(config, "architectures", None),
        "num_attention_sites": len(layer_rows),
        "stack_summary": stack_summary,
        "elapsed_sec": round(time.time() - start, 3),
    }

    with (model_dir / "layers.json").open("w", encoding="utf-8") as f:
        json.dump(layer_rows, f, indent=2)
    with (model_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = model_dir / "layers.csv"
    if layer_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys()))
            writer.writeheader()
            writer.writerows(layer_rows)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def write_inventory_summary(path: Path, rows: List[Dict]):
    if not rows:
        return
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
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    script_dir = Path(__file__).resolve().parent
    inventory_path = Path(args.inventory)
    if not inventory_path.is_absolute():
        inventory_path = script_dir / inventory_path

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = script_dir / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    entries = load_inventory(inventory_path, args.split)
    entries = filter_inventory(entries, args.category, args.models, args.limit)

    summaries: List[Dict] = []
    failures: List[Dict] = []
    for entry in entries:
        model_id = entry["model_id"]
        try:
            print(f"[analyze] {model_id}")
            summary = analyze_one_model(entry, args, output_root)
            summaries.append(summary)
        except Exception as exc:
            failure = {
                "model_id": model_id,
                "category": entry["category"],
                "family": entry["family"],
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(f"[failed] {model_id}: {exc}")

    with (output_root / "inventory_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    write_inventory_summary(output_root / "inventory_summary.csv", summaries)
    with (output_root / "failures.json").open("w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    print(f"Analyzed {len(summaries)} models; failures: {len(failures)}")
    print(f"Outputs saved to {output_root}")


if __name__ == "__main__":
    main()
