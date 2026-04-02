import argparse
import csv
import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from src.data import build_cifar10_loaders
from src.init_schemes import apply_layerwise_structural_prior, collect_layerwise_attention_metrics
from src.mim_model import PrefixMaskedAutoRegressiveMIM


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 autoregressive MIM experiments.")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return EXPERIMENTS_ROOT / path


def save_json(path: Path, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_model(cfg: Dict) -> PrefixMaskedAutoRegressiveMIM:
    return PrefixMaskedAutoRegressiveMIM(
        img_size=cfg["model"]["img_size"],
        patch_size=cfg["model"]["patch_size"],
        embed_dim=cfg["model"]["embed_dim"],
        depth=cfg["model"]["depth"],
        num_heads=cfg["model"]["num_heads"],
        mlp_ratio=cfg["model"]["mlp_ratio"],
        qkv_bias=cfg["model"]["qkv_bias"],
        drop_rate=cfg["model"]["drop_rate"],
        attn_drop_rate=cfg["model"]["attn_drop_rate"],
        min_visible_patches=cfg["train"].get("min_visible_patches", 4),
    )


def maybe_apply_init_scheme(model, cfg: Dict):
    init_name = cfg["experiment"]["init"]
    if init_name == "standard":
        return []
    if init_name == "layerwise_prior":
        return apply_layerwise_structural_prior(
            model=model,
            lambdas=cfg["experiment"]["lambdas"],
            base_std=cfg["experiment"].get("base_std", 0.02),
        )
    raise ValueError(f"Unknown init scheme: {init_name}")


def evaluate(model, loader, device: torch.device):
    model.eval()
    loss_sum = 0.0
    visible_sum = 0.0
    total = 0
    with torch.no_grad():
        for images, _labels in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            outputs = model(images, training=False)
            batch_size = images.size(0)
            loss_sum += outputs["loss"].item() * batch_size
            visible_sum += outputs["visible_ratio"].item() * batch_size
            total += batch_size
    return {
        "val_loss": loss_sum / total,
        "val_visible_ratio": visible_sum / total,
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_dir = resolve_path(cfg["experiment"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", cfg)

    set_seed(cfg["experiment"]["seed"])
    device = resolve_device()
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    use_channels_last = bool(cfg["train"].get("channels_last", True)) and device.type in {"cuda", "mps"}
    non_blocking = device.type == "cuda"

    train_loader, val_loader = build_cifar10_loaders(
        data_dir=str(resolve_path(cfg["data"]["root"])),
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = create_model(cfg).to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    init_stats = maybe_apply_init_scheme(model, cfg)
    save_json(run_dir / "init_stats.json", {"layers": init_stats})

    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        betas=(cfg["train"]["beta1"], cfg["train"]["beta2"]),
        weight_decay=cfg["train"]["weight_decay"],
    )
    warmup = LinearLR(
        optimizer,
        start_factor=cfg["train"]["warmup_start_factor"],
        total_iters=cfg["train"]["warmup_epochs"],
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cfg["train"]["epochs"] - cfg["train"]["warmup_epochs"],
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[cfg["train"]["warmup_epochs"]],
    )

    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type in {"cuda", "mps"}
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_enabled)
    eval_every = int(cfg["train"].get("eval_every", 10))

    metrics_path = run_dir / "metrics.csv"
    fieldnames = [
        "epoch",
        "train_loss",
        "train_visible_ratio",
        "val_loss",
        "val_visible_ratio",
        "lr",
        "epoch_time_sec",
    ]
    sample_metrics = collect_layerwise_attention_metrics(model, include_uvcos=False)
    fieldnames.extend(sample_metrics.keys())
    best_val = float("inf")

    with open(metrics_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, cfg["train"]["epochs"] + 1):
            start = time.time()
            model.train()
            running_loss = 0.0
            running_visible = 0.0
            total = 0

            train_bar = tqdm(
                train_loader,
                desc=f"mim {epoch:03d}/{cfg['train']['epochs']:03d}",
                leave=False,
                dynamic_ncols=True,
                mininterval=0.5,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )
            for images, _labels in train_bar:
                images = images.to(device, non_blocking=non_blocking)
                if use_channels_last:
                    images = images.contiguous(memory_format=torch.channels_last)
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    outputs = model(images, training=True)
                    loss = outputs["loss"]

                if not torch.isfinite(loss):
                    failure_state = {
                        "epoch": epoch,
                        "step_in_epoch": total // images.size(0) + 1 if images.size(0) > 0 else None,
                        "loss": float(loss.detach().cpu().item()),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    save_json(run_dir / "failure_state.json", failure_state)
                    raise RuntimeError(
                        f"Non-finite loss detected at epoch {epoch}, "
                        f"step {failure_state['step_in_epoch']}."
                    )

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                running_visible += outputs["visible_ratio"].item() * batch_size
                total += batch_size
                train_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.5f}",
                    },
                    refresh=False,
                )

            scheduler.step()

            should_eval = (epoch % eval_every == 0) or (epoch == cfg["train"]["epochs"])
            if should_eval:
                eval_metrics = evaluate(model, val_loader, device)
            else:
                eval_metrics = {"val_loss": float("nan"), "val_visible_ratio": float("nan")}
            struct_metrics = collect_layerwise_attention_metrics(model, include_uvcos=False)
            epoch_time = time.time() - start

            row = {
                "epoch": epoch,
                "train_loss": running_loss / total,
                "train_visible_ratio": running_visible / total,
                "val_loss": eval_metrics["val_loss"],
                "val_visible_ratio": eval_metrics["val_visible_ratio"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time_sec": epoch_time,
            }
            row.update(struct_metrics)
            writer.writerow(row)
            f.flush()

            if should_eval and eval_metrics["val_loss"] < best_val:
                best_val = eval_metrics["val_loss"]
                torch.save(deepcopy(model.state_dict()), run_dir / "best_val_loss.pt")

            torch.save(model.state_dict(), run_dir / "last.pt")

            train_bar.close()
            tqdm.write(
                f"[mim epoch {epoch:03d}] "
                f"train_loss={row['train_loss']:.4f} "
                f"val_loss={row['val_loss']:.4f} "
                f"visible={row['train_visible_ratio']:.2f} "
                f"best_val={best_val:.4f} "
                f"lr={row['lr']:.5f} "
                f"time={epoch_time:.1f}s"
            )

    final_metrics = collect_layerwise_attention_metrics(model, include_uvcos=True)
    save_json(
        run_dir / "final_structure.json",
        {"best_val_loss": best_val, "final_layer_metrics": final_metrics},
    )


if __name__ == "__main__":
    main()
