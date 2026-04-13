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

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn as nn
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.vision_transformer import VisionTransformer
from timm.utils import ModelEmaV3, accuracy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from src.data import build_classification_loaders
from src.config_utils import load_yaml_config
from src.init_schemes import apply_layerwise_structural_prior, collect_layerwise_attention_metrics
from src.method3 import describe_lambda_schedule, structural_asymmetry_regularization


def parse_args():
    parser = argparse.ArgumentParser(description="Train vision classification experiments with structure-aware loss regularization.")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return EXPERIMENTS_ROOT / path


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_model(cfg: Dict) -> VisionTransformer:
    model = VisionTransformer(
        img_size=cfg["model"]["img_size"],
        patch_size=cfg["model"]["patch_size"],
        in_chans=3,
        num_classes=cfg["model"]["num_classes"],
        embed_dim=cfg["model"]["embed_dim"],
        depth=cfg["model"]["depth"],
        num_heads=cfg["model"]["num_heads"],
        mlp_ratio=cfg["model"]["mlp_ratio"],
        qkv_bias=cfg["model"]["qkv_bias"],
        drop_rate=cfg["model"]["drop_rate"],
        proj_drop_rate=cfg["model"]["drop_rate"],
        attn_drop_rate=cfg["model"]["attn_drop_rate"],
    )
    return model


def maybe_apply_init_scheme(model: nn.Module, cfg: Dict):
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


def build_criterion(cfg: Dict):
    if cfg["train"]["mixup_alpha"] > 0 or cfg["train"]["cutmix_alpha"] > 0:
        return SoftTargetCrossEntropy()
    if cfg["train"]["label_smoothing"] > 0:
        return LabelSmoothingCrossEntropy(smoothing=cfg["train"]["label_smoothing"])
    return nn.CrossEntropyLoss()


def evaluate(model: nn.Module, loader, criterion, device: torch.device):
    model.eval()
    loss_sum = 0.0
    total = 0
    correct = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            acc1 = accuracy(logits, labels, topk=(1,))[0]
            batch_size = images.size(0)
            loss_sum += loss.item() * batch_size
            correct += acc1.item() * batch_size / 100.0
            total += batch_size
    return {"val_loss": loss_sum / total, "top1_acc": 100.0 * correct / total}


def save_json(path: Path, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
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

    train_loader, val_loader = build_classification_loaders(
        dataset_name=str(cfg["data"].get("dataset", "cifar10")),
        data_dir=str(resolve_path(cfg["data"]["root"])),
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=device.type == "cuda",
        image_size=int(cfg["model"]["img_size"]),
    )

    model = create_model(cfg).to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    init_stats = maybe_apply_init_scheme(model, cfg)
    save_json(run_dir / "init_stats.json", {"layers": init_stats})
    save_json(
        run_dir / "init_structure.json",
        {"initial_layer_metrics": collect_layerwise_attention_metrics(model, include_uvcos=True)},
    )
    save_json(
        run_dir / "method3_schedule.json",
        {"layers": describe_lambda_schedule(model, cfg["method3"])},
    )

    use_mixup = cfg["train"]["mixup_alpha"] > 0 or cfg["train"]["cutmix_alpha"] > 0
    mixup_fn = (
        Mixup(
            mixup_alpha=cfg["train"]["mixup_alpha"],
            cutmix_alpha=cfg["train"]["cutmix_alpha"],
            label_smoothing=cfg["train"]["label_smoothing"],
            num_classes=cfg["model"]["num_classes"],
        )
        if use_mixup
        else None
    )
    criterion_train = build_criterion(cfg)
    criterion_eval = nn.CrossEntropyLoss()

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
    use_ema = bool(cfg["train"].get("use_ema", True))
    ema = ModelEmaV3(model, decay=cfg["train"]["ema_decay"]) if use_ema else None
    amp_enabled = bool(cfg["train"]["amp"]) and device.type in {"cuda", "mps"}
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and amp_enabled)
    eval_every = int(cfg["train"].get("eval_every", 1))

    metrics_path = run_dir / "metrics.csv"
    fieldnames = ["epoch", "train_loss", "train_task_loss", "train_reg_loss", "val_loss", "top1_acc", "lr", "epoch_time_sec"]
    sample_metrics = collect_layerwise_attention_metrics(model, include_uvcos=False)
    fieldnames.extend(sample_metrics.keys())
    best_acc = -1.0

    with open(metrics_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, cfg["train"]["epochs"] + 1):
            start = time.time()
            model.train()
            running_loss = 0.0
            running_task_loss = 0.0
            running_reg_loss = 0.0
            total = 0

            train_bar = tqdm(
                train_loader,
                desc=f"epoch {epoch:03d}/{cfg['train']['epochs']:03d}",
                leave=False,
                dynamic_ncols=True,
                mininterval=0.5,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )
            for step_idx, (images, labels) in enumerate(train_bar, start=1):
                images = images.to(device, non_blocking=non_blocking)
                labels = labels.to(device, non_blocking=non_blocking)
                if use_channels_last:
                    images = images.contiguous(memory_format=torch.channels_last)
                if mixup_fn is not None:
                    mixed_images, mixed_labels = mixup_fn(images, labels)
                else:
                    mixed_images, mixed_labels = images, labels
                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    logits = model(mixed_images)
                    task_loss = criterion_train(logits, mixed_labels)
                reg_loss = structural_asymmetry_regularization(model, cfg["method3"])
                loss = task_loss + reg_loss

                if not torch.isfinite(loss):
                    failure_state = {
                        "epoch": epoch,
                        "step_in_epoch": step_idx,
                        "loss": float(loss.detach().cpu().item()),
                        "task_loss": float(task_loss.detach().cpu().item()),
                        "reg_loss": float(reg_loss.detach().cpu().item()),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    save_json(run_dir / "failure_state.json", failure_state)
                    raise RuntimeError(
                        f"Non-finite loss detected at epoch {epoch}, step {step_idx}. "
                        "Training stopped to avoid corrupting the run."
                    )

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                if ema is not None:
                    ema.update(model)
                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                running_task_loss += task_loss.item() * batch_size
                running_reg_loss += reg_loss.item() * batch_size
                total += batch_size
                train_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "reg": f"{reg_loss.item():.5f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.5f}",
                    },
                    refresh=False,
                )

            scheduler.step()

            should_eval = (epoch % eval_every == 0) or (epoch == cfg["train"]["epochs"])
            eval_model = ema.module if ema is not None else model
            if should_eval:
                eval_metrics = evaluate(eval_model, val_loader, criterion_eval, device)
            else:
                eval_metrics = {"val_loss": float("nan"), "top1_acc": float("nan")}
            struct_metrics = collect_layerwise_attention_metrics(model, include_uvcos=False)
            epoch_time = time.time() - start

            row = {
                "epoch": epoch,
                "train_loss": running_loss / total,
                "train_task_loss": running_task_loss / total,
                "train_reg_loss": running_reg_loss / total,
                "val_loss": eval_metrics["val_loss"],
                "top1_acc": eval_metrics["top1_acc"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time_sec": epoch_time,
            }
            row.update(struct_metrics)
            writer.writerow(row)
            f.flush()

            if should_eval and eval_metrics["top1_acc"] > best_acc:
                best_acc = eval_metrics["top1_acc"]
                torch.save(deepcopy(eval_model.state_dict()), run_dir / "best_val_acc.pt")

            torch.save(model.state_dict(), run_dir / "last.pt")

            if device.type == "mps":
                torch.mps.empty_cache()

            train_bar.close()
            tqdm.write(
                f"[epoch {epoch:03d}] "
                f"train_loss={row['train_loss']:.4f} "
                f"task_loss={row['train_task_loss']:.4f} "
                f"reg_loss={row['train_reg_loss']:.5f} "
                f"val_loss={row['val_loss']:.4f} "
                f"top1={row['top1_acc']:.2f} "
                f"best_top1={best_acc:.2f} "
                f"lr={row['lr']:.5f} "
                f"time={epoch_time:.1f}s"
            )

    final_metrics = collect_layerwise_attention_metrics(model, include_uvcos=True)
    _, final_reg_details = structural_asymmetry_regularization(model, cfg["method3"], return_details=True)
    save_json(
        run_dir / "final_structure.json",
        {
            "best_top1_acc": best_acc,
            "final_layer_metrics": final_metrics,
            "final_regularization_terms": final_reg_details,
        },
    )


if __name__ == "__main__":
    main()
