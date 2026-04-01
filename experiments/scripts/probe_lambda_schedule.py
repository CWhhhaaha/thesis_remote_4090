import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from timm.models.vision_transformer import VisionTransformer


def signed_weighted_uv_cosine(matrix: torch.Tensor) -> float:
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    v = vh.transpose(-1, -2)
    cos = torch.sum(u * v, dim=0)
    weight_sum = s.sum().clamp_min(1e-12)
    return ((s * cos).sum() / weight_sum).item()


def sample_normal(shape, std: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(shape, device=device, dtype=dtype) * std


def sample_skew(size: int, std: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    base = sample_normal((size, size), std=std, device=device, dtype=dtype)
    return (base - base.transpose(-1, -2)) / math.sqrt(2.0)


def build_model() -> VisionTransformer:
    return VisionTransformer(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=512,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
    )


def probe(lambdas, seeds, base_std):
    device = torch.device("cpu")
    all_values = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = build_model().to(device)
        layer_values = []
        for block, lambda_l in zip(model.blocks, lambdas):
            qkv = block.attn.qkv
            embed_dim = qkv.weight.shape[1]
            shared = sample_normal((embed_dim, embed_dim), std=base_std, device=device, dtype=qkv.weight.dtype)
            skew = sample_skew(embed_dim, std=base_std, device=device, dtype=qkv.weight.dtype)
            scale = math.sqrt(1.0 + lambda_l ** 2)
            q_weight = (shared + lambda_l * skew) / scale
            k_weight = (shared - lambda_l * skew) / scale
            w_qk = q_weight @ k_weight.transpose(-1, -2)
            layer_values.append(signed_weighted_uv_cosine(w_qk))
        all_values.append(layer_values)
    return np.array(all_values, dtype=float)


def plot(target, mean_values, std_values, out_path: Path):
    layers = np.arange(1, len(target) + 1)
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "grid.color": "#D9D9D9",
            "grid.alpha": 0.5,
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(layers, target, color="#4C78A8", lw=2.2, marker="o", ms=5, label="Target")
    ax.plot(layers, mean_values, color="#E45756", lw=2.2, marker="o", ms=5, label="Initialized")
    ax.fill_between(layers, mean_values - std_values, mean_values + std_values, color="#E45756", alpha=0.16)
    ax.axhline(0.0, color="#888888", lw=1.0, ls="--")
    ax.set_title("Initialization Signed Weighted U-V Cosine")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Weighted Cosine")
    ax.set_xticks(layers)
    ax.set_ylim(-0.4, 1.02)
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Probe signed weighted U-V cosine for a lambda schedule.")
    parser.add_argument("--lambdas", type=float, nargs=6, required=True)
    parser.add_argument("--targets", type=float, nargs=6, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--base-std", type=float, default=0.02)
    parser.add_argument("--out-prefix", type=str, default="figures/lambda_probe")
    args = parser.parse_args()

    values = probe(args.lambdas, args.seeds, args.base_std)
    mean_values = values.mean(axis=0)
    std_values = values.std(axis=0)

    out_prefix = Path(args.out_prefix)
    if not out_prefix.is_absolute():
        out_prefix = Path(__file__).resolve().parents[1] / out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    plot(np.array(args.targets), mean_values, std_values, out_prefix)
    summary = {
        "lambdas": args.lambdas,
        "targets": args.targets,
        "seeds": args.seeds,
        "mean_signed_weighted_uv_cosine": mean_values.tolist(),
        "std_signed_weighted_uv_cosine": std_values.tolist(),
    }
    summary_path = out_prefix.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
