import math
from typing import Dict, List

import torch
import torch.nn as nn


def _symmetry_energy_ratio(matrix: torch.Tensor) -> float:
    sym = 0.5 * (matrix + matrix.transpose(-1, -2))
    denom = torch.linalg.matrix_norm(matrix, ord="fro").pow(2).clamp_min(1e-12)
    num = torch.linalg.matrix_norm(sym, ord="fro").pow(2)
    return (num / denom).item()


def _weighted_uv_alignment(matrix: torch.Tensor) -> float:
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    v = vh.transpose(-1, -2)
    cos = torch.sum(u * v, dim=0).abs()
    weight_sum = s.sum().clamp_min(1e-12)
    return ((s * cos).sum() / weight_sum).item()


def _sample_normal(shape, std: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.randn(shape, device=device, dtype=dtype) * std


def _sample_skew_symmetric(size: int, std: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    base = _sample_normal((size, size), std=std, device=device, dtype=dtype)
    return (base - base.transpose(-1, -2)) / math.sqrt(2.0)


def apply_layerwise_structural_prior(
    model: nn.Module,
    lambdas: List[float],
    base_std: float = 0.02,
) -> List[Dict[str, float]]:
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("Expected a timm VisionTransformer with model.blocks")
    if len(blocks) != len(lambdas):
        raise ValueError(f"Expected {len(blocks)} lambdas, got {len(lambdas)}")

    stats = []
    for layer_idx, (block, lambda_l) in enumerate(zip(blocks, lambdas), start=1):
        qkv = block.attn.qkv
        embed_dim = qkv.weight.shape[1]
        if qkv.weight.shape[0] != 3 * embed_dim:
            raise ValueError("Expected fused qkv weight with shape [3d, d]")

        device = qkv.weight.device
        dtype = qkv.weight.dtype
        shared = _sample_normal((embed_dim, embed_dim), std=base_std, device=device, dtype=dtype)
        skew = _sample_skew_symmetric(embed_dim, std=base_std, device=device, dtype=dtype)
        scale = math.sqrt(1.0 + lambda_l ** 2)
        q_weight = (shared + lambda_l * skew) / scale
        k_weight = (shared - lambda_l * skew) / scale

        with torch.no_grad():
            qkv.weight[:embed_dim].copy_(q_weight)
            qkv.weight[embed_dim : 2 * embed_dim].copy_(k_weight)

            if qkv.bias is not None:
                qkv.bias[: 2 * embed_dim].zero_()

        w_qk = q_weight @ k_weight.transpose(-1, -2)
        stats.append(
            {
                "layer": layer_idx,
                "lambda": float(lambda_l),
                "symmetry_ratio": _symmetry_energy_ratio(w_qk),
                "weighted_uv_alignment": _weighted_uv_alignment(w_qk),
                "qk_gap_ratio": (
                    torch.linalg.matrix_norm(q_weight - k_weight, ord="fro")
                    / (
                        torch.linalg.matrix_norm(q_weight, ord="fro")
                        + torch.linalg.matrix_norm(k_weight, ord="fro")
                        + 1e-12
                    )
                ).item(),
            }
        )
    return stats


def collect_layerwise_attention_metrics(model: nn.Module, include_uvcos: bool = True) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for layer_idx, block in enumerate(model.blocks, start=1):
        qkv = block.attn.qkv.weight.detach()
        embed_dim = qkv.shape[1]
        q_weight = qkv[:embed_dim]
        k_weight = qkv[embed_dim : 2 * embed_dim]
        w_qk = q_weight @ k_weight.transpose(-1, -2)
        metrics[f"sym_l{layer_idx}"] = _symmetry_energy_ratio(w_qk)
        if include_uvcos:
            metrics[f"uvcos_l{layer_idx}"] = _weighted_uv_alignment(w_qk)
        metrics[f"gap_l{layer_idx}"] = (
            torch.linalg.matrix_norm(q_weight - k_weight, ord="fro")
            / (
                torch.linalg.matrix_norm(q_weight, ord="fro")
                + torch.linalg.matrix_norm(k_weight, ord="fro")
                + 1e-12
            )
        ).item()
        metrics[f"qk_norm_l{layer_idx}"] = torch.linalg.matrix_norm(w_qk, ord="fro").item()
    return metrics
