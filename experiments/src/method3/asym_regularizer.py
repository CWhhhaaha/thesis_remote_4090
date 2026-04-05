from typing import Dict, List

import torch
import torch.nn as nn


def build_lambda_schedule(depth: int, cfg: Dict) -> List[float]:
    if "lambdas" in cfg:
        lambdas = [float(value) for value in cfg["lambdas"]]
        if len(lambdas) != depth:
            raise ValueError(f"Expected {depth} lambdas, got {len(lambdas)}")
        return lambdas

    lambda_max = float(cfg.get("lambda_max", 0.0))
    weights = cfg.get("lambda_weights")
    if weights is None:
        raise ValueError("Method3 config must provide either `lambdas` or (`lambda_max` and `lambda_weights`).")

    if len(weights) != depth:
        raise ValueError(f"Expected {depth} lambda weights, got {len(weights)}")
    return [lambda_max * float(weight) for weight in weights]


def describe_lambda_schedule(model: nn.Module, cfg: Dict) -> List[Dict[str, float]]:
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("Expected a timm VisionTransformer with model.blocks")

    lambdas = build_lambda_schedule(len(blocks), cfg)
    stats: List[Dict[str, float]] = []
    for layer_idx, lambda_l in enumerate(lambdas, start=1):
        stats.append({"layer": layer_idx, "lambda": float(lambda_l)})
    return stats


def _head_asymmetry_energy(q_head_rows: torch.Tensor, k_head_rows: torch.Tensor) -> torch.Tensor:
    qq_t = q_head_rows @ q_head_rows.transpose(-1, -2)
    kk_t = k_head_rows @ k_head_rows.transpose(-1, -2)
    qk_t = q_head_rows @ k_head_rows.transpose(-1, -2)

    term1 = torch.sum(qq_t * kk_t)
    term2 = torch.trace(qk_t @ qk_t)
    energy = 0.5 * (term1 - term2)
    return energy.clamp_min(0.0)


def structural_asymmetry_regularization(
    model: nn.Module,
    cfg: Dict,
    return_details: bool = False,
):
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("Expected a timm VisionTransformer with model.blocks")

    lambdas = build_lambda_schedule(len(blocks), cfg)
    first_param = next(model.parameters())
    total_reg = torch.zeros((), device=first_param.device, dtype=first_param.dtype)
    details: Dict[str, float] = {}

    for layer_idx, (block, lambda_l) in enumerate(zip(blocks, lambdas), start=1):
        if lambda_l == 0.0:
            if return_details:
                details[f"reg_l{layer_idx}"] = 0.0
            continue

        qkv_weight = block.attn.qkv.weight
        embed_dim = qkv_weight.shape[1]
        num_heads = block.attn.num_heads
        head_dim = embed_dim // num_heads

        q_weight = qkv_weight[:embed_dim]
        k_weight = qkv_weight[embed_dim : 2 * embed_dim]

        layer_reg = torch.zeros((), device=qkv_weight.device, dtype=qkv_weight.dtype)
        for head_idx in range(num_heads):
            start = head_idx * head_dim
            end = start + head_dim
            q_head_rows = q_weight[start:end]
            k_head_rows = k_weight[start:end]
            layer_reg = layer_reg + _head_asymmetry_energy(q_head_rows, k_head_rows)

        total_reg = total_reg + float(lambda_l) * layer_reg
        if return_details:
            details[f"reg_l{layer_idx}"] = float(layer_reg.detach().cpu().item())

    if return_details:
        return total_reg, details
    return total_reg
