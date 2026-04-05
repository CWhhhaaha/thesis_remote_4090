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

    depth = len(blocks)
    lambdas = build_lambda_schedule(depth, cfg)
    regularizer_type = str(cfg.get("regularizer_type", "absolute"))
    epsilon = float(cfg.get("epsilon", 1e-12))
    rho_targets = build_rho_targets(depth, cfg) if regularizer_type == "target_ratio" else None
    stats: List[Dict[str, float]] = []
    for layer_idx, lambda_l in enumerate(lambdas, start=1):
        item = {
            "layer": layer_idx,
            "lambda": float(lambda_l),
            "regularizer_type": regularizer_type,
            "epsilon": epsilon,
        }
        if rho_targets is not None:
            item["rho_target"] = float(rho_targets[layer_idx - 1])
        stats.append(item)
    return stats


def build_rho_targets(depth: int, cfg: Dict) -> List[float]:
    targets = cfg.get("rho_targets")
    if targets is None:
        raise ValueError("Method3 target_ratio config must provide `rho_targets`.")
    rho_targets = [float(value) for value in targets]
    if len(rho_targets) != depth:
        raise ValueError(f"Expected {depth} rho targets, got {len(rho_targets)}")
    return rho_targets


def _layer_total_qk_matrix(q_weight: torch.Tensor, k_weight: torch.Tensor) -> torch.Tensor:
    return q_weight @ k_weight.transpose(-1, -2)


def _layer_total_asymmetry_energy(w_qk: torch.Tensor) -> torch.Tensor:
    asym = 0.5 * (w_qk - w_qk.transpose(-1, -2))
    return torch.sum(asym * asym)


def _layer_total_qk_energy(w_qk: torch.Tensor) -> torch.Tensor:
    return torch.sum(w_qk * w_qk)


def _layer_total_ratio(q_weight: torch.Tensor, k_weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    w_qk = _layer_total_qk_matrix(q_weight, k_weight)
    asym_energy = _layer_total_asymmetry_energy(w_qk)
    total_energy = _layer_total_qk_energy(w_qk).clamp_min(epsilon)
    return asym_energy / total_energy


def _layer_regularizer_value(q_weight: torch.Tensor, k_weight: torch.Tensor, cfg: Dict) -> torch.Tensor:
    regularizer_type = str(cfg.get("regularizer_type", "absolute"))
    epsilon = float(cfg.get("epsilon", 1e-12))
    w_qk = _layer_total_qk_matrix(q_weight, k_weight)
    asym_energy = _layer_total_asymmetry_energy(w_qk)

    if regularizer_type == "absolute":
        return asym_energy
    if regularizer_type == "ratio":
        total_energy = _layer_total_qk_energy(w_qk).clamp_min(epsilon)
        return asym_energy / total_energy
    if regularizer_type == "target_ratio":
        raise RuntimeError("target_ratio requires the layer-specific rho target and should be handled by the caller.")

    raise ValueError(f"Unknown Method3 regularizer_type: {regularizer_type}")


def _layer_diagnostics(q_weight: torch.Tensor, k_weight: torch.Tensor, cfg: Dict) -> Dict[str, float]:
    epsilon = float(cfg.get("epsilon", 1e-12))
    w_qk = _layer_total_qk_matrix(q_weight, k_weight)
    asym_energy = _layer_total_asymmetry_energy(w_qk)
    total_energy = _layer_total_qk_energy(w_qk).clamp_min(epsilon)
    ratio = asym_energy / total_energy
    return {
        "asym_energy": float(asym_energy.detach().cpu().item()),
        "total_energy": float(total_energy.detach().cpu().item()),
        "ratio": float(ratio.detach().cpu().item()),
    }


def structural_asymmetry_regularization(
    model: nn.Module,
    cfg: Dict,
    return_details: bool = False,
):
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("Expected a timm VisionTransformer with model.blocks")

    depth = len(blocks)
    lambdas = build_lambda_schedule(depth, cfg)
    regularizer_type = str(cfg.get("regularizer_type", "absolute"))
    rho_targets = build_rho_targets(depth, cfg) if regularizer_type == "target_ratio" else None
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
        q_weight = qkv_weight[:embed_dim]
        k_weight = qkv_weight[embed_dim : 2 * embed_dim]
        if regularizer_type == "target_ratio":
            epsilon = float(cfg.get("epsilon", 1e-12))
            ratio = _layer_total_ratio(q_weight, k_weight, epsilon)
            diagnostics = _layer_diagnostics(q_weight, k_weight, cfg)
            rho_target = float(rho_targets[layer_idx - 1])
            layer_reg = (ratio - rho_target) ** 2
        else:
            layer_reg = _layer_regularizer_value(q_weight, k_weight, cfg)
            diagnostics = _layer_diagnostics(q_weight, k_weight, cfg) if return_details else None

        total_reg = total_reg + float(lambda_l) * layer_reg
        if return_details:
            details[f"reg_l{layer_idx}"] = float(layer_reg.detach().cpu().item())
            details[f"asym_energy_l{layer_idx}"] = diagnostics["asym_energy"]
            details[f"total_energy_l{layer_idx}"] = diagnostics["total_energy"]
            details[f"ratio_l{layer_idx}"] = diagnostics["ratio"]
            if rho_targets is not None:
                details[f"rho_target_l{layer_idx}"] = float(rho_targets[layer_idx - 1])

    if return_details:
        return total_reg, details
    return total_reg
