import math
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


def _encoder_layers(model: nn.Module) -> Iterable[nn.Module]:
    backbone = getattr(model, "bert", None)
    if backbone is None or not hasattr(backbone, "encoder"):
        raise ValueError("Expected a BERT-style MLM model with model.bert.encoder.layer")
    return backbone.encoder.layer


def _layer_qk_weights(layer: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    self_attn = layer.attention.self
    return self_attn.query.weight, self_attn.key.weight


def _symmetry_energy_ratio(matrix: torch.Tensor) -> float:
    sym = 0.5 * (matrix + matrix.transpose(-1, -2))
    denom = torch.linalg.matrix_norm(matrix, ord="fro").pow(2).clamp_min(1e-12)
    num = torch.linalg.matrix_norm(sym, ord="fro").pow(2)
    return float((num / denom).item())


def _asymmetry_energy_ratio(matrix: torch.Tensor) -> float:
    asym = 0.5 * (matrix - matrix.transpose(-1, -2))
    denom = torch.linalg.matrix_norm(matrix, ord="fro").pow(2).clamp_min(1e-12)
    num = torch.linalg.matrix_norm(asym, ord="fro").pow(2)
    return float((num / denom).item())


def _weighted_uv_alignment(matrix: torch.Tensor) -> float:
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    v = vh.transpose(-1, -2)
    cos = torch.sum(u * v, dim=0).abs()
    weight_sum = s.sum().clamp_min(1e-12)
    return float(((s * cos).sum() / weight_sum).item())


def build_lambda_schedule(depth: int, cfg: Dict) -> List[float]:
    if "lambdas" in cfg:
        lambdas = [float(value) for value in cfg["lambdas"]]
        if len(lambdas) != depth:
            raise ValueError(f"Expected {depth} lambdas, got {len(lambdas)}")
        return lambdas

    lambda_value = float(cfg.get("lambda", 0.0))
    return [lambda_value for _ in range(depth)]


def describe_lambda_schedule(model: nn.Module, cfg: Dict) -> List[Dict[str, float]]:
    depth = len(list(_encoder_layers(model)))
    lambdas = build_lambda_schedule(depth, cfg)
    regularizer_type = str(cfg.get("regularizer_type", "ratio"))
    epsilon = float(cfg.get("epsilon", 1e-12))
    return [
        {
            "layer": layer_idx,
            "lambda": float(lambda_l),
            "regularizer_type": regularizer_type,
            "epsilon": epsilon,
        }
        for layer_idx, lambda_l in enumerate(lambdas, start=1)
    ]


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


def structural_asymmetry_regularization(
    model: nn.Module,
    cfg: Dict,
    return_details: bool = False,
):
    layers = list(_encoder_layers(model))
    lambdas = build_lambda_schedule(len(layers), cfg)
    regularizer_type = str(cfg.get("regularizer_type", "ratio"))
    epsilon = float(cfg.get("epsilon", 1e-12))

    if regularizer_type != "ratio":
        raise ValueError("This BERT Method3 path currently supports only `regularizer_type=ratio`.")

    first_param = next(model.parameters())
    total_reg = torch.zeros((), device=first_param.device, dtype=first_param.dtype)
    details: Dict[str, float] = {}

    for layer_idx, (layer, lambda_l) in enumerate(zip(layers, lambdas), start=1):
        if lambda_l == 0.0:
            if return_details:
                details[f"reg_l{layer_idx}"] = 0.0
            continue

        q_weight, k_weight = _layer_qk_weights(layer)
        layer_reg = _layer_total_ratio(q_weight, k_weight, epsilon)
        total_reg = total_reg + float(lambda_l) * layer_reg

        if return_details:
            w_qk = _layer_total_qk_matrix(q_weight, k_weight)
            details[f"reg_l{layer_idx}"] = float(layer_reg.detach().cpu().item())
            details[f"asym_ratio_l{layer_idx}"] = _asymmetry_energy_ratio(w_qk.detach())
            details[f"sym_ratio_l{layer_idx}"] = _symmetry_energy_ratio(w_qk.detach())

    if return_details:
        return total_reg, details
    return total_reg


def collect_bert_attention_metrics(model: nn.Module, include_uvcos: bool = True) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for layer_idx, layer in enumerate(_encoder_layers(model), start=1):
        q_weight, k_weight = _layer_qk_weights(layer)
        w_qk = _layer_total_qk_matrix(q_weight.detach(), k_weight.detach())
        metrics[f"sym_l{layer_idx}"] = _symmetry_energy_ratio(w_qk)
        metrics[f"asym_l{layer_idx}"] = _asymmetry_energy_ratio(w_qk)
        metrics[f"gap_l{layer_idx}"] = (
            torch.linalg.matrix_norm(q_weight.detach() - k_weight.detach(), ord="fro")
            / (
                torch.linalg.matrix_norm(q_weight.detach(), ord="fro")
                + torch.linalg.matrix_norm(k_weight.detach(), ord="fro")
                + 1e-12
            )
        ).item()
        metrics[f"qk_norm_l{layer_idx}"] = torch.linalg.matrix_norm(w_qk, ord="fro").item()
        if include_uvcos:
            metrics[f"uvcos_l{layer_idx}"] = _weighted_uv_alignment(w_qk)
    return metrics
