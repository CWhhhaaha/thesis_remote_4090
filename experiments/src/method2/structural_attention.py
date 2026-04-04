import math
from typing import Dict, List

import torch
import torch.nn as nn


class StructuredPriorAttention(nn.Module):
    """Wrap a timm ViT attention module and reconstruct pre-softmax scores."""

    def __init__(
        self,
        base_attn: nn.Module,
        alpha: float,
        gamma_mode: str = "approx",
        score_tanh_clip: float = 0.0,
    ):
        super().__init__()
        self.num_heads = base_attn.num_heads
        self.head_dim = getattr(base_attn, "head_dim", base_attn.qkv.weight.shape[1] // base_attn.num_heads)
        self.scale = getattr(base_attn, "scale", self.head_dim ** -0.5)

        # Reuse the original timm modules/parameters to stay architecture-compatible.
        self.qkv = base_attn.qkv
        self.q_norm = getattr(base_attn, "q_norm", nn.Identity())
        self.k_norm = getattr(base_attn, "k_norm", nn.Identity())
        self.attn_drop = base_attn.attn_drop
        self.proj = base_attn.proj
        self.proj_drop = base_attn.proj_drop

        self.alpha = float(alpha)
        self.gamma_mode = gamma_mode
        self.score_tanh_clip = float(score_tanh_clip)

        # Force the explicit score path because we modify logits before softmax.
        self.fused_attn = False

    def _compute_gamma(self, sym: torch.Tensor, asym: torch.Tensor):
        if self.gamma_mode == "none":
            return None
        if self.gamma_mode == "approx":
            value = math.sqrt((1.0 + self.alpha ** 2) / 2.0)
            return torch.as_tensor(value, device=sym.device, dtype=sym.dtype)
        if self.gamma_mode == "exact":
            sym_energy = sym.pow(2).sum(dim=(-2, -1), keepdim=True)
            asym_energy = asym.pow(2).sum(dim=(-2, -1), keepdim=True)
            denom = (sym_energy + asym_energy).clamp_min(1e-12)
            return ((sym_energy + (self.alpha ** 2) * asym_energy) / denom).sqrt()
        raise ValueError(f"Unknown gamma_mode: {self.gamma_mode}")

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        scores = (q * self.scale) @ k.transpose(-2, -1)
        scores_sym = 0.5 * (scores + scores.transpose(-2, -1))
        scores_asym = 0.5 * (scores - scores.transpose(-2, -1))
        scores = scores_sym + self.alpha * scores_asym

        gamma = self._compute_gamma(scores_sym, scores_asym)
        if gamma is not None:
            scores = scores / gamma

        if self.score_tanh_clip > 0:
            scores = self.score_tanh_clip * torch.tanh(scores / self.score_tanh_clip)

        if attn_mask is not None:
            scores = scores + attn_mask

        attn = scores.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def _build_alpha_schedule(depth: int, cfg: Dict) -> List[float]:
    if "alphas" in cfg:
        alphas = [float(value) for value in cfg["alphas"]]
        if len(alphas) != depth:
            raise ValueError(f"Expected {depth} alphas, got {len(alphas)}")
        return alphas

    alpha_start = float(cfg.get("alpha_start", 0.5))
    alpha_end = float(cfg.get("alpha_end", 1.5))
    alpha_power = float(cfg.get("alpha_power", 1.0))
    if depth == 1:
        return [alpha_end]

    alphas = []
    for idx in range(depth):
        t = idx / (depth - 1)
        t = t ** alpha_power
        alphas.append(alpha_start + (alpha_end - alpha_start) * t)
    return alphas


def apply_forward_structural_prior(model: nn.Module, cfg: Dict) -> List[Dict[str, float]]:
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("Expected a timm VisionTransformer with model.blocks")

    alphas = _build_alpha_schedule(len(blocks), cfg)
    gamma_mode = str(cfg.get("gamma_mode", "approx"))
    score_tanh_clip = float(cfg.get("score_tanh_clip", 0.0))

    stats: List[Dict[str, float]] = []
    for layer_idx, (block, alpha) in enumerate(zip(blocks, alphas), start=1):
        block.attn = StructuredPriorAttention(
            base_attn=block.attn,
            alpha=alpha,
            gamma_mode=gamma_mode,
            score_tanh_clip=score_tanh_clip,
        )
        stats.append(
            {
                "layer": layer_idx,
                "alpha": float(alpha),
                "gamma_mode": gamma_mode,
                "score_tanh_clip": score_tanh_clip,
            }
        )
    return stats
