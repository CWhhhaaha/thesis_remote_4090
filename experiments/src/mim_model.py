from __future__ import annotations

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


class PrefixMaskedAutoRegressiveMIM(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop_rate: float,
        attn_drop_rate: float,
        min_visible_patches: int = 4,
    ):
        super().__init__()
        encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            num_classes=0,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            proj_drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )
        self.patch_embed = encoder.patch_embed
        self.cls_token = encoder.cls_token
        self.pos_embed = encoder.pos_embed
        self.pos_drop = encoder.pos_drop
        self.blocks = encoder.blocks
        self.norm = encoder.norm

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = self.patch_embed.num_patches
        self.patch_dim = 3 * patch_size * patch_size
        self.min_visible_patches = min_visible_patches

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.recon_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.patch_dim),
        )
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        patches = images.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        return patches.view(images.size(0), -1, self.patch_dim)

    def _sample_prefix_mask(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        visible_counts = torch.randint(
            low=self.min_visible_patches,
            high=self.num_patches,
            size=(batch_size,),
            device=device,
        )
        patch_positions = torch.arange(self.num_patches, device=device).unsqueeze(0)
        visible_mask = patch_positions < visible_counts.unsqueeze(1)
        masked_mask = ~visible_mask
        return visible_mask, masked_mask

    def _fixed_eval_mask(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        visible_count = max(self.min_visible_patches, self.num_patches // 2)
        patch_positions = torch.arange(self.num_patches, device=device).unsqueeze(0)
        visible_mask = patch_positions < visible_count
        visible_mask = visible_mask.expand(batch_size, -1)
        masked_mask = ~visible_mask
        return visible_mask, masked_mask

    def forward(self, images: torch.Tensor, training: bool = True) -> dict[str, torch.Tensor]:
        patch_tokens = self.patch_embed(images)
        patch_targets = self.patchify(images)

        if training:
            visible_mask, masked_mask = self._sample_prefix_mask(images.size(0), images.device)
        else:
            visible_mask, masked_mask = self._fixed_eval_mask(images.size(0), images.device)

        mask_tokens = self.mask_token.expand(images.size(0), self.num_patches, -1)
        tokens = torch.where(visible_mask.unsqueeze(-1), patch_tokens, mask_tokens)

        cls_tokens = self.cls_token.expand(images.size(0), -1, -1)
        x = torch.cat((cls_tokens, tokens), dim=1)
        x = x + self.pos_embed[:, : x.size(1)]
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        patch_features = x[:, 1:]
        patch_pred = self.recon_head(patch_features)
        patch_loss = (patch_pred - patch_targets).pow(2).mean(dim=-1)
        masked_loss = (patch_loss * masked_mask.float()).sum() / masked_mask.float().sum().clamp_min(1.0)

        return {
            "loss": masked_loss,
            "patch_loss": patch_loss,
            "visible_ratio": visible_mask.float().mean(),
            "masked_ratio": masked_mask.float().mean(),
        }
