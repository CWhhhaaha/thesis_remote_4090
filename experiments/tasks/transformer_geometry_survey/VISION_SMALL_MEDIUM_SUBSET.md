# Vision Small/Medium Subset

This subset is designed as a practical first-pass vision track for the thesis.

Its purpose is to compare:

- supervised vision Transformers
- self-supervised vision Transformers

while staying in the small-to-medium model regime so the 4090 workflow remains
stable and interpretable.

## Why this subset exists

The larger `QK interaction` vision inventory is useful, but it mixes:

- small/base models
- large/huge models
- pure vision checkpoints
- vision-language checkpoints such as CLIP

For an overnight thesis run, we want a cleaner first-stage slice:

1. small/medium models only
2. pure vision models only
3. both supervised and self-supervised training objectives represented

## Included supervised models

- `google/vit-base-patch16-224`
- `google/vit-base-patch32-384`
- `facebook/deit-tiny-distilled-patch16-224`
- `facebook/deit-small-distilled-patch16-224`
- `facebook/deit-base-distilled-patch16-224`

These are classification-oriented checkpoints.

## Included self-supervised models

- `facebook/dino-vits16`
- `facebook/dino-vitb16`
- `facebook/dinov2-base`

These are representation-learning checkpoints rather than plain classification
models.

## Why not include these yet

- `google/vit-large-*` and `google/vit-huge-*`: heavier than needed for the
  first clean pass
- `openai/clip-*`: multimodal rather than pure vision
- `facebook/ijepa_vith14_1k`: useful extension, but not required for the first
  small/medium overnight run

## Intended thesis use

This subset supports a focused claim:

- query-key geometry differs not only across architectures, but also across
  supervised vs self-supervised vision training objectives
