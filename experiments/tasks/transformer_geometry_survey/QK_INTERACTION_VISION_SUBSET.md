# QK Interaction Vision Subset

This subset mirrors the main vision families highlighted in
`Dissecting Query-Key Interaction in Vision Transformers`.

## Why keep this subset separate

- It gives us a cleaner replication-style comparison point for the vision-only
  claims in that paper.
- It provides practical checkpoint-loading hints for the exact model families
  they emphasize.
- It should remain separate from the larger 36-model cross-task survey so that
  we do not mix `paper-aligned vision replication` with `cross-task geometry
  atlas`.

## Direct Hugging Face checkpoints

These are the checkpoints we should be able to feed directly into our current
survey pipeline.

### ViT

- `google/vit-base-patch16-224`
- `google/vit-base-patch32-384`
- `google/vit-large-patch16-224`
- `google/vit-large-patch32-384`
- `google/vit-huge-patch14-224-in21k`

### DINO

- `facebook/dino-vits16`
- `facebook/dino-vitb16`

### CLIP

- `openai/clip-vit-base-patch16`
- `openai/clip-vit-base-patch32`
- `openai/clip-vit-large-patch14`

### DeiT distilled

- `facebook/deit-tiny-distilled-patch16-224`
- `facebook/deit-small-distilled-patch16-224`
- `facebook/deit-base-distilled-patch16-224`

### I-JEPA

- `facebook/ijepa_vith14_1k`

## Special-case/manual checkpoints

These belong in the conceptual subset, but should not be treated as generic
`AutoModel.from_pretrained` entries in the first pass.

- `SimMIM-vit-b16-pretrain`
- `SimMIM-vit-b16-finetune`

Official source:
- [Microsoft SimMIM repo](https://github.com/microsoft/SimMIM)

## Code-level loading hints

The most useful implementation hint from existing code is that many of these
families can be loaded with plain Hugging Face APIs and exact checkpoint ids:

- ViT
- DINO
- DeiT
- many CLIP variants
- I-JEPA

We can therefore reuse the same core discovery logic from
`analyze_qk_geometry.py` and only keep a short list of family-specific caveats:

1. For CLIP, report the vision stack separately from the text stack if we want
   paper-aligned comparisons.
2. For I-JEPA, prefer the official Transformers support rather than a custom
   wrapper.
3. For SimMIM, treat it as a manual/official-repo family first and only fold it
   into the automated sweep after the checkpoint path is verified.

## Useful public references

- [Dissecting Query-Key Interaction in Vision Transformers](https://arxiv.org/abs/2405.14880)
- [I-JEPA docs](https://huggingface.co/docs/transformers/model_doc/ijepa)
- [I-JEPA official repo](https://github.com/facebookresearch/ijepa)
- [SimMIM official repo](https://github.com/microsoft/SimMIM)
