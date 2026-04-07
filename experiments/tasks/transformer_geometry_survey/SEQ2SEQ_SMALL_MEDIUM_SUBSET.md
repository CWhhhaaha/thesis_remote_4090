# Seq2Seq Small/Medium Subset

This subset is designed as a compact encoder-decoder track for the thesis.

Its purpose is to measure query-key geometry in text-to-text and seq2seq
Transformers without immediately moving to larger multilingual checkpoints.

## Why this subset exists

The main inventory includes several seq2seq families, but a first thesis pass
should stay with small/medium checkpoints that are:

- widely used
- easy to load through plain Hugging Face APIs
- interpretable as encoder-decoder text models

## Included models

- `t5-base`
- `google/flan-t5-base`
- `facebook/bart-base`

These give us:

- a classic text-to-text family (`T5`)
- an instruction-tuned text-to-text family (`FLAN-T5`)
- a denoising encoder-decoder family (`BART`)

## Deliberately deferred for later

- `google/mt5-base`
- `facebook/m2m100_418M`
- `facebook/nllb-200-distilled-600M`

Those are useful extensions, but they add multilingual and larger-scale effects
 that are better treated after the small/medium baseline is stable.
