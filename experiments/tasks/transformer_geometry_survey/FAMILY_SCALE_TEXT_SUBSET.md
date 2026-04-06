# Family-and-Scale Text Subset

This subset is designed to complement the cross-task survey. Its purpose is to
study family effects and scale effects within text transformers, rather than
only task differences across modalities.

## Why this subset matters

This subset supports three stronger paper-level claims:

1. cross-task comparison
2. family and scale comparison
3. data-informed target design for Method 3

The key idea is that a target-ratio prior should not only be informed by task
family; it may also depend on the architectural family and model scale.

## Conservative first-pass subset

We intentionally start with checkpoints that are:

- public
- widely used
- easy to load from Hugging Face
- already reflected in the `attention-geometry-main` notebooks

## Encoder-oriented families

- `google/bert_uncased_L-2_H-128_A-2`
- `google/bert_uncased_L-4_H-256_A-4`
- `google/bert_uncased_L-8_H-512_A-8`
- `bert-base-uncased`
- `bert-large-uncased`
- `distilbert-base-uncased`
- `albert-base-v2`
- `albert-xxlarge-v2`
- `FacebookAI/roberta-base`
- `FacebookAI/roberta-large`
- `distilbert/distilroberta-base`
- `microsoft/deberta-base`

## Decoder-oriented families

- `openai-gpt`
- `gpt2`
- `gpt2-medium`
- `gpt2-xl`
- `distilbert/distilgpt2`
- `EleutherAI/gpt-neo-125m`
- `EleutherAI/gpt-neo-1.3B`
- `EleutherAI/gpt-neo-2.7B`
- `EleutherAI/gpt-j-6B`
- `microsoft/phi-1`
- `microsoft/phi-1_5`
- `microsoft/phi-2`

## Why not start with LLaMA/Mistral immediately

These models are important, but not ideal for the first stable pass:

- some are gated
- some are much larger
- some use grouped-query attention or other architectural details that make the
  first-pass analysis less clean

They are better treated as a later extension once the family-and-scale pipeline
is stable on the conservative subset.

## Source of these model choices

These families are not arbitrary. They align closely with the language-model
notebooks in the local `attention-geometry-main` repository, which already
contains notebook-level checkpoint choices for:

- BERT
- ALBERT
- RoBERTa
- DeBERTa
- GPT / GPT-2 / DistilGPT2
- GPT-Neo / GPT-J
- Phi
- LLaMA / Mistral as later-stage families

That makes this subset a good bridge between:

- the symmetry/directionality paper
- our cross-task survey
- and our Method 3 target-design pipeline
