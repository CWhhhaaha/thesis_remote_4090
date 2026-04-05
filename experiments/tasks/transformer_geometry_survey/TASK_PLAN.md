# Transformer Geometry Survey Task Plan

## Goal
Build a reproducible survey pipeline that measures layerwise query-key geometry in influential open-weight Transformer checkpoints across multiple task families.

The main object of interest is:

\[
W_{qk,l} = W_{q,l} W_{k,l}^{\top},
\]

and the main layerwise geometry metrics are:

- symmetry ratio
- asymmetry ratio
- weighted left-right singular-vector alignment (`uvcos`)
- `qk` Frobenius norm
- `q/k` gap ratio

## Why This Task Matters
This survey can support two paper-level contributions:

1. empirical characterization of query-key geometry across tasks and architectures
2. data-informed design of target-ratio regularization for Method 3

## Scope
Primary analysis set:

- 36 influential checkpoints
- 6 task families
- 6 models per family

Task families:

1. vision representation / image classification
2. text encoder-only
3. decoder-only language modeling
4. seq2seq / translation / text-to-text
5. audio / speech
6. multimodal vision-language

Optional extension set:

- 8 to 12 larger or more specialized checkpoints
- used only after the primary set is stable

## Deliverables

### 1. Model inventory
- `MODEL_INVENTORY.md`
- `model_inventory.json`

### 2. Analysis code
- `analyze_qk_geometry.py`

### 3. Output artifacts
For each model:

- raw layer metrics JSON
- per-layer CSV
- stack summary JSON

For the full run:

- inventory summary CSV
- failure log JSON

## Suggested Execution Order

### Pilot subset first
Run 12 models before the full sweep:

- 2 vision
- 2 text encoder
- 2 decoder-only
- 2 seq2seq
- 2 audio
- 2 multimodal

Recommended pilot models:

- `google/vit-base-patch16-224`
- `facebook/dinov2-base`
- `google-bert/bert-base-uncased`
- `microsoft/deberta-v3-base`
- `openai-community/gpt2`
- `facebook/opt-350m`
- `t5-base`
- `facebook/bart-base`
- `facebook/wav2vec2-base`
- `openai/whisper-small`
- `openai/clip-vit-base-patch32`
- `dandelin/vilt-b32-mlm`

### Then scale up
After the pilot:

- run the full 36-model inventory
- inspect failures
- add extension models only if the main inventory is stable

## Design Choices

### Layerwise object
The survey should report the layer-total bilinear object:

\[
W_{qk,l} = W_{q,l} W_{k,l}^{\top}
\]

for each attention layer or stack.

### Multi-stack models
For multimodal and encoder-decoder models:

- keep separate stacks separate
- do not collapse all towers into one summary

Examples:

- CLIP: report vision and text stacks separately
- T5/BART: report encoder self-attention and decoder self-attention separately
- BLIP / ViLT / BridgeTower: report whichever self-attention stacks are discoverable

### Architecture heterogeneity
The code should support the most common parameterization patterns:

- separate `query/key`
- separate `q_proj/k_proj`
- fused `qkv`
- fused GPT-style `c_attn`

The code should skip unsupported models gracefully and continue the sweep.

## Hyperparameter-Relevant Outputs
This survey is meant to inform Method 3 target design. The most useful plots or tables later will be:

- mean layerwise asymmetry ratio by task family
- mean layerwise symmetry ratio by task family
- mean `uvcos` by task family
- within-family variance across checkpoints

These statistics can be used to propose more evidence-based:

- `rho_l^*`
- layerwise `lambda_l` strength ordering

## Practical Notes for 4090 Execution

- load one checkpoint at a time
- save outputs after every model
- keep a failure log so long runs are resumable
- prefer primary/base-size checkpoints before larger ones

## Immediate Next Step
Run the pilot subset first and inspect:

- whether layer discovery works
- whether multimodal towers separate cleanly
- whether the resulting layerwise `rho` curves look reasonable
