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

Targeted replication subset:

- a dedicated vision-only subset aligned with the model families emphasized in
  `Dissecting Query-Key Interaction in Vision Transformers`
- used to compare our layerwise `W_{qk}` survey against the vision-only findings
  in that paper
- kept separate from the main 36-model cross-task inventory so that replication
  and cross-task survey do not get mixed

Family-and-scale subset:

- a dedicated text-heavy subset focused on comparing model families and scale
- motivated by the large family/scale tables reported in the
  `symmetry/directionality` paper
- used to answer a different question from the cross-task survey:
  not only `which tasks differ?`, but also `how much do family and model scale
  matter inside text transformers?`

## Deliverables

### 1. Model inventory
- `MODEL_INVENTORY.md`
- `model_inventory.json`
- `QK_INTERACTION_VISION_SUBSET.md`
- `qk_interaction_vision_inventory.json`

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

### Parallel vision-only replication track
In parallel with the cross-task survey, run the dedicated
`QK interaction` vision subset. This subset is useful for two reasons:

- it gives us a closer apples-to-apples comparison with the model families
  used in the `QK interaction` paper
- it provides code and checkpoint naming hints for common vision transformer
  families such as ViT, DINO, DeiT, CLIP, SimMIM, and I-JEPA

The dedicated files for this track are:

- `QK_INTERACTION_VISION_SUBSET.md`
- `qk_interaction_vision_inventory.json`
- `run_qk_interaction_vision_subset.sh`
- `run_qk_interaction_vision_cached_only.sh`

### Parallel family-and-scale text track
In parallel with the cross-task survey and the vision-only replication subset,
run a dedicated `family and scale` text subset.

This subset is useful because it lets us separate three different explanatory
axes:

1. cross-task differences
2. family-specific inductive biases
3. model-scale effects within the same family

The dedicated files for this track are:

- `FAMILY_SCALE_TEXT_SUBSET.md`
- `family_scale_text_inventory.json`
- `run_family_scale_text_subset.sh`
- `run_family_scale_text_cached_only.sh`

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
- reuse cache aggressively because remote access to Hugging Face may be
  intermittent

## Code Hints from Existing Open Repos

For the `QK interaction`-style vision subset, the most useful implementation
lesson is that many model families can be loaded with plain Hugging Face APIs
 using exact checkpoint ids. In particular:

- ViT, DeiT, DINO, and many CLIP variants use straightforward
  `AutoConfig.from_pretrained(...)` and `AutoModel.from_pretrained(...)` style
  loading
- I-JEPA now has direct Transformers support
- SimMIM is better treated as a special-case/manual-checkpoint family, because
  its official checkpoints are distributed through the official Microsoft repo
  rather than a single stable Hugging Face family we can rely on

For the `family and scale` text subset, the most useful implementation lesson
from the `attention-geometry-main` repository is that many relevant families are
already organized by notebook and checkpoint id:

- BERT and its compact variants
- ALBERT
- RoBERTa
- DeBERTa
- GPT / GPT-2 / DistilGPT2
- GPT-Neo / GPT-J
- Phi
- LLaMA and Mistral as optional later additions

This means we can start from a conservative, directly loadable subset and only
expand toward gated or more heterogeneous families after the analysis pipeline is
stable.

## Immediate Next Step
Run the pilot subset first and inspect:

- whether layer discovery works
- whether multimodal towers separate cleanly
- whether the resulting layerwise `rho` curves look reasonable
