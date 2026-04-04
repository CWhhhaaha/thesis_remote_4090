# Method2 on Top of B0

This note documents the current `method2` experiment as the cleanest possible comparison against the original `B0_standard` setup.

## Goal

The purpose of this experiment is to test a **forward structural prior** on top of the existing B0 Vision Transformer, while keeping the rest of the training pipeline as close as possible to the original baseline.

Concretely, the experiment asks:

- If we bias early layers toward more symmetric / grouping-style interaction,
- and bias later layers toward more asymmetric / contextual interaction,
- does optimization or final performance change relative to the original B0 run?

## What Is Preserved From B0

The following parts remain unchanged relative to the original B0 experiment:

- Same CIFAR-10 dataset and data pipeline
- Same ViT backbone size:
  - `embed_dim = 512`
  - `depth = 6`
  - `num_heads = 8`
  - `patch_size = 4`
- Same optimizer family (`AdamW`)
- Same learning-rate schedule shape (warmup + cosine)
- Same batch size
- Same mixup / cutmix / label smoothing settings
- Same EMA setting
- Same AMP setting
- Same multi-head attention implementation style
- Same factorized `W_q / W_k / W_v` parameterization

In other words, this experiment does **not** replace standard ViT attention with a new attention family. It keeps the original architecture and only intervenes in the **pre-softmax score geometry**.

## What Is Changed

The only conceptual change is inside the forward pass of attention.

For each layer and each head, the raw score matrix is still computed in the standard way:

\[
S_h = Q_h K_h^\top.
\]

Then we decompose it into symmetric and antisymmetric parts:

\[
S_h^{\mathrm{sym}} = \frac{1}{2}(S_h + S_h^\top), \qquad
S_h^{\mathrm{asym}} = \frac{1}{2}(S_h - S_h^\top),
\]

and reconstruct it using a layerwise coefficient \(\alpha_l\):

\[
\widetilde S_h = S_h^{\mathrm{sym}} + \alpha_l S_h^{\mathrm{asym}}.
\]

The same \(\alpha_l\) is shared by all heads in a given layer.

This means:

- the implementation is still **per-head**
- the structural prior is **per-layer**

So the method preserves the standard multi-head computation pattern, while avoiding hand-designed head-specific roles.

## Layerwise Schedule Used

The current `method2` B0 config uses a linear layerwise schedule:

- `alpha_start = 0.35`
- `alpha_end = 1.40`
- `alpha_power = 1.0`

With depth `6`, this gives the approximate values:

\[
[0.35,\ 0.56,\ 0.77,\ 0.98,\ 1.19,\ 1.40].
\]

Interpretation:

- early layers: weaker antisymmetric component, hence more symmetric / grouping-oriented
- late layers: stronger antisymmetric component, hence more directional / contextual

## Normalization and Stability

This experiment keeps the original attention scaling:

\[
\frac{1}{\sqrt{d_h}}
\]

because that factor compensates for the standard dot-product variance growth with head dimension.

In addition, the reconstruction introduces an extra scale factor:

\[
\gamma_l \approx \sqrt{\frac{1+\alpha_l^2}{2}}.
\]

This is the current approximation used in code (`gamma_mode: approx`).

So the reconstructed score is normalized before softmax, but we do **not** add extra clipping in this clean B0 comparison.

## What Was Removed to Keep the Comparison Clean

To keep this experiment as close as possible to B0, the following extra stabilization trick was intentionally disabled:

- `score_tanh_clip = 0.0`

This means the current version does **not** use additional `tanh` score squashing.

The intent is to make the comparison easier to interpret:

- any difference should come mainly from the structural reconstruction itself,
- not from an additional clipping mechanism.

## What This Method Is Not

This experiment is **not**:

- a direct-\(M\) bilinear parameterization experiment
- a memory-optimized blockwise attention implementation
- a head-specific structural prior with different \(\alpha_{l,h}\)
- a change to the value path \(W_v\)

It is only a **minimal forward-geometry intervention** on top of standard B0.

## Files

The current implementation is stored in:

- `src/method2/structural_attention.py`
- `scripts/train_cifar_vit_method2.py`
- `configs/method2/b0_forward_structural_prior.yaml`
- `run_method2_b0.sh`
- `run_method2_b0_bg.sh`

## Recommended Interpretation

This experiment should be read as:

> Keep the original B0 training setup intact, but bias each layer toward a different score geometry before softmax.

More specifically:

- early layers are nudged toward reciprocal / grouping interactions
- late layers are nudged toward contextual / directional interactions
- the model is still free to discover head specialization inside each layer

## Suggested Next Comparison

The most informative first comparison is:

- `B0_standard`
- `M2_B0_forward_structural_prior`

with attention to:

- early training loss
- validation accuracy trajectory
- final accuracy
- layerwise symmetry / weighted U-V alignment metrics

