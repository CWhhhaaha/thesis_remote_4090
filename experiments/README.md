# Remote 4090 Training Workspace

This copy is prepared for running the CIFAR-10 ViT initialization experiments on a CUDA machine such as an RTX 4090.

## What Is Different Here

- CUDA-friendly training script with `cudnn.benchmark=True`
- TF32 enabled on CUDA
- larger default batch size: `256`
- more dataloader workers: `8`
- full experiment length: `500 epochs`
- remote-specific output directories under `outputs/runs/*_cuda4090`

## Active Configs

- `B0`: `configs/current/b0_standard.yaml`
- `B2`: `configs/current/b2_layerwise_signedcos.yaml`
- `MIM B0 (lite)`: `configs/current/mim_b0_lite.yaml`
- `MIM B2 (lite)`: `configs/current/mim_b2_lite.yaml`
- `Wiki BERT-mini B0`: `configs/language_mlm/b0_wiki_bertmini_200k.yaml`
- `Wiki BERT-mini M3-ratio`: `configs/language_mlm/m3_ratio_wiki_bertmini_200k.yaml`

Current `B2` lambda schedule:

- `(0.5, 0.75, 0.9, 1.0, 1.0, 1.2)`

Current lightweight autoregressive MIM setup:

- task: prefix-masked patch reconstruction on CIFAR-10
- model: 4-layer ViT, `embed_dim=256`, `patch_size=4`
- runs:
  - `run_mim_b0.sh`
  - `run_mim_b2.sh`

## Quick Start

```bash
cd remote_4090/experiments
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_cifar10.py --data-dir data/cifar10
python scripts/train_cifar_vit.py --config configs/current/b0_standard.yaml
python scripts/train_cifar_vit.py --config configs/current/b2_layerwise_signedcos.yaml
```

## One-Command Runs

Foreground:

```bash
cd remote_4090/experiments
bash run_b0.sh
bash run_b2.sh
bash run_mim_b0.sh
bash run_mim_b2.sh
bash run_wiki_bertmini_b0.sh
bash run_wiki_bertmini_m3_ratio.sh
```

Background with logs:

```bash
cd remote_4090/experiments
bash run_b0_bg.sh
bash run_b2_bg.sh
bash run_mim_b0_bg.sh
bash run_mim_b2_bg.sh
```

Tail logs:

```bash
tail -f outputs/logs/<log_file>.log
```

## Notes

- Data, outputs, checkpoints, and archives are ignored by git.
- This workspace is intended to be pushed to GitHub as code-only.
- The Wikipedia BERT-mini experiments follow the symmetry/directionality paper setup:
  - model: `prajjwal1/bert-mini`
  - task: encoder-only masked language modeling
  - tokenizer: `google-bert/bert-base-uncased`
  - max sequence length: `512`
  - optimization: `200k` steps, batch `32 x grad_accum 8`, lr `5e-5`, warmup `200`
- The language configs intentionally do not pre-download data locally; the first run on the 4090 will fetch/tokenize Wikipedia and cache it under `data/`.
