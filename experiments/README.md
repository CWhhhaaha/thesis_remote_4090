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

Current `B2` lambda schedule:

- `(0.5, 0.75, 0.9, 1.0, 1.0, 1.2)`

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

## Notes

- Data, outputs, checkpoints, and archives are ignored by git.
- This workspace is intended to be pushed to GitHub as code-only.
