# Transformer Geometry Survey Inventory

This inventory is organized by task family. All links point to publicly accessible model-weight pages.

## 1. Vision Representation / Image Classification

| Model | Family | Weight page |
|---|---|---|
| `google/vit-base-patch16-224` | ViT | [link](https://huggingface.co/google/vit-base-patch16-224) |
| `facebook/deit-base-patch16-224` | DeiT | [link](https://huggingface.co/facebook/deit-base-patch16-224) |
| `facebook/vit-mae-base` | MAE | [link](https://huggingface.co/facebook/vit-mae-base) |
| `facebook/dinov2-base` | DINOv2 | [link](https://huggingface.co/facebook/dinov2-base) |
| `microsoft/beit-base-patch16-224-pt22k-ft22k` | BEiT | [link](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k) |
| `microsoft/swin-base-patch4-window7-224` | Swin | [link](https://huggingface.co/microsoft/swin-base-patch4-window7-224) |

## 2. Text Encoder-Only

| Model | Family | Weight page |
|---|---|---|
| `google-bert/bert-base-uncased` | BERT | [link](https://huggingface.co/google-bert/bert-base-uncased) |
| `FacebookAI/roberta-base` | RoBERTa | [link](https://huggingface.co/FacebookAI/roberta-base) |
| `microsoft/deberta-v3-base` | DeBERTa-v3 | [link](https://huggingface.co/microsoft/deberta-v3-base) |
| `google/electra-base-discriminator` | ELECTRA | [link](https://huggingface.co/google/electra-base-discriminator) |
| `microsoft/mpnet-base` | MPNet | [link](https://huggingface.co/microsoft/mpnet-base) |
| `FacebookAI/xlm-roberta-base` | XLM-R | [link](https://huggingface.co/FacebookAI/xlm-roberta-base) |

## 3. Decoder-Only Language Models

| Model | Family | Weight page |
|---|---|---|
| `openai-community/gpt2` | GPT-2 | [link](https://huggingface.co/openai-community/gpt2) |
| `openai-community/gpt2-medium` | GPT-2 | [link](https://huggingface.co/openai-community/gpt2-medium) |
| `facebook/opt-350m` | OPT | [link](https://huggingface.co/facebook/opt-350m) |
| `facebook/opt-1.3b` | OPT | [link](https://huggingface.co/facebook/opt-1.3b) |
| `bigscience/bloom-560m` | BLOOM | [link](https://huggingface.co/bigscience/bloom-560m) |
| `EleutherAI/pythia-410m` | Pythia | [link](https://huggingface.co/EleutherAI/pythia-410m) |

## 4. Seq2Seq / Translation / Text-to-Text

| Model | Family | Weight page |
|---|---|---|
| `t5-base` | T5 | [link](https://huggingface.co/t5-base) |
| `google/flan-t5-base` | FLAN-T5 | [link](https://huggingface.co/google/flan-t5-base) |
| `facebook/bart-base` | BART | [link](https://huggingface.co/facebook/bart-base) |
| `google/mt5-base` | mT5 | [link](https://huggingface.co/google/mt5-base) |
| `facebook/m2m100_418M` | M2M100 | [link](https://huggingface.co/facebook/m2m100_418M) |
| `facebook/nllb-200-distilled-600M` | NLLB | [link](https://huggingface.co/facebook/nllb-200-distilled-600M) |

## 5. Audio / Speech

| Model | Family | Weight page |
|---|---|---|
| `facebook/wav2vec2-base` | Wav2Vec2 | [link](https://huggingface.co/facebook/wav2vec2-base) |
| `facebook/hubert-base-ls960` | HuBERT | [link](https://huggingface.co/facebook/hubert-base-ls960) |
| `microsoft/wavlm-base` | WavLM | [link](https://huggingface.co/microsoft/wavlm-base) |
| `openai/whisper-small` | Whisper | [link](https://huggingface.co/openai/whisper-small) |
| `MIT/ast-finetuned-audioset-10-10-0.4593` | AST | [link](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) |
| `facebook/data2vec-audio-base-960h` | Data2Vec Audio | [link](https://huggingface.co/facebook/data2vec-audio-base-960h) |

## 6. Multimodal Vision-Language

| Model | Family | Weight page |
|---|---|---|
| `openai/clip-vit-base-patch32` | CLIP | [link](https://huggingface.co/openai/clip-vit-base-patch32) |
| `google/siglip-base-patch16-224` | SigLIP | [link](https://huggingface.co/google/siglip-base-patch16-224) |
| `BAAI/AltCLIP` | AltCLIP | [link](https://huggingface.co/BAAI/AltCLIP) |
| `dandelin/vilt-b32-mlm` | ViLT | [link](https://huggingface.co/dandelin/vilt-b32-mlm) |
| `BridgeTower/bridgetower-base` | BridgeTower | [link](https://huggingface.co/BridgeTower/bridgetower-base) |
| `Salesforce/blip-image-captioning-base` | BLIP | [link](https://huggingface.co/Salesforce/blip-image-captioning-base) |

## Optional Extension Set

These are useful later, but not necessary for the first reproducible sweep.

| Model | Family | Weight page |
|---|---|---|
| `meta-llama/Meta-Llama-3-8B` | Llama 3 | [link](https://huggingface.co/meta-llama/Meta-Llama-3-8B) |
| `mistralai/Mistral-7B-v0.1` | Mistral | [link](https://huggingface.co/mistralai/Mistral-7B-v0.1) |
| `Qwen/Qwen2.5-7B` | Qwen 2.5 | [link](https://huggingface.co/Qwen/Qwen2.5-7B) |
| `google/gemma-2-9b` | Gemma 2 | [link](https://huggingface.co/google/gemma-2-9b) |
| `Qwen/Qwen2-VL-2B-Instruct` | Qwen2-VL | [link](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) |
| `llava-hf/llava-1.5-7b-hf` | LLaVA | [link](https://huggingface.co/llava-hf/llava-1.5-7b-hf) |
| `microsoft/layoutlmv3-base` | LayoutLMv3 | [link](https://huggingface.co/microsoft/layoutlmv3-base) |
| `microsoft/trocr-base-printed` | TrOCR | [link](https://huggingface.co/microsoft/trocr-base-printed) |

## Suggested First Pilot

Run these first:

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
