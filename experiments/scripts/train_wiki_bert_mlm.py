import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import yaml
from datasets import Dataset, load_dataset, load_from_disk
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from src.language_modeling import (
    collect_bert_attention_metrics,
    describe_lambda_schedule,
    structural_asymmetry_regularization,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT-mini MLM experiments on Wikipedia.")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return EXPERIMENTS_ROOT / path


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _prepare_texts(texts: List[str], cfg: Dict) -> List[str]:
    if cfg["data"].get("paper_join_characters", False):
        return [" ".join(text) for text in texts]
    return texts


def build_train_dataset(cfg: Dict, tokenizer, run_dir: Path):
    tokenized_cache_dir = resolve_path(cfg["data"]["tokenized_cache_dir"])
    tokenized_cache_dir.mkdir(parents=True, exist_ok=True)

    if any(tokenized_cache_dir.iterdir()):
        dataset = load_from_disk(str(tokenized_cache_dir))
        save_json(run_dir / "dataset_info.json", {"tokenized_cache_dir": str(tokenized_cache_dir), "loaded_from_cache": True})
        return dataset

    dataset_kwargs = {}
    cache_dir = cfg["data"].get("cache_dir")
    if cache_dir:
        dataset_kwargs["cache_dir"] = str(resolve_path(cache_dir))

    raw = load_dataset(
        cfg["data"]["dataset_name"],
        cfg["data"]["dataset_config"],
        **dataset_kwargs,
    )
    train_split = raw["train"]

    max_length = int(cfg["model"]["max_length"])
    num_proc = int(cfg["data"].get("num_proc", 1))

    def tokenize_batch(batch):
        texts = _prepare_texts(batch["text"], cfg)
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    tokenized = train_split.map(
        tokenize_batch,
        batched=True,
        num_proc=num_proc,
        remove_columns=train_split.column_names,
        desc="Tokenizing Wikipedia train split",
    )
    tokenized.save_to_disk(str(tokenized_cache_dir))

    save_json(
        run_dir / "dataset_info.json",
        {
            "tokenized_cache_dir": str(tokenized_cache_dir),
            "loaded_from_cache": False,
            "dataset_name": cfg["data"]["dataset_name"],
            "dataset_config": cfg["data"]["dataset_config"],
            "paper_join_characters": bool(cfg["data"].get("paper_join_characters", False)),
            "max_length": max_length,
        },
    )
    return tokenized


class BertMlmMethod3Trainer(Trainer):
    def __init__(self, *args, method3_cfg: Dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.method3_cfg = method3_cfg
        self.latest_task_loss = 0.0
        self.latest_reg_loss = 0.0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        task_loss = outputs.loss
        reg_loss = task_loss.new_zeros(())
        if self.method3_cfg is not None:
            reg_loss = structural_asymmetry_regularization(model, self.method3_cfg)
        loss = task_loss + reg_loss
        self.latest_task_loss = float(task_loss.detach().cpu().item())
        self.latest_reg_loss = float(reg_loss.detach().cpu().item())
        if return_outputs:
            return loss, outputs
        return loss


class StructureLoggingCallback(TrainerCallback):
    def __init__(self, trainer: BertMlmMethod3Trainer, run_dir: Path):
        self.trainer = trainer
        self.run_dir = run_dir

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        save_json(
            self.run_dir / "init_structure.json",
            {"initial_layer_metrics": collect_bert_attention_metrics(self.trainer.model, include_uvcos=True)},
        )
        if self.trainer.method3_cfg is not None:
            save_json(
                self.run_dir / "method3_schedule.json",
                {"layers": describe_lambda_schedule(self.trainer.model, self.trainer.method3_cfg)},
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        logs["task_loss"] = self.trainer.latest_task_loss
        logs["reg_loss"] = self.trainer.latest_reg_loss

    def on_train_end(self, args, state, control, model=None, **kwargs):
        save_json(
            self.run_dir / "final_structure.json",
            {"final_layer_metrics": collect_bert_attention_metrics(self.trainer.model, include_uvcos=True)},
        )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_dir = resolve_path(cfg["experiment"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", cfg)

    set_seed(int(cfg["experiment"]["seed"]))
    model_config = AutoConfig.from_pretrained(cfg["model"]["config_name"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["tokenizer_name"], use_fast=True)
    if tokenizer.model_max_length > 10000:
        tokenizer.model_max_length = int(cfg["model"]["max_length"])

    train_dataset = build_train_dataset(cfg, tokenizer, run_dir)
    model = BertForMaskedLM(model_config)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=float(cfg["mlm"]["probability"]),
    )

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        overwrite_output_dir=True,
        max_steps=int(cfg["train"]["max_steps"]),
        per_device_train_batch_size=int(cfg["train"]["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["train"]["gradient_accumulation_steps"]),
        learning_rate=float(cfg["train"]["learning_rate"]),
        warmup_steps=int(cfg["train"]["warmup_steps"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        logging_strategy="steps",
        logging_steps=int(cfg["train"]["logging_steps"]),
        save_strategy="steps",
        save_steps=int(cfg["train"]["save_steps"]),
        save_total_limit=int(cfg["train"]["save_total_limit"]),
        dataloader_num_workers=int(cfg["train"].get("dataloader_num_workers", 4)),
        report_to="none",
        remove_unused_columns=False,
        evaluation_strategy="no",
        fp16=bool(cfg["train"].get("fp16", False)),
        bf16=bool(cfg["train"].get("bf16", False)),
        gradient_checkpointing=bool(cfg["train"].get("gradient_checkpointing", False)),
        seed=int(cfg["experiment"]["seed"]),
        data_seed=int(cfg["experiment"]["seed"]),
        disable_tqdm=bool(cfg["train"].get("disable_tqdm", False)),
    )

    method3_cfg = cfg.get("method3")
    trainer = BertMlmMethod3Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        method3_cfg=method3_cfg,
    )
    trainer.add_callback(StructureLoggingCallback(trainer, run_dir))
    trainer.train()
    trainer.save_model(str(run_dir / "final_model"))
    tokenizer.save_pretrained(str(run_dir / "final_model"))
    save_json(run_dir / "trainer_state_summary.json", trainer.state.log_history)


if __name__ == "__main__":
    main()
