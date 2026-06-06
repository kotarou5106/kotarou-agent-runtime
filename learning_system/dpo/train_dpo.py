from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from learning_system.dpo.config import DPOTrainConfig
from learning_system.preference_data.io import read_jsonl
from learning_system.preference_data.schema import PreferenceSample


def validate_trl_dataset(path: Path) -> list[dict[str, str]]:
    records = read_jsonl(path)
    validated: list[dict[str, str]] = []
    for index, record in enumerate(records, start=1):
        try:
            prompt = str(record.get("prompt") or "").strip()
            chosen = str(record.get("chosen") or "").strip()
            rejected = str(record.get("rejected") or "").strip()
            if not prompt or not chosen or not rejected:
                raise ValueError("prompt, chosen and rejected are required")
            if chosen == rejected:
                raise ValueError("chosen and rejected must differ")
            validated.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        except Exception as exc:
            raise ValueError(f"invalid DPO record at line {index}: {exc}") from exc
    if not validated:
        raise ValueError(f"dataset is empty: {path}")
    return validated


def validate_full_dataset(path: Path) -> list[PreferenceSample]:
    records = read_jsonl(path)
    return [PreferenceSample.model_validate(record) for record in records]


def run_dpo_training(config: DPOTrainConfig) -> dict[str, Any]:
    dataset_path = Path(config.dataset_path)
    records = validate_trl_dataset(dataset_path)
    summary: dict[str, Any] = {
        "dry_run": bool(config.dry_run),
        "num_samples": len(records),
        "config": config.to_dict(),
    }
    if config.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    try:
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOTrainer
    except ImportError as exc:
        raise RuntimeError(
            "DPO training requires optional dependencies. Install requirements-dpo.txt "
            "or the dpo extra before running without --dry-run."
        ) from exc

    train_dataset = Dataset.from_list(records)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.model_name, trust_remote_code=True)
    ref_model = None
    peft_config = None
    if config.use_lora:
        try:
            from peft import LoraConfig
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules="all-linear",
            )
        except ImportError as exc:
            raise RuntimeError("use_lora=true requires peft. Install requirements-dpo.txt.") from exc
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
        )

    trainer = _build_trainer(
        DPOTrainer=DPOTrainer,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        peft_config=peft_config,
        config=config,
    )
    train_result = trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    summary["train_result"] = str(train_result)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _build_trainer(
    *,
    DPOTrainer,
    model,
    ref_model,
    tokenizer,
    train_dataset,
    peft_config,
    config: DPOTrainConfig,
):
    try:
        from trl import DPOConfig

        args = DPOConfig(
            output_dir=config.output_dir,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_epochs=config.num_train_epochs,
            beta=config.beta,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            logging_steps=1,
            save_strategy="epoch",
            report_to=[],
        )
        return DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )
    except Exception:
        from transformers import TrainingArguments

        args = TrainingArguments(
            output_dir=config.output_dir,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_epochs=config.num_train_epochs,
            logging_steps=1,
            save_strategy="epoch",
            report_to=[],
        )
        return DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=args,
            beta=config.beta,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            peft_config=peft_config,
        )


def parse_args(argv: list[str] | None = None) -> DPOTrainConfig:
    parser = argparse.ArgumentParser(description="Train a model with TRL DPOTrainer.")
    parser.add_argument("--model-name", default=DPOTrainConfig.model_name)
    parser.add_argument("--dataset-path", default=DPOTrainConfig.dataset_path)
    parser.add_argument("--output-dir", default=DPOTrainConfig.output_dir)
    parser.add_argument("--learning-rate", type=float, default=DPOTrainConfig.learning_rate)
    parser.add_argument("--batch-size", type=int, default=DPOTrainConfig.batch_size)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DPOTrainConfig.gradient_accumulation_steps,
    )
    parser.add_argument("--num-train-epochs", type=float, default=DPOTrainConfig.num_train_epochs)
    parser.add_argument("--beta", type=float, default=DPOTrainConfig.beta)
    parser.add_argument("--max-length", type=int, default=DPOTrainConfig.max_length)
    parser.add_argument("--max-prompt-length", type=int, default=DPOTrainConfig.max_prompt_length)
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return DPOTrainConfig(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        use_lora=args.use_lora,
        dry_run=args.dry_run,
    )


def main(argv: list[str] | None = None) -> int:
    run_dpo_training(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

