"""Phase 4: LoRA SFT of Qwen3-8B on KFRD dataset (Appendix C.2, Eq 28)."""

from __future__ import annotations
import argparse
import inspect
import json
from loguru import logger
import yaml


def load_sft_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f).get("sft", {})


def resolve_precision(precision: str):
    import torch

    precision = precision.lower()
    if precision == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            precision = "bf16"
        elif torch.cuda.is_available():
            precision = "fp16"
        else:
            precision = "fp32"
    if precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError(f"Unsupported precision: {precision}")
    return precision, precision == "fp16", precision == "bf16"


def run_sft(
    model_id: str = "Qwen/Qwen3-8B-Instruct",
    dataset_path: str = "data/datasets/kfrd/kfrd_sft.json",
    output_dir: str = "data/models/kubellm",
    lora_rank: int = 64,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    num_epochs: int = 5,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.01,
    max_length: int = 1024,
    precision: str = "auto",
    load_in_4bit: bool = True,
    device_map: str = "auto",
    gradient_checkpointing: bool = True,
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    load_best_model_at_end: bool = True,
    logging_steps: int = 10,
    dataloader_pin_memory: bool = False,
    seed: int = 42,
) -> None:
    """
    SFT with LoRA: W = W_0 + ΔW = W_0 + A·B^T  (Eq 19/27)
    Loss: L_SFT = -E[(ρ,χ,ς)~D_final] [log p_θ(ς | ρ, χ)]  (Eq 28)
    """
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
        TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from datasets import Dataset

    if lora_alpha is None:
        lora_alpha = 2 * lora_rank
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    logger.info(f"[SFT] loading dataset from {dataset_path}")
    with open(dataset_path) as f:
        raw = json.load(f)

    def format_sample(s: dict) -> dict:
        response = (
            f"Root Cause:\n{s['reasoning']}\n\n"
            f"Resolution Commands:\n{s['final_solution']}\n\n"
            f"Prevention:\n{s.get('prevention', 'Monitor the workload and validate configuration changes.')}"
        )
        prompt = (
            "You are KubeLLM, an expert Kubernetes fault diagnosis system. "
            "Diagnose the failure and provide concrete kubectl commands.\n\n"
            f"### Diagnostic Query:\n{s['problem']}\n\n"
            f"### Prior Attempts:\n{chr(10).join(s.get('attempted_solutions', []))}\n\n"
            f"### Response:\n{response}"
        )
        return {"text": prompt}

    formatted = [format_sample(s) for s in raw]
    split_idx = int(len(formatted) * 0.8)
    train_ds = Dataset.from_list(formatted[:split_idx])
    eval_ds = Dataset.from_list(formatted[split_idx:])
    logger.info(f"[SFT] train={len(train_ds)} eval={len(eval_ds)}")

    resolved_precision, fp16, bf16 = resolve_precision(precision)
    compute_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[resolved_precision]
    logger.info(f"[SFT] precision={resolved_precision} load_in_4bit={load_in_4bit}")

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    logger.info(f"[SFT] loading {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    def tokenize(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)
        out["labels"] = out["input_ids"].copy()
        return out

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    eval_tok = eval_ds.map(tokenize, batched=True, remove_columns=["text"])

    training_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": "cosine",
        "save_strategy": save_strategy,
        "load_best_model_at_end": load_best_model_at_end,
        "fp16": fp16,
        "bf16": bf16,
        "seed": seed,
        "logging_steps": logging_steps,
        "report_to": "none",
        "gradient_checkpointing": gradient_checkpointing,
        "dataloader_pin_memory": dataloader_pin_memory,
    }
    if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        training_kwargs["eval_strategy"] = eval_strategy
    else:
        training_kwargs["evaluation_strategy"] = eval_strategy
    args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    logger.info("[SFT] training start")
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"[SFT] saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model_id")
    parser.add_argument("--dataset_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_dropout", type=float)
    parser.add_argument("--target_modules", nargs="+")
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction)
    parser.add_argument("--device_map")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction)
    parser.add_argument("--eval_strategy")
    parser.add_argument("--save_strategy")
    parser.add_argument("--load_best_model_at_end", action=argparse.BooleanOptionalAction)
    parser.add_argument("--logging_steps", type=int)
    parser.add_argument("--dataloader_pin_memory", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


def merged_sft_args(args: argparse.Namespace) -> dict:
    cfg = load_sft_config(args.config)
    cli_to_cfg = {
        "model_id": args.model_id,
        "dataset_path": args.dataset_path,
        "output_dir": args.output_dir,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "learning_rate": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "precision": args.precision,
        "load_in_4bit": args.load_in_4bit,
        "device_map": args.device_map,
        "gradient_checkpointing": args.gradient_checkpointing,
        "eval_strategy": args.eval_strategy,
        "save_strategy": args.save_strategy,
        "load_best_model_at_end": args.load_best_model_at_end,
        "logging_steps": args.logging_steps,
        "dataloader_pin_memory": args.dataloader_pin_memory,
        "seed": args.seed,
    }
    for key, value in cli_to_cfg.items():
        if value is not None:
            cfg[key] = value
    return cfg


if __name__ == "__main__":
    args = parse_args()
    run_sft(**merged_sft_args(args))
