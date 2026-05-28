"""Phase 4: LoRA SFT of Qwen3-8B on KFRD dataset (Appendix C.2, Eq 28)."""

from __future__ import annotations
import json
from pathlib import Path
from loguru import logger


def run_sft(
    model_id: str = "Qwen/Qwen3-8B-Instruct",
    dataset_path: str = "/ai-data/datasets/kfrd/synthetic_kfrd.json",
    output_dir: str = "/ai-data/models/kubellm",
    lora_rank: int = 64,
    lora_alpha: int = 128,
    num_epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.01,
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
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    logger.info(f"[SFT] loading dataset from {dataset_path}")
    with open(dataset_path) as f:
        raw = json.load(f)

    def format_sample(s: dict) -> dict:
        prompt = (
            f"### Problem:\n{s['problem']}\n\n"
            f"### Attempted Solutions:\n{chr(10).join(s.get('attempted_solutions', []))}\n\n"
            f"### Reasoning:\n{s['reasoning']}\n\n"
            f"### Solution:\n{s['final_solution']}"
        )
        return {"text": prompt}

    formatted = [format_sample(s) for s in raw]
    split_idx = int(len(formatted) * 0.8)
    train_ds = Dataset.from_list(formatted[:split_idx])
    eval_ds = Dataset.from_list(formatted[split_idx:])
    logger.info(f"[SFT] train={len(train_ds)} eval={len(eval_ds)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    logger.info(f"[SFT] loading {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    def tokenize(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    eval_tok = eval_ds.map(tokenize, batched=True, remove_columns=["text"])

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=False,
        bf16=True,
        seed=seed,
        logging_steps=10,
        report_to="none",
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
    )

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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen3-8B-Instruct")
    parser.add_argument("--dataset_path", default="/ai-data/datasets/kfrd/synthetic_kfrd.json")
    parser.add_argument("--output_dir", default="/ai-data/models/kubellm")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()
    run_sft(
        model_id=args.model_id,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
