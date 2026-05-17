"""Qwen3-8B-Instruct + 4-bit BitsAndBytes + LoRA — Phase 4 KubeLLM."""

from __future__ import annotations
from loguru import logger

from .base import KubeLLMBase
from epmn.memory import EpisodicMemory, PatternAbstraction

INTUITIVE_SYSTEM = """You are KubeLLM, an expert Kubernetes fault diagnosis system.
You have access to historical resolution patterns from episodic memory.
Provide rapid, accurate diagnosis grounded in the provided patterns.
Be concise and actionable. Always suggest concrete kubectl commands."""

ANALYTICAL_SYSTEM = """You are KubeLLM, an expert Kubernetes fault diagnosis system.
You have access to causal chains from the KubeGraph knowledge graph.
Perform deep causal analysis, explain the full failure propagation path,
and provide step-by-step resolution with prevention recommendations."""

INTUITIVE_TEMPLATE = """{system}

## Retrieved Patterns:
{patterns}

## Diagnostic Query:
{query}

## Task:
Diagnose this Kubernetes failure using the retrieved patterns as contextual anchors.
Provide: 1) Root Cause, 2) Resolution Commands, 3) Prevention."""

ANALYTICAL_TEMPLATE = """{system}

## Retrieved Memory Patterns:
{patterns}

## Causal Chains from KubeGraph:
{chains}

## Diagnostic Query:
{query}

## Task:
Perform comprehensive causal analysis. Explain the full failure propagation,
provide step-by-step resolution, and document prevention strategies."""


class QwenKubeLLM(KubeLLMBase):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-8B-Instruct",
        load_in_4bit: bool = True,
        lora_path: str | None = None,
        max_new_tokens: int = 1024,
    ):
        logger.info(f"[QwenKubeLLM] loading {model_id} (4bit={load_in_4bit})")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        if lora_path:
            from peft import PeftModel
            logger.info(f"[QwenKubeLLM] loading LoRA from {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)

        self.model.eval()
        self.max_new_tokens = max_new_tokens
        logger.info("[QwenKubeLLM] ready")

    def _format_patterns(self, patterns: list) -> str:
        lines = []
        for i, m in enumerate(patterns[:5]):
            if isinstance(m, PatternAbstraction):
                lines.append(f"{i+1}. [{m.fault_category}] Strategy: {m.resolution_strategy} (reliability={m.reliability:.2f})")
                if m.canonical_symptoms:
                    lines.append(f"   Symptoms: {', '.join(m.canonical_symptoms[:3])}")
            elif isinstance(m, EpisodicMemory):
                lines.append(f"{i+1}. Symptoms: {', '.join(m.symptoms[:3])}")
                if m.actions:
                    lines.append(f"   Actions: {', '.join(m.actions[:2])}")
        return "\n".join(lines) if lines else "No patterns available."

    def _format_chains(self, chains: list[list[str]]) -> str:
        return "\n".join(
            f"{i+1}. {' → '.join(chain)}" for i, chain in enumerate(chains[:5])
        ) if chains else "No causal chains identified."

    def _generate(self, prompt: str) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_intuitive(
        self,
        query: str,
        patterns: list,
        hints: list[str] | None = None,
    ) -> str:
        logger.info("[QwenKubeLLM] intuitive pathway")
        patterns_str = self._format_patterns(patterns)
        prompt = INTUITIVE_TEMPLATE.format(
            system=INTUITIVE_SYSTEM,
            patterns=patterns_str,
            query=query,
        )
        return self._generate(prompt)

    def generate_analytical(
        self,
        query: str,
        patterns: list,
        causal_chains: list[list[str]],
    ) -> str:
        logger.info("[QwenKubeLLM] analytical pathway")
        patterns_str = self._format_patterns(patterns)
        chains_str = self._format_chains(causal_chains)
        prompt = ANALYTICAL_TEMPLATE.format(
            system=ANALYTICAL_SYSTEM,
            patterns=patterns_str,
            chains=chains_str,
            query=query,
        )
        return self._generate(prompt)
