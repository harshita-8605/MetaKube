"""OllamaKubeLLM — uses local ollama REST API as KubeLLM backend."""

from __future__ import annotations
import json
import urllib.request
import urllib.error
from loguru import logger

from .base import KubeLLMBase
from epmn.memory import EpisodicMemory, PatternAbstraction

INTUITIVE_SYSTEM = """You are KubeLLM, an expert Kubernetes fault diagnosis system.
You have access to historical resolution patterns from episodic memory.
Provide rapid, accurate diagnosis grounded in the provided patterns.
Be concise and actionable. Always include concrete kubectl commands."""

ANALYTICAL_SYSTEM = """You are KubeLLM, an expert Kubernetes fault diagnosis system.
You have access to causal chains from the KubeGraph knowledge graph.
Perform deep causal analysis, explain the full failure propagation path,
and provide step-by-step resolution with prevention recommendations.
Always include kubectl commands."""


class OllamaKubeLLM(KubeLLMBase):
    def __init__(
        self,
        model: str = "gemma4:e2b",
        base_url: str = "http://localhost:11434",
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        logger.info(f"[OllamaKubeLLM] model={model} url={base_url}")

    def _call(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            },
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                return result.get("response", "").strip()
        except urllib.error.URLError as e:
            logger.error(f"[OllamaKubeLLM] request failed: {e}")
            return f"[ERROR] Ollama unavailable: {e}"

    def _format_patterns(self, patterns: list) -> str:
        lines = []
        for i, m in enumerate(patterns[:4]):
            if isinstance(m, PatternAbstraction):
                lines.append(f"{i+1}. [{m.fault_category}] {m.resolution_strategy} (reliability={m.reliability:.2f})")
                if m.canonical_symptoms:
                    lines.append(f"   Symptoms: {', '.join(m.canonical_symptoms[:3])}")
            elif isinstance(m, EpisodicMemory):
                lines.append(f"{i+1}. Symptoms: {', '.join(m.symptoms[:3])}")
                if m.actions:
                    lines.append(f"   Resolution: {m.actions[0]}")
        return "\n".join(lines) if lines else "No prior patterns available."

    def _format_chains(self, chains: list[list[str]]) -> str:
        if not chains:
            return "No causal chains identified."
        return "\n".join(
            f"{i+1}. {' → '.join(chain)}" for i, chain in enumerate(chains[:5])
        )

    def generate_intuitive(
        self,
        query: str,
        patterns: list,
        hints: list[str] | None = None,
    ) -> str:
        logger.info(f"[OllamaKubeLLM] intuitive pathway — {self.model}")
        patterns_str = self._format_patterns(patterns)
        hint_str = ", ".join((hints or [])[:5]) or "none"

        prompt = f"""{INTUITIVE_SYSTEM}

## Retrieved Memory Patterns:
{patterns_str}

## Knowledge Graph Hints:
{hint_str}

## Diagnostic Query:
{query}

## Instructions:
Diagnose this Kubernetes failure. Provide:
1. Root Cause (1-2 sentences)
2. Resolution Commands (kubectl commands)
3. Prevention (1 sentence)

Use the patterns above as context."""

        return self._call(prompt)

    def generate_analytical(
        self,
        query: str,
        patterns: list,
        causal_chains: list[list[str]],
    ) -> str:
        logger.info(f"[OllamaKubeLLM] analytical pathway — {self.model}")
        patterns_str = self._format_patterns(patterns)
        chains_str = self._format_chains(causal_chains)

        prompt = f"""{ANALYTICAL_SYSTEM}

## Retrieved Memory Patterns:
{patterns_str}

## Causal Chains from KubeGraph:
{chains_str}

## Diagnostic Query:
{query}

## Instructions:
Perform comprehensive causal analysis. Provide:
1. Root Cause Analysis (explain failure propagation through the causal chains)
2. Step-by-Step Resolution (numbered kubectl commands)
3. Verification Steps (how to confirm fix worked)
4. Prevention (1-2 sentences)"""

        return self._call(prompt)
