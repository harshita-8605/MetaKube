"""Toy KubeLLM stub — template strings, no GPU required (Phase 3)."""

from __future__ import annotations
from loguru import logger

from .base import KubeLLMBase
from epmn.memory import EpisodicMemory, PatternAbstraction


class StubKubeLLM(KubeLLMBase):
    """Deterministic template-based stub. Interfaces identical to QwenKubeLLM."""

    def generate_intuitive(
        self,
        query: str,
        patterns: list,
        hints: list[str] | None = None,
    ) -> str:
        logger.info("[StubKubeLLM] intuitive pathway — template mode")

        resolution = "No matching resolution found."
        fault_category = "unknown"

        for m in patterns:
            if isinstance(m, PatternAbstraction):
                resolution = m.resolution_strategy or resolution
                fault_category = m.fault_category
                break
            elif isinstance(m, EpisodicMemory) and m.actions:
                resolution = m.actions[0]
                fault_category = m.context.get("fault_category", fault_category)
                break

        hint_str = ", ".join(hints[:3]) if hints else "none"
        response = (
            f"[INTUITIVE DIAGNOSIS]\n"
            f"Query: {query[:200]}\n"
            f"Fault Category: {fault_category}\n"
            f"Root Cause: Pattern match from episodic memory.\n"
            f"Resolution: {resolution}\n"
            f"Knowledge Hints: {hint_str}\n"
            f"Confidence: HIGH (pattern-matched)"
        )
        logger.debug(f"[StubKubeLLM] intuitive response generated")
        return response

    def generate_analytical(
        self,
        query: str,
        patterns: list,
        causal_chains: list[list[str]],
    ) -> str:
        logger.info("[StubKubeLLM] analytical pathway — causal synthesis mode")

        chains_str = "\n".join(
            f"  Chain {i+1}: {' → '.join(chain)}"
            for i, chain in enumerate(causal_chains[:3])
        )
        if not chains_str:
            chains_str = "  No causal chains identified."

        resolution_steps = []
        for chain in causal_chains[:2]:
            resolutions = [n for n in chain if "fix" in n.lower() or "restart" in n.lower() or "increase" in n.lower() or "add" in n.lower() or "free" in n.lower()]
            resolution_steps.extend(resolutions)

        steps_str = "\n".join(f"  {i+1}. kubectl {s}" for i, s in enumerate(resolution_steps[:3])) or "  1. Investigate cluster state\n  2. Check pod logs\n  3. Verify resource quotas"

        response = (
            f"[ANALYTICAL DIAGNOSIS]\n"
            f"Query: {query[:200]}\n\n"
            f"Causal Analysis:\n{chains_str}\n\n"
            f"Resolution Steps:\n{steps_str}\n\n"
            f"Prevention: Monitor resource usage and configure appropriate limits.\n"
            f"Confidence: MEDIUM (causal reasoning required)"
        )
        logger.debug(f"[StubKubeLLM] analytical response generated")
        return response
