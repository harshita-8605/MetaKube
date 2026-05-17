"""Abstract KubeLLM interface — identical for stub and Qwen3-8B."""

from __future__ import annotations
from abc import ABC, abstractmethod


class KubeLLMBase(ABC):
    @abstractmethod
    def generate_intuitive(
        self,
        query: str,
        patterns: list,
        hints: list[str] | None = None,
    ) -> str:
        """Template-mode: ground response in retrieved patterns (θ_int)."""
        ...

    @abstractmethod
    def generate_analytical(
        self,
        query: str,
        patterns: list,
        causal_chains: list[list[str]],
    ) -> str:
        """Synthesis-mode: integrate causal chains into full diagnostic (θ_ana)."""
        ...
