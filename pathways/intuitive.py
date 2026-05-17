"""Intuitive pathway P_int: Q → EPMN → KubeLLM(θ_int) → S_int  (Eq 1)."""

from __future__ import annotations
import time
from loguru import logger

from epmn.epmn import EPMN
from kubegraph.graph import KubeGraph
from kubellm.base import KubeLLMBase


class IntuitivePathway:
    def __init__(self, epmn: EPMN, kubegraph: KubeGraph, kubellm: KubeLLMBase):
        self.epmn = epmn
        self.kubegraph = kubegraph
        self.kubellm = kubellm

    def run(self, query: str, query_vec, memories: list, c_max: float) -> dict:
        """
        P_int: Q --EPMN--> M* --KubeLLM(θ_int)--> S_int
        Returns solution dict with response text and metadata.
        """
        t0 = time.time()
        logger.info(f"[IntuitivePathway] running — C_max={c_max:.3f}")

        hints = self.kubegraph.get_hints(memories, top_k=5)
        logger.debug(f"[IntuitivePathway] graph hints: {hints[:3]}")

        response = self.kubellm.generate_intuitive(query, memories, hints=hints)
        latency = time.time() - t0

        logger.info(f"[IntuitivePathway] done in {latency:.2f}s")
        return {
            "pathway": "intuitive",
            "response": response,
            "c_max": c_max,
            "latency_s": latency,
            "memories_used": len(memories),
            "hints": hints,
        }
