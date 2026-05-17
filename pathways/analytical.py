"""Analytical pathway P_ana: Q → EPMN → KubeGraph → KubeLLM(θ_ana) → S_ana  (Eq 4)."""

from __future__ import annotations
import time
from loguru import logger

from epmn.epmn import EPMN
from kubegraph.graph import KubeGraph
from kubellm.base import KubeLLMBase


class AnalyticalPathway:
    def __init__(
        self,
        epmn: EPMN,
        kubegraph: KubeGraph,
        kubellm: KubeLLMBase,
        max_hops: int = 3,
        alpha: tuple[float, float, float] = (0.5, 0.3, 0.2),
    ):
        self.epmn = epmn
        self.kubegraph = kubegraph
        self.kubellm = kubellm
        self.max_hops = max_hops
        self.alpha = alpha

    def run(self, query: str, query_vec, memories: list, c_max: float) -> dict:
        """
        P_ana: Q --EPMN--> M* --KubeGraph--> G* --KubeLLM(θ_ana)--> S_ana
        Returns solution dict with causal chains and deep response.
        """
        t0 = time.time()
        logger.info(f"[AnalyticalPathway] running — C_max={c_max:.3f}")

        causal_chains = self.kubegraph.memory_biased_search(
            memories,
            max_hops=self.max_hops,
            alpha=self.alpha,
            top_n=5,
        )
        logger.info(f"[AnalyticalPathway] {len(causal_chains)} causal chains extracted")
        for i, chain in enumerate(causal_chains):
            logger.debug(f"  Chain {i+1}: {' → '.join(chain)}")

        response = self.kubellm.generate_analytical(query, memories, causal_chains)
        latency = time.time() - t0

        logger.info(f"[AnalyticalPathway] done in {latency:.2f}s")
        return {
            "pathway": "analytical",
            "response": response,
            "c_max": c_max,
            "latency_s": latency,
            "memories_used": len(memories),
            "causal_chains": causal_chains,
        }
