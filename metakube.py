"""
MetaKube — top-level orchestrator F(Q, Θ) (Section 3, Eq 6).
Wires EPMN + MetaCognitiveController + dual pathways.
"""

from __future__ import annotations
import time
from pathlib import Path
import yaml
from loguru import logger

from epmn import EPMN, Embedder, EpisodicMemory
from kubegraph import KubeGraph
from kubellm.base import KubeLLMBase
from controller import MetaCognitiveController
from pathways import IntuitivePathway, AnalyticalPathway


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_kubellm(cfg: dict, backend: str = "ollama") -> KubeLLMBase:
    """
    Factory for KubeLLM backends.
    backend: 'stub' | 'ollama' | 'qwen'
    """
    if backend == "stub":
        from kubellm.stub import StubKubeLLM
        return StubKubeLLM()
    if backend == "ollama":
        from kubellm.ollama_llm import OllamaKubeLLM
        ollama_cfg = cfg.get("ollama", {})
        return OllamaKubeLLM(
            model=ollama_cfg.get("model", "gemma4:e2b"),
            base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
            max_tokens=ollama_cfg.get("max_tokens", 1024),
        )
    if backend == "qwen":
        from kubellm.qwen import QwenKubeLLM
        kcfg = cfg["kubellm"]
        lora_path = kcfg["model_path"] if Path(kcfg["model_path"]).exists() else None
        model_id = str(Path("/ai-data/models/qwen3-8b")) if Path("/ai-data/models/qwen3-8b").exists() else kcfg["model_id"]
        logger.info(f"[build_kubellm] qwen model={model_id}  lora={lora_path}")
        return QwenKubeLLM(
            model_id=model_id,
            load_in_4bit=kcfg["load_in_4bit"],
            lora_path=lora_path,
            max_new_tokens=kcfg["max_new_tokens"],
        )
    raise ValueError(f"Unknown backend: {backend}")


class MetaKube:
    """
    F : (Q, Θ) →
        P_int(S_int, C_int)   if C(M*) > τ
        P_ana(S_ana, C_ana)   otherwise
    """

    def __init__(self, cfg: dict, kubellm: KubeLLMBase, kubegraph: KubeGraph | None = None):
        ecfg = cfg["epmn"]
        mcfg = cfg["meta_controller"]
        kcfg = cfg["kubegraph"]

        self.embedder = Embedder(model_name=ecfg["embedding_model"])
        self.epmn = EPMN(
            embedder=self.embedder,
            max_episodes=ecfg["max_episodes"],
            pattern_threshold=ecfg["pattern_threshold"],
            retrieval_K=ecfg["retrieval_K"],
            temporal_decay_days=ecfg["temporal_decay_days"],
            lambda_mix=ecfg["lambda_mix"],
            pattern_min_size=ecfg["pattern_min_size"],
            confidence_weights=mcfg["confidence_weights"],
        )
        self.kubegraph = kubegraph or KubeGraph.from_toy()
        self.kubellm = kubellm
        self.controller = MetaCognitiveController(
            tau_init=mcfg["tau_init"],
            eta_meta=mcfg["eta_meta"],
            xi=mcfg["xi"],
        )

        alpha = tuple(kcfg["priority_weights"].values())  # (prior, path, novelty)
        self.intuitive = IntuitivePathway(self.epmn, self.kubegraph, self.kubellm)
        self.analytical = AnalyticalPathway(
            self.epmn, self.kubegraph, self.kubellm,
            max_hops=kcfg["max_hops"],
            alpha=alpha,
        )
        logger.info("[MetaKube] initialized")

    # ------------------------------------------------------------------
    # Main diagnostic entry point
    # ------------------------------------------------------------------
    def diagnose(self, query: str, symptoms: list[str] | None = None) -> dict:
        """
        Diagnose a Kubernetes failure query.
        Returns full result dict including pathway, response, confidence.
        """
        t0 = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"[MetaKube.diagnose] query={query[:100]}")

        if symptoms is None:
            symptoms = [query]

        query_vec = self.embedder.encode_query(symptoms)
        logger.debug(f"[MetaKube] query_vec shape={query_vec.shape}")

        memories, c_max = self.epmn.retrieve(query_vec)
        logger.info(f"[MetaKube] retrieved {len(memories)} memories, C_max={c_max:.3f}")

        pathway_name = self.controller.route(c_max)

        if pathway_name == "intuitive":
            result = self.intuitive.run(query, query_vec, memories, c_max)
        else:
            result = self.analytical.run(query, query_vec, memories, c_max)

        result["total_latency_s"] = time.time() - t0
        result["query"] = query
        result["controller_stats"] = self.controller.stats

        logger.info(f"[MetaKube] pathway={pathway_name} total={result['total_latency_s']:.2f}s")
        logger.info(f"{'='*60}")
        return result

    # ------------------------------------------------------------------
    # Continuous learning update
    # ------------------------------------------------------------------
    def update_memory(
        self,
        query: str,
        symptoms: list[str],
        resolution: str,
        outcomes: list[str],
        success: bool = True,
        context: dict | None = None,
    ) -> None:
        """Add resolved episode back into EPMN and re-form patterns."""
        episode = EpisodicMemory(
            symptoms=symptoms,
            context=context or {"query": query},
            actions=[resolution],
            outcomes=outcomes,
            adaptive_value=2.0 if success else 0.5,
        )
        self.epmn.add_episode(episode)

        # re-form patterns every 10 new episodes
        if len(self.epmn.episodes) % 10 == 0:
            self.epmn.form_patterns()
            logger.info(f"[MetaKube] patterns reformed: {len(self.epmn.patterns)}")

    def seed_from_kfrd(self, episodes: list[EpisodicMemory]) -> None:
        """Bulk-load episodes into EPMN and form initial patterns."""
        logger.info(f"[MetaKube] seeding {len(episodes)} episodes from KFRD")
        for ep in episodes:
            self.epmn.add_episode(ep)
        self.epmn.form_patterns()
        logger.info(f"[MetaKube] seeded — {len(self.epmn.patterns)} patterns formed")
