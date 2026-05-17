"""Episodic Pattern Memory Network — Algorithm 1 from paper (Section 4.1)."""

from __future__ import annotations
import math
import time
import numpy as np
from loguru import logger
from sklearn.cluster import AgglomerativeClustering

from .memory import EpisodicMemory, PatternAbstraction
from .embedder import Embedder


class EPMN:
    def __init__(
        self,
        embedder: Embedder,
        max_episodes: int = 5000,
        pattern_threshold: float = 0.85,
        retrieval_K: int = 10,
        temporal_decay_days: float = 30.0,
        lambda_mix: float = 0.7,
        pattern_min_size: int = 3,
        confidence_weights: dict | None = None,
    ):
        self.embedder = embedder
        self.max_episodes = max_episodes
        self.theta_sim = pattern_threshold
        self.K = retrieval_K
        self.tau_r = temporal_decay_days
        self.lam = lambda_mix
        self.theta_pattern = pattern_min_size

        cw = confidence_weights or {}
        self.zeta_sim = cw.get("similarity", 1.0)
        self.zeta_rel = cw.get("relevance", 0.8)
        self.zeta_rel2 = cw.get("reliability", 0.6)

        self.episodes: list[EpisodicMemory] = []
        self.patterns: list[PatternAbstraction] = []

        logger.info("[EPMN] initialized — pool empty")

    # ------------------------------------------------------------------
    # FormPatterns (Algorithm 1, lines 1-10)
    # ------------------------------------------------------------------
    def form_patterns(self) -> list[PatternAbstraction]:
        """Cluster episodes by embedding similarity; abstract patterns."""
        if len(self.episodes) < self.theta_pattern:
            logger.debug(f"[EPMN.FormPatterns] only {len(self.episodes)} episodes, skip")
            return self.patterns

        vecs = np.stack([e.embedding for e in self.episodes])

        # distance = 1 - cosine_similarity; threshold = 1 - theta_sim
        distance_threshold = 1.0 - self.theta_sim
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold,
        )
        labels = clustering.fit_predict(vecs)
        n_clusters = labels.max() + 1
        logger.info(f"[EPMN.FormPatterns] {n_clusters} patterns from {len(self.episodes)} episodes")

        new_patterns: list[PatternAbstraction] = []
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if mask.sum() < self.theta_pattern:
                continue
            cluster_eps = [e for e, m in zip(self.episodes, mask) if m]
            cluster_vecs = vecs[mask]

            centroid = cluster_vecs.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-9
            variance = cluster_vecs.var(axis=0)

            # reliability = fraction of episodes with positive outcome
            positive = sum(1 for e in cluster_eps if any("resolved" in o.lower() or "fixed" in o.lower() for o in e.outcomes))
            reliability = positive / len(cluster_eps) if cluster_eps else 0.5

            all_symptoms = [s for e in cluster_eps for s in e.symptoms]
            canonical = list(dict.fromkeys(all_symptoms))[:5]

            p = PatternAbstraction(
                symptom_centroid=centroid,
                symptom_variance=variance,
                resolution_strategy=cluster_eps[0].actions[0] if cluster_eps[0].actions else "",
                reliability=reliability,
                episode_ids=[e.episode_id for e in cluster_eps],
                canonical_symptoms=canonical,
                fault_category=cluster_eps[0].context.get("fault_category", "unknown"),
            )
            new_patterns.append(p)

        self.patterns = new_patterns
        return self.patterns

    # ------------------------------------------------------------------
    # Retrieve (Algorithm 1, lines 11-19)
    # ------------------------------------------------------------------
    def retrieve(self, query_vec: np.ndarray) -> tuple[list[EpisodicMemory | PatternAbstraction], float]:
        """
        M* = TopK(ψ · M_P + (1-ψ) · M_E, K)
        Returns (top-K memories, C_max).
        """
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)

        novelty = self._novelty(query_vec)
        complexity = self._complexity(query_vec)
        psi = self._mixing_weight(novelty, complexity)
        logger.debug(f"[EPMN.Retrieve] novelty={novelty:.3f} complexity={complexity:.3f} ψ={psi:.3f}")

        scored_patterns = [
            (p, self._score_pattern(p, query_vec))
            for p in self.patterns
        ]
        scored_episodes = [
            (e, self._score_episode(e, query_vec))
            for e in self.episodes
        ]

        # Weighted pool: patterns get ψ weight, episodes get (1-ψ)
        pool: list[tuple[EpisodicMemory | PatternAbstraction, float]] = (
            [(p, psi * s) for p, s in scored_patterns] +
            [(e, (1 - psi) * s) for e, s in scored_episodes]
        )

        if not pool:
            logger.debug("[EPMN.Retrieve] empty pool, returning empty")
            return [], 0.0

        pool.sort(key=lambda x: x[1], reverse=True)
        top_k = pool[:self.K]
        memories = [m for m, _ in top_k]
        c_max = self.confidence(memories[0], query_vec) if memories else 0.0
        logger.info(f"[EPMN.Retrieve] retrieved {len(memories)} memories, C_max={c_max:.3f}")
        return memories, c_max

    # ------------------------------------------------------------------
    # Confidence (Algorithm 1, lines 20-23) — Eq 15
    # ------------------------------------------------------------------
    def confidence(self, memory: EpisodicMemory | PatternAbstraction, query_vec: np.ndarray) -> float:
        """C(m, Q) = Π f_j(m, Q)^ζ_j  (Eq 15)"""
        sim = self._cosine_sim(memory, query_vec)
        rel = self._temporal_relevance(memory)
        reliability = self._reliability(memory)

        c = (max(sim, 1e-6) ** self.zeta_sim *
             max(rel, 1e-6) ** self.zeta_rel *
             max(reliability, 1e-6) ** self.zeta_rel2)
        return float(c)

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------
    def add_episode(self, episode: EpisodicMemory) -> None:
        if episode.embedding is None:
            text = ". ".join(episode.symptoms)
            episode.embedding = self.embedder.encode([text])[0]

        self.episodes.append(episode)
        if len(self.episodes) > self.max_episodes:
            # evict lowest adaptive_value
            self.episodes.sort(key=lambda e: e.adaptive_value, reverse=True)
            self.episodes = self.episodes[:self.max_episodes]
            logger.debug("[EPMN] evicted low-value episodes")

    def update_adaptive_value(self, episode_id: str, success: bool) -> None:
        """Refine memory weights based on diagnostic outcome."""
        for e in self.episodes:
            if e.episode_id == episode_id:
                e.adaptive_value = min(e.adaptive_value * 1.2 if success else e.adaptive_value * 0.8, 5.0)
                logger.debug(f"[EPMN] updated adaptive_value for {episode_id} → {e.adaptive_value:.2f}")
                return

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _score_pattern(self, p: PatternAbstraction, q: np.ndarray) -> float:
        """λ·sim(p,Q) + (1-λ)·rec(p)  — Eq 2 applied to patterns."""
        if p.symptom_centroid is None:
            return 0.0
        sim = float(np.dot(p.symptom_centroid, q))
        rec = p.reliability  # patterns don't decay; use reliability as proxy
        return self.lam * sim + (1 - self.lam) * rec

    def _score_episode(self, e: EpisodicMemory, q: np.ndarray) -> float:
        """λ·sim(e,Q) + (1-λ)·rec(e_i)  — Eq 2."""
        if e.embedding is None:
            return 0.0
        sim = float(np.dot(e.embedding, q))
        rec = e.recency_score(self.tau_r)
        return self.lam * sim + (1 - self.lam) * rec

    def _cosine_sim(self, memory, query_vec: np.ndarray) -> float:
        if isinstance(memory, PatternAbstraction):
            vec = memory.symptom_centroid
        else:
            vec = memory.embedding
        if vec is None:
            return 0.0
        return float(np.dot(vec, query_vec) / (np.linalg.norm(vec) * np.linalg.norm(query_vec) + 1e-9))

    def _temporal_relevance(self, memory) -> float:
        """f_temp = exp(-Δt / T_temp); patterns decay slowly."""
        if isinstance(memory, EpisodicMemory):
            delta_days = (time.time() - memory.timestamp) / 86400.0
            return float(math.exp(-delta_days / self.tau_r))
        return 0.9  # patterns don't timestamp-decay

    def _reliability(self, memory) -> float:
        if isinstance(memory, PatternAbstraction):
            return memory.reliability
        return memory.adaptive_value / 5.0  # normalize to [0,1]

    def _novelty(self, query_vec: np.ndarray) -> float:
        """f_nov(Q) = min_{m∈M} d(Q, m) — distance to nearest episode."""
        if not self.episodes:
            return 1.0
        vecs = np.stack([e.embedding for e in self.episodes if e.embedding is not None])
        sims = vecs @ query_vec
        return float(1.0 - sims.max())

    def _complexity(self, query_vec: np.ndarray) -> float:
        """f_comp proxy: entropy of similarity distribution."""
        if not self.episodes:
            return 1.0
        vecs = np.stack([e.embedding for e in self.episodes if e.embedding is not None])
        sims = np.clip(vecs @ query_vec, 1e-9, 1.0)
        probs = sims / sims.sum()
        entropy = -float((probs * np.log(probs)).sum())
        return min(entropy / math.log(len(self.episodes) + 1), 1.0)

    def _mixing_weight(self, novelty: float, complexity: float) -> float:
        """ψ = σ(W_ψ · [f_nov, f_comp]) — Eq 14, simplified to logistic."""
        score = 0.5 * novelty + 0.5 * complexity
        return float(1.0 / (1.0 + math.exp(-6.0 * (score - 0.5))))
