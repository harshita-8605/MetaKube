"""EPMN dual-layer memory structures (Section 4.1.1)."""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
import numpy as np


@dataclass
class EpisodicMemory:
    """Single diagnostic episode e_i = (s_i, c_i, a_i, o_i, t_i, ω_i)."""
    symptoms: list[str]
    context: dict
    actions: list[str]
    outcomes: list[str]
    timestamp: float = field(default_factory=time.time)
    adaptive_value: float = 1.0
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    embedding: np.ndarray | None = field(default=None, repr=False)

    def recency_score(self, decay_days: float = 30.0) -> float:
        """rec(m_i) = exp(-Δt_i / τ_r)"""
        delta_days = (time.time() - self.timestamp) / 86400.0
        return float(np.exp(-delta_days / decay_days))


@dataclass
class PatternAbstraction:
    """Abstracted pattern p_j capturing statistical properties of similar episodes."""
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    symptom_centroid: np.ndarray = field(default=None, repr=False)
    symptom_variance: np.ndarray = field(default=None, repr=False)
    resolution_strategy: str = ""
    reliability: float = 0.5          # ρ_j: success rate across pattern episodes
    episode_ids: list[str] = field(default_factory=list)
    canonical_symptoms: list[str] = field(default_factory=list)
    fault_category: str = "unknown"

    def confidence(self, query_vec: np.ndarray, lam_sim: float, lam_rel: float) -> float:
        """Weighted confidence for this pattern given a query."""
        if self.symptom_centroid is None:
            return 0.0
        sim = float(np.dot(self.symptom_centroid, query_vec) /
                    (np.linalg.norm(self.symptom_centroid) * np.linalg.norm(query_vec) + 1e-9))
        return lam_sim * sim + lam_rel * self.reliability
