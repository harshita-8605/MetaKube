"""Meta-cognitive controller Ψ: confidence → pathway routing + τ adaptation (Eq 6-11)."""

from __future__ import annotations
import math
import time
from collections import deque
from loguru import logger


class MetaCognitiveController:
    """
    Ψ : (Q, M*, Θ) → (p, τ, ω)
    Routes to 'intuitive' or 'analytical' based on C(M*) vs τ.
    Adapts τ via meta-learning: τ_{t+1} = τ_t - η·∇_τ L(τ, H)  (Eq 10)
    """

    def __init__(
        self,
        tau_init: float = 0.75,
        eta_meta: float = 0.01,
        xi: float = 0.6,
        history_size: int = 100,
    ):
        self.tau = tau_init
        self.eta = eta_meta
        self.xi = xi
        self._history: deque[dict] = deque(maxlen=history_size)
        logger.info(f"[MetaController] τ₀={tau_init}, η={eta_meta}, ξ={xi}")

    # ------------------------------------------------------------------
    # Route (Eq 6 / 9)
    # ------------------------------------------------------------------
    def route(self, c_max: float) -> str:
        """Return 'intuitive' if C(M*) > τ, else 'analytical'."""
        pathway = "intuitive" if c_max > self.tau else "analytical"
        logger.info(f"[MetaController] C_max={c_max:.3f} τ={self.tau:.3f} → {pathway}")
        return pathway

    # ------------------------------------------------------------------
    # Record outcome for meta-learning
    # ------------------------------------------------------------------
    def record_outcome(
        self,
        pathway: str,
        c_max: float,
        error: float,
        latency_s: float,
    ) -> None:
        """Store (pathway, c_max, error, latency) for threshold adaptation."""
        self._history.append({
            "pathway": pathway,
            "c_max": c_max,
            "error": error,
            "latency": latency_s,
            "timestamp": time.time(),
        })
        self._adapt_threshold()

    # ------------------------------------------------------------------
    # τ adaptation (Eq 10-11)
    # ------------------------------------------------------------------
    def _adapt_threshold(self) -> None:
        """
        L(τ, H) = ξ·Error(τ,H) + (1-ξ)·Latency(τ,H)
        τ_{t+1} = τ_t - η·∇_τ L
        Gradient estimated numerically from recent history.
        """
        if len(self._history) < 10:
            return

        recent = list(self._history)[-20:]

        # Intuitive path outcomes when c_max > current τ
        int_errors = [h["error"] for h in recent if h["pathway"] == "intuitive"]
        int_latencies = [h["latency"] for h in recent if h["pathway"] == "intuitive"]
        ana_errors = [h["error"] for h in recent if h["pathway"] == "analytical"]
        ana_latencies = [h["latency"] for h in recent if h["pathway"] == "analytical"]

        def safe_mean(lst):
            return sum(lst) / len(lst) if lst else 0.5

        error_int = safe_mean(int_errors)
        lat_int = safe_mean(int_latencies)
        error_ana = safe_mean(ana_errors)
        lat_ana = safe_mean(ana_latencies)

        # Loss gradient w.r.t. τ: raising τ → more analytical (lower error, higher latency)
        # ∂L/∂τ ≈ ξ·(error_ana - error_int) + (1-ξ)·(lat_int - lat_ana)
        grad = self.xi * (error_ana - error_int) + (1 - self.xi) * (lat_int - lat_ana)
        self.tau = float(max(0.3, min(0.95, self.tau - self.eta * grad)))
        logger.debug(f"[MetaController] τ adapted to {self.tau:.4f} (grad={grad:.4f})")

    @property
    def stats(self) -> dict:
        return {
            "tau": self.tau,
            "history_len": len(self._history),
            "intuitive_count": sum(1 for h in self._history if h["pathway"] == "intuitive"),
            "analytical_count": sum(1 for h in self._history if h["pathway"] == "analytical"),
        }
