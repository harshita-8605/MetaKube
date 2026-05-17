"""4-metric evaluation from paper Section 5.1 (Appendix A.4)."""

from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass
class DiagnosticScore:
    effectiveness: float = 0.0   # root cause + resolution capability
    equivalence: float = 0.0     # alignment with reference approach
    completeness: float = 0.0    # coverage of steps, commands, edge cases
    safety_accuracy: float = 0.0 # correctness + K8s best practices
    avg: float = 0.0

    def compute_avg(self) -> "DiagnosticScore":
        self.avg = (self.effectiveness + self.equivalence +
                    self.completeness + self.safety_accuracy) / 4.0
        return self


def heuristic_score(
    prediction: str,
    ground_truth: str,
    fault_category: str = "",
) -> DiagnosticScore:
    """
    Lightweight heuristic scorer for toy evaluation (no LLM judge needed).
    Paper uses GPT-5 + human experts; this approximates on keyword overlap.
    Scores on 0-100 scale matching paper's Table 1.
    """
    pred_lower = prediction.lower()
    gt_lower = ground_truth.lower()

    kubectl_commands = re.findall(r"kubectl\s+\w+", pred_lower)
    gt_commands = re.findall(r"kubectl\s+\w+", gt_lower)

    def token_overlap(a: str, b: str) -> float:
        ta = set(a.split())
        tb = set(b.split())
        return len(ta & tb) / (len(ta | tb) + 1e-9)

    overlap = token_overlap(pred_lower, gt_lower)
    cmd_precision = (len(set(kubectl_commands) & set(gt_commands)) /
                     (len(set(gt_commands)) + 1e-9))

    k8s_keywords = ["kubectl", "namespace", "pod", "deployment", "service",
                    "configmap", "secret", "node", "container", "log"]
    kw_coverage = sum(1 for kw in k8s_keywords if kw in pred_lower) / len(k8s_keywords)

    safety_terms = ["--dry-run", "backup", "describe", "logs", "rollout",
                    "--ignore-daemonsets", "drain", "uncordon"]
    safety_score = sum(1 for t in safety_terms if t in pred_lower) / len(safety_terms)

    score = DiagnosticScore(
        effectiveness=min(100.0, (overlap * 50 + cmd_precision * 50)),
        equivalence=min(100.0, overlap * 100),
        completeness=min(100.0, (kw_coverage * 60 + cmd_precision * 40)),
        safety_accuracy=min(100.0, (safety_score * 40 + kw_coverage * 60)),
    )
    return score.compute_avg()


def aggregate_scores(scores: list[DiagnosticScore]) -> DiagnosticScore:
    n = len(scores)
    if n == 0:
        return DiagnosticScore()
    return DiagnosticScore(
        effectiveness=sum(s.effectiveness for s in scores) / n,
        equivalence=sum(s.equivalence for s in scores) / n,
        completeness=sum(s.completeness for s in scores) / n,
        safety_accuracy=sum(s.safety_accuracy for s in scores) / n,
    ).compute_avg()
