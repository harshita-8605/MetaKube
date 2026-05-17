"""Adapter: converts AIOpsLab problem registry into MetaKube diagnostic queries."""

from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass
from loguru import logger


AIOPSLAB_PATH = Path("/ai-data/aoi/AIOpsLab")

PROBLEM_DESCRIPTIONS = {
    # Maps problem name prefix → (symptoms, fault_category, ground_truth_hint)
    "ad_service_failure": (
        ["Ad service returning errors", "Feature flag misconfiguration", "HTTP 5xx from ad endpoint"],
        "Configuration",
        "Disable adServiceFailure feature flag via kubectl",
    ),
    "ad_service_high_cpu": (
        ["Ad service CPU usage spike", "Service latency increased", "Container CPU throttling"],
        "Resource",
        "Scale ad service resources or fix CPU-intensive code path",
    ),
    "ad_service_manual_gc": (
        ["Ad service GC pauses", "Memory pressure", "JVM garbage collection overhead"],
        "Resource",
        "Trigger manual GC or adjust JVM heap settings",
    ),
    "cart_service_failure": (
        ["Cart service unavailable", "Redis connection refused", "Shopping cart errors"],
        "System",
        "Restart cart service and verify Redis connectivity",
    ),
    "image_slow_load": (
        ["Image loading slow", "High response time for image assets", "S3/CDN latency"],
        "Network",
        "Check CDN configuration and image service networking",
    ),
    "kafka_queue_problems": (
        ["Kafka consumer lag increasing", "Message queue backed up", "Consumer group rebalancing"],
        "System",
        "Restart Kafka consumers and check partition assignment",
    ),
    "loadgenerator_flood_homepage": (
        ["Homepage overloaded", "High request rate", "Load generator misconfigured"],
        "Resource",
        "Scale frontend deployment or adjust load generator rate",
    ),
    "payment_service_failure": (
        ["Payment service unavailable", "Transaction errors", "gRPC connection failed"],
        "System",
        "Restart payment service and verify downstream dependencies",
    ),
    "payment_service_unreachable": (
        ["Payment service unreachable", "Network timeout", "DNS resolution failure for payment"],
        "Network",
        "Check network policies and service endpoints for payment service",
    ),
    "product_catalog_failure": (
        ["Product catalog returning errors", "gRPC server error", "Product listing unavailable"],
        "System",
        "Restart product catalog service and check configuration",
    ),
    "recommendation_service_cache_failure": (
        ["Recommendation cache miss", "Redis cache connection error", "Slow recommendations"],
        "Configuration",
        "Fix Redis connection configuration for recommendation service",
    ),
    "k8s_target_port_misconfig": (
        ["Service unreachable", "Target port mismatch", "kubectl port-forward fails"],
        "Configuration",
        "Fix targetPort in Service spec to match container port",
    ),
    "auth_miss_mongodb": (
        ["MongoDB authentication failed", "ECONNREFUSED", "Database connection error"],
        "Configuration",
        "Configure MongoDB authentication credentials and RBAC",
    ),
    "revoke_auth": (
        ["Authorization revoked", "403 Forbidden", "RBAC permission denied"],
        "Configuration",
        "Restore RBAC permissions via kubectl apply",
    ),
    "network_loss": (
        ["Packet loss detected", "Network degradation", "Intermittent connection drops"],
        "Network",
        "Investigate CNI configuration and network policies",
    ),
    "network_delay": (
        ["High network latency", "Service timeout", "Slow inter-pod communication"],
        "Network",
        "Check tc/netem rules and network bandwidth allocation",
    ),
    "container_kill": (
        ["Container killed unexpectedly", "OOMKilled or SIGKILL", "Pod restart loop"],
        "Resource",
        "Increase memory limits or fix container crash",
    ),
    "pod_failure": (
        ["Pod in failed state", "CrashLoopBackOff", "Init container error"],
        "Configuration",
        "Fix init container or application configuration",
    ),
    "pod_kill": (
        ["Pod killed", "Disruption budget violated", "Unexpected pod termination"],
        "System",
        "Investigate pod disruption and restart policies",
    ),
    "disk_woreout": (
        ["Disk pressure on node", "PVC write errors", "Node DiskPressure condition"],
        "System",
        "Free disk space or resize PersistentVolume",
    ),
    "kernel_fault": (
        ["Kernel panic on node", "Node unreachable", "OOM in kernel space"],
        "System",
        "Reboot affected node and investigate kernel logs",
    ),
}


@dataclass
class AIOpsLabQuery:
    problem_id: str
    task_type: str          # detection | localization | analysis | mitigation
    query: str
    symptoms: list[str]
    fault_category: str
    ground_truth_hint: str
    expected_answer: str    # used for eval


def load_aiopslab_queries(max_problems: int = 30) -> list[AIOpsLabQuery]:
    """
    Extract diagnostic queries from AIOpsLab problem registry.
    Does NOT inject faults — only reads metadata for offline eval.
    """
    queries: list[AIOpsLabQuery] = []

    for prefix, (symptoms, category, gt_hint) in PROBLEM_DESCRIPTIONS.items():
        for task_type in ["detection", "localization", "analysis", "mitigation"]:
            problem_id = f"{prefix}-{task_type}-1"
            query = _build_query(symptoms, task_type, prefix)
            expected = _expected_answer(task_type, prefix)

            queries.append(AIOpsLabQuery(
                problem_id=problem_id,
                task_type=task_type,
                query=query,
                symptoms=symptoms,
                fault_category=category,
                ground_truth_hint=gt_hint,
                expected_answer=expected,
            ))

            if len(queries) >= max_problems:
                break
        if len(queries) >= max_problems:
            break

    logger.info(f"[AIOpsLabAdapter] loaded {len(queries)} queries")
    return queries


def _build_query(symptoms: list[str], task_type: str, prefix: str) -> str:
    symptom_str = ". ".join(symptoms)
    task_prompts = {
        "detection": f"Is there an active fault in the cluster? Symptoms: {symptom_str}",
        "localization": f"Which service/component is the root cause? Symptoms: {symptom_str}",
        "analysis": f"What is the root cause and failure propagation? Symptoms: {symptom_str}",
        "mitigation": f"How should we resolve this fault? Symptoms: {symptom_str} Provide kubectl commands.",
    }
    return task_prompts.get(task_type, f"Diagnose: {symptom_str}")


def _expected_answer(task_type: str, prefix: str) -> str:
    answers = {
        "detection": "Yes",
        "localization": prefix.replace("_", "-").split("-")[0],
        "analysis": f"Root cause in {prefix.replace('_', ' ')} component",
        "mitigation": "kubectl rollout restart / apply fix",
    }
    return answers.get(task_type, "")
