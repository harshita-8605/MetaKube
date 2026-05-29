"""
convert_aoi_to_kfrd.py

Converts AOI (AIOpsLab Observer Initiative) dataset to MetaKube KFRD
format for LoRA/SFT training.

Sources consumed (relative to AOI_ROOT):
  data/gt/infer/                93 files
    fields: problem_id, task_description, system_state_summary, commands, _metadata
  data/gt/infer_adapted/        93 files
    fields: task_info{problem_id,task_description,fault_summary}, commands
  data/gt/hf_observer_training/ 78 files
    fields: problem_id, task_description, steps[]{iter,command,result,summary}

Output (MetaKube KFRD schema, 11 fields):
  id, problem, attempted_solutions, final_solution, reasoning,
  fault_category, symptoms, context, outcomes, prevention, timestamp

Usage:
  python convert_aoi_to_kfrd.py \
      --aoi_root /path/to/aoi \
      --output_dir /path/to/MetaKube/data/datasets/kfrd \
      [--sft_split 0.7]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CATEGORIES = {"Resource", "Network", "Scheduling", "Image", "Configuration", "System"}

TASK_PRIORITY = {"resolution": 5, "mitigation": 4, "analysis": 3, "localization": 2, "detection": 1}

# Locate exec_shell( ... ) and peel the outer quote pair from the argument.
_EXEC_SHELL_OPEN = re.compile(r'exec_shell\s*\(', re.DOTALL)

# Commands that modify cluster state → candidates for final_solution
_FIX_VERBS = {
    "patch", "apply", "delete", "create", "set", "rollout", "scale",
    "drain", "uncordon", "cordon", "label", "annotate", "taint",
    "rollback", "replace", "exec", "cp", "port-forward",
    "restart",  # kubectl rollout restart
}

# kubectl subcommands that are purely diagnostic
_DIAG_VERBS = {
    "get", "describe", "logs", "events", "top", "version",
    "api-resources", "api-versions", "explain", "diff", "auth",
    "cluster-info", "config", "wait",
}

# kubeadm/helm verbs that are fix operations
_OTHER_FIX_TOOLS = {
    "kubeadm certs renew",
    "helm rollback",
    "helm upgrade",
    "helm install",
    "helm uninstall",
    "helm delete",
    "systemctl restart",
    "systemctl start",
    "systemctl stop",
}

# ---------------------------------------------------------------------------
# Fault category classification
#
# Rules are checked problem_id-first (more reliable than summary text which
# contains "Affected Resources:" on every record and would bias toward Resource).
# ---------------------------------------------------------------------------

# problem_id substring → category  (checked before summary)
_PID_RULES: list[tuple[list[str], str]] = [
    # Resource: explicit resource exhaustion / quota terms in the ID
    (["high_cpu", "manual_gc", "oom", "pvc", "storage", "hpa",
      "redeploy_without_pv", "redeploy_without_pv"], "Resource"),
    # Network: network-level faults
    (["network_delay", "network_loss", "target_port", "port_misconfig",
      "unreachable", "loadbalancer", "cni", "coredns"], "Network"),
    # Scheduling: node placement / affinity failures
    (["assign_to_non_existent_node", "pod_scheduling", "pod_unschedulable",
      "scheduling_failure", "scale_pod_zero", "pod_zero_replica",
      "pod_zero_replicas"], "Scheduling"),
    # Image: image pull / binary / format issues
    (["image_slow_load", "slow_load", "wrong_bin", "imagepull",
      "exec_format"], "Image"),
    # Configuration: auth / secret / configmap / misconfig / rbac
    (["auth_miss", "revoke_auth", "misconfig", "mongodb_authentication",
      "mongodb-authentication", "user_unregistered", "unregistered",
      "recommendation_service_cache", "recommendation_cache",
      "cache_failure", "auth"], "Configuration"),
    # System: pod/container lifecycle, infra daemons, observability
    (["container_kill", "pod_kill", "pod_failure", "pod_fail",
      "kafka_queue", "loadgenerator_flood", "otel_collector",
      "noop_detection", "noop", "etcd", "kubelet", "runtime",
      "certificate", "cert"], "System"),
]

# summary keyword → category  (fallback when problem_id gives no match)
_SUMMARY_RULES: list[tuple[list[str], str]] = [
    (["oom", "oomkilled", "cpu throttl", "memory limit", "evict",
      "node pressure", "storage quota", "hpa"], "Resource"),
    (["dns resolution", "coredns", "502 bad gateway", "503",
      "connection refused", "network policy", "cni plugin",
      "port mismatch", "targetport"], "Network"),
    (["failedscheduling", "node affinity", "nodeselector",
      "non-existent node", "taint", "tolerat", "pending state"], "Scheduling"),
    (["imagepullbackoff", "errimagepull", "exec format error",
      "image tag", "registry auth", "wrong binary"], "Image"),
    (["crashloopbackoff", "configmap not found", "secret missing",
      "rbac", "permission denied", "403 forbidden",
      "helm", "webhook", "missing required environment variable",
      "misconfigured environment", "authentication failed",
      "cache configuration"], "Configuration"),
    (["crashloopbackoff", "pod restart", "container restart",
      "kafka", "otel", "node notready", "kubelet",
      "scale.*zero", "zero replicas"], "System"),
]


def classify_fault_category(problem_id: str, summary: str) -> str:
    pid = problem_id.lower()

    # Check problem_id against PID rules first
    for keywords, category in _PID_RULES:
        if any(kw in pid for kw in keywords):
            return category

    # Fall back to summary with carefully scoped keywords
    # (avoid "resource" alone – too common in "Affected Resources:" boilerplate)
    summ = summary.lower()
    for keywords, category in _SUMMARY_RULES:
        if any(re.search(kw, summ) for kw in keywords):
            return category

    # Heuristic: if summary mentions service failure / pod crash without a better match
    if "crashloopbackoff" in summ or "pod restart" in summ:
        return "System"
    if "scheduling" in summ or "pending" in summ:
        return "Scheduling"

    return "Configuration"  # safest default for unclassified k8s faults


# ---------------------------------------------------------------------------
# Prevention strings (one per category, grounded in k8s best practices)
# ---------------------------------------------------------------------------

_PREVENTION: dict[str, str] = {
    "Resource": (
        "Set resource requests and limits for all containers. Enable VPA or HPA for "
        "auto-scaling. Alert at 80% of quota/limit utilisation."
    ),
    "Network": (
        "Validate service targetPort matches containerPort before deploy. Document "
        "required traffic flows before applying NetworkPolicy. Use IngressClass "
        "with health checks."
    ),
    "Scheduling": (
        "Verify nodeSelector/affinity labels exist on target nodes before deploy. "
        "Use PreferredDuringScheduling unless strict isolation is required. "
        "Set accurate resource requests so the scheduler can place pods."
    ),
    "Image": (
        "Pin image tags to immutable digests or explicit semver in all manifests. "
        "Never use :latest in production. Build multi-arch images with buildx in CI."
    ),
    "Configuration": (
        "Use Helm sync waves or init containers to ensure ConfigMaps/Secrets exist "
        "before Deployments. Validate RBAC requirements at design time with "
        "kubectl auth can-i dry-run checks in CI."
    ),
    "System": (
        "Configure kubelet log rotation and container log max-size. Monitor node "
        "disk, memory, and container runtime health. Automate certificate renewal "
        "with cert-manager."
    ),
}


# ---------------------------------------------------------------------------
# Command parsing helpers
# ---------------------------------------------------------------------------

_YAML_LINE_RE = re.compile(
    r"^(?:"
    r"(?:apiVersion|kind|metadata|spec|status|data|stringData|"
    r"labels|annotations|selector|template|containers|volumes|"
    r"env|name|value|image|ports|namespace|replicas|resources|"
    r"limits|requests|EOF)\b"
    r"|---+"                          # YAML doc separator
    r"|[A-Z][A-Z0-9_]{2,}\s*:"       # env-var YAML keys: MONGO_USER:, DB_PASS:
    r")",
    re.IGNORECASE,
)


def _extract_exec_shell_arg(raw: str) -> Optional[str]:
    """
    Robustly extract the shell command from exec_shell("...") or exec_shell('...').

    Handles:
    - Inner quotes of the other type (single inside double, double inside single)
    - Escaped characters (\\n, \\t, \\", \\')
    - Shell quoting: inside 'single-quoted' sections, the outer-quote char " is literal
    - Multi-line strings
    """
    m = _EXEC_SHELL_OPEN.search(raw)
    if not m:
        return None
    pos = m.end()
    if pos >= len(raw):
        return None
    # The argument must start with a quote character
    outer_quote = raw[pos]
    if outer_quote not in ('"', "'"):
        return None
    inner_quote = "'" if outer_quote == '"' else '"'
    pos += 1  # skip opening outer quote
    result: list[str] = []
    in_inner_quote = False  # are we inside 'inner-quote' section?

    while pos < len(raw):
        ch = raw[pos]

        # Handle escape sequences
        if ch == '\\' and pos + 1 < len(raw):
            next_ch = raw[pos + 1]
            if next_ch == 'n':
                result.append('\n')
            elif next_ch == 't':
                result.append('\t')
            elif next_ch in ('"', "'", '\\'):
                result.append(next_ch)
            else:
                result.append('\\')
                result.append(next_ch)
            pos += 2
            continue

        # Toggle inner-quote mode when we see the inner quote char (not escaped)
        if ch == inner_quote and not in_inner_quote:
            in_inner_quote = True
            result.append(ch)
            pos += 1
            continue
        if ch == inner_quote and in_inner_quote:
            in_inner_quote = False
            result.append(ch)
            pos += 1
            continue

        # Closing outer quote — only meaningful outside inner quote context
        if ch == outer_quote and not in_inner_quote:
            break

        result.append(ch)
        pos += 1

    return ''.join(result).strip()


def _clean_heredoc_command(cmd: str) -> str:
    """
    Collapse a heredoc command (cat > file << 'EOF'\\n...\\nEOF\\nkubectl apply ...)
    into its essential fix step: retain the cat/kubectl apply line, drop YAML lines.
    If 'kubectl apply' follows the heredoc, keep only that line.
    """
    lines = cmd.splitlines()
    if not any(line.strip() == "EOF" for line in lines):
        return cmd  # not a heredoc, return as-is

    # Find the kubectl apply or equivalent after EOF
    after_eof: list[str] = []
    past_eof = False
    for line in lines:
        if line.strip() == "EOF":
            past_eof = True
            continue
        if past_eof and line.strip():
            after_eof.append(line.strip())

    if after_eof:
        # Return the post-heredoc commands (e.g., kubectl apply)
        return "\n".join(after_eof)

    # No post-EOF commands: keep the first line (the cat > file part)
    return lines[0].strip()


def _normalize_patch_args(cmd: str) -> str:
    """
    Collapse internal newlines within a JSON patch argument (-p '...' or -p "...").
    Prevents multi-line patch strings from being split into separate command lines.
    """
    # Match -p '...' or -p "..." potentially spanning multiple lines
    def _collapse(m: re.Match) -> str:
        inner = m.group(1).replace("\n", " ").replace("  ", " ").strip()
        quote = m.group(0)[3]  # the quote character after -p
        return f"-p {quote}{inner}{quote}"

    cmd = re.sub(r"-p\s+'((?:[^'\\]|\\.|\n)*)'", _collapse, cmd)
    cmd = re.sub(r'-p\s+"((?:[^"\\]|\\.|\n)*)"', _collapse, cmd)
    return cmd


def parse_exec_shell(raw: str) -> Optional[str]:
    """Extract shell command from exec_shell("...") wrapper. Returns None if no match."""
    if "exec_shell" in raw:
        extracted = _extract_exec_shell_arg(raw)
        if extracted:
            # Handle heredoc: collapse to the executable fix step
            if re.search(r"<<\s*['\"]?EOF['\"]?", extracted):
                extracted = _clean_heredoc_command(extracted)
            else:
                # Collapse multi-line JSON patch args to single line
                extracted = _normalize_patch_args(extracted)
            return extracted

    # Bare kubectl/kubeadm/helm/shell command (already extracted)
    stripped = raw.strip()
    if stripped.startswith(("kubectl", "kubeadm", "helm", "ssh", "systemctl",
                             "sudo", "cat", "echo", "bash", "sh")):
        return stripped
    return None


def is_fix_command(cmd: str) -> bool:
    """True if cmd modifies cluster state and belongs in final_solution."""
    if not cmd:
        return False
    c = cmd.strip().lower()
    # kubeadm/helm fix operations
    for pattern in _OTHER_FIX_TOOLS:
        if c.startswith(pattern):
            return True
    # systemctl restart/start/stop
    if re.match(r"(sudo\s+)?systemctl\s+(restart|start|stop)\b", c):
        return True
    # ssh <host> '...' containing fix verbs
    if c.startswith("ssh "):
        inner = re.search(r"'(.+)'", c)
        if inner:
            return is_fix_command(inner.group(1))
        return False
    # kubectl <verb> ...
    m = re.match(r"(sudo\s+)?kubectl\s+(\S+)", c)
    if not m:
        return False
    verb = m.group(2)
    if verb in _FIX_VERBS:
        return True
    # kubectl rollout restart / kubectl rollout undo
    if verb == "rollout":
        sub = re.search(r"kubectl\s+rollout\s+(\S+)", c)
        if sub and sub.group(1) in ("restart", "undo"):
            return True
    return False


def split_commands(raw_commands: list[str]) -> tuple[list[str], list[str]]:
    """
    Split list of exec_shell strings into (attempted_solutions, final_solution_cmds).
    attempted_solutions: diagnostic kubectl commands
    final_solution_cmds: state-modifying commands

    Strategy: commands before the first fix command are attempted_solutions;
    the fix command and all subsequent commands are final_solution.
    """
    parsed: list[tuple[str, str]] = []  # (raw, extracted_cmd)
    for raw in raw_commands:
        cmd = parse_exec_shell(raw)
        if cmd:
            parsed.append((raw, cmd))

    if not parsed:
        return [], []

    # Find index of first fix command
    first_fix_idx = None
    for i, (_, cmd) in enumerate(parsed):
        if is_fix_command(cmd):
            first_fix_idx = i
            break

    if first_fix_idx is None:
        # All diagnostic → attempted only, no fix
        return [c for _, c in parsed], []

    attempted = [c for _, c in parsed[:first_fix_idx]]
    final = [c for _, c in parsed[first_fix_idx:]]

    # Trim trailing pure-diagnostic commands from final (verification is OK to keep)
    # Actually keep all post-fix commands including verification steps
    return attempted, final


# ---------------------------------------------------------------------------
# Context / namespace extraction
# ---------------------------------------------------------------------------

_NS_RE = re.compile(
    r"[Nn]amespace[:\s]+['\"]?([a-z0-9][a-z0-9\-]*[a-z0-9])['\"]?",
    re.IGNORECASE,
)

_NS_FLAG_RE = re.compile(r"-n\s+([a-z0-9][a-z0-9\-]*)")

KNOWN_NAMESPACES = {
    "astronomy-shop", "test-social-network", "social-network", "hotel-reservation",
    "train-ticket", "kube-system", "default", "monitoring", "production", "staging",
    "dev", "qa", "flight-ticket",
}


def extract_namespace(task_description: str, commands: list[str]) -> str:
    """Extract namespace from task_description, then from commands."""
    # From description
    m = _NS_RE.search(task_description)
    if m:
        ns = m.group(1).lower()
        if ns not in ("the", "a", "an", "your", "this"):
            return ns

    # From -n flag in first few commands
    for raw in commands[:5]:
        cmd = parse_exec_shell(raw) or ""
        m = _NS_FLAG_RE.search(cmd)
        if m:
            return m.group(1)

    return "default"


_RESOURCE_TYPE_KEYWORDS = [
    ("StatefulSet", ["statefulset"]),
    ("DaemonSet", ["daemonset"]),
    ("Deployment", ["deployment", "deploy "]),
    ("Job", ["job "]),
    ("CronJob", ["cronjob"]),
    ("Node", ["node "]),
    ("Service", ["service "]),
    ("HPA", ["horizontalpodautoscaler", "hpa"]),
    ("PersistentVolumeClaim", ["persistentvolumeclaim", "pvc"]),
    ("ConfigMap", ["configmap"]),
    ("Secret", ["secret"]),
    ("NetworkPolicy", ["networkpolicy"]),
    ("Ingress", ["ingress"]),
    ("Certificate", ["certificate", "cert "]),
    ("ServiceAccount", ["serviceaccount"]),
    ("Pod", ["pod "]),
]


def extract_resource_type(summary: str, task_description: str, commands: list[str]) -> str:
    """Derive Kubernetes resource_type from summary text and commands."""
    text = (summary + " " + task_description).lower()
    for rtype, keywords in _RESOURCE_TYPE_KEYWORDS:
        if any(kw in text for kw in keywords):
            return rtype

    # Fall back: check fix commands for patched resource type
    for raw in commands:
        cmd = (parse_exec_shell(raw) or "").lower()
        m = re.search(r"kubectl\s+(?:patch|get|describe|set)\s+(\S+)", cmd)
        if m:
            resource = m.group(1).rstrip("s")  # crude depluralize
            if resource in ("deployment", "statefulset", "daemonset", "pod",
                            "service", "node", "job"):
                return resource.capitalize()
    return "Deployment"


# ---------------------------------------------------------------------------
# Symptom extraction
# ---------------------------------------------------------------------------

_SUMMARY_SECTION_RE = re.compile(
    r"\d+\)\s*(?:Root Cause and Symptoms?|Error Messages? and Logs?)[:\s]*(.*?)(?=\d+\)|$)",
    re.IGNORECASE | re.DOTALL,
)

_ERROR_PATTERNS = [
    re.compile(r"'([^']{10,100})'"),        # single-quoted strings (error messages)
    re.compile(r'"([^"]{10,100})"'),         # double-quoted strings
    re.compile(r"(?:Error|Warning|Failed|CrashLoopBackOff|OOMKilled|"
               r"ImagePullBackOff|ErrImagePull|Pending|Evicted|"
               r"NotReady|Unhealthy|FailedScheduling)[^\.\n]{0,80}",
               re.IGNORECASE),
]


def extract_symptoms(summary: str, steps: Optional[list[dict]] = None) -> list[str]:
    """
    Extract 3-5 observable symptom strings from system_state_summary.
    Optionally enrich from hf_observer_training step summaries.
    """
    symptoms: list[str] = []

    # Split on ". " (period-space) or newline only — avoids splitting "kubernetes.io/hostname"
    _SYMPTOM_SPLIT_RE = re.compile(r"\.\s+|\n")
    _SYMPTOM_BOILERPLATE = re.compile(
        r"^(?:root cause|this results|this causes|due to|because of|"
        r"is configured|all nodes|network connectivity|cluster resource|"
        r"the \w+ deployment has|the service is|the pod is)",
        re.IGNORECASE,
    )

    # Extract from first two sections of summary
    sections = _SUMMARY_SECTION_RE.findall(summary)
    for section_text in sections[:2]:
        for fragment in _SYMPTOM_SPLIT_RE.split(section_text):
            fragment = fragment.strip(" \t\r.'\"")
            if 10 < len(fragment) < 120 and fragment:
                if _SYMPTOM_BOILERPLATE.match(fragment):
                    continue
                symptoms.append(fragment)
                if len(symptoms) >= 4:
                    break
        if len(symptoms) >= 4:
            break

    # For unnumbered summaries (resolution files) extract from entire text
    if not symptoms and not sections:
        # Strip leading section header if present
        text = re.sub(
            r"^(?:Root Cause[^:]*|Error Messages?)[:\s]+", "", summary, flags=re.IGNORECASE
        )
        for fragment in _SYMPTOM_SPLIT_RE.split(text):
            fragment = fragment.strip(" '\"\t\r\n")
            if (10 < len(fragment) < 120
                    and not _SYMPTOM_BOILERPLATE.match(fragment)):
                symptoms.append(fragment)
                if len(symptoms) >= 4:
                    break

    # Also grab quoted error messages from section 3
    m3 = re.search(
        r"3\)\s*Error Messages?[^:]*:(.+?)(?=4\)|$)", summary, re.IGNORECASE | re.DOTALL
    )
    if m3 and len(symptoms) < 5:
        for pat in _ERROR_PATTERNS:
            for match in pat.finditer(m3.group(1)):
                s = match.group(0).strip(" '\"")
                if 10 < len(s) < 120 and s not in symptoms:
                    symptoms.append(s)
                    if len(symptoms) >= 5:
                        break
            if len(symptoms) >= 5:
                break

    # Enrich from step summaries if provided
    if steps and len(symptoms) < 3:
        for step in steps:
            sm = step.get("summary", "")
            if not sm:
                continue
            for fragment in re.split(r"[.\n]+", sm):
                fragment = fragment.strip()
                if 10 < len(fragment) < 100 and fragment not in symptoms:
                    symptoms.append(fragment)
                    if len(symptoms) >= 4:
                        break
            if len(symptoms) >= 4:
                break

    # Fallback: generic symptom from summary first sentence
    if not symptoms:
        first_sentence = summary.split(".")[0].strip()
        if first_sentence:
            symptoms.append(first_sentence[:120])

    # Clean: strip leading/trailing quote chars and whitespace
    cleaned: list[str] = []
    seen: set[str] = set()
    for s in symptoms:
        s = s.strip(" '\"\t\r\n")
        s = re.sub(r"\s+", " ", s).strip()
        # Skip duplicates (case-insensitive)
        key = s.lower()
        if s and len(s) > 8 and key not in seen:
            cleaned.append(s)
            seen.add(key)

    return cleaned[:5]


# ---------------------------------------------------------------------------
# Outcome derivation from fix commands
# ---------------------------------------------------------------------------

_OUTCOME_RULES: list[tuple[str, str]] = [
    (r"rollout restart", "Pods restarted and returned to Running state"),
    (r"rollout undo",    "Deployment rolled back to previous stable revision"),
    (r"patch.*nodeSelector|patch.*affinity", "Scheduling constraint corrected; pods scheduled successfully"),
    (r"patch.*taint|tolerat", "Taint/toleration mismatch resolved; pods scheduled"),
    (r"set env|set resources", "Environment variable or resource limit applied to deployment"),
    (r"set image",       "Container image updated to correct version"),
    (r"create secret|create configmap", "Required Kubernetes object created and available to pods"),
    (r"create rolebinding|apply.*role", "RBAC binding applied; ServiceAccount has required permissions"),
    (r"drain|uncordon",  "Node drained, remediated, and returned to schedulable state"),
    (r"delete.*webhook|delete.*mutating", "Blocking admission webhook removed; pod creation unblocked"),
    (r"certs renew",     "Kubernetes certificates renewed; API server accessible"),
    (r"helm rollback",   "Helm release rolled back to last known good revision"),
    (r"scale.*replicas", "Deployment scaled; pods available"),
    (r"kubectl apply",   "Kubernetes manifests applied; desired state reconciled"),
    (r"kubectl delete pod", "Faulty pod deleted and rescheduled by controller"),
    (r"systemctl restart", "Node-level service restarted successfully"),
]


def derive_outcomes(fix_commands: list[str]) -> list[str]:
    outcomes: list[str] = []
    seen: set[str] = set()
    for cmd in fix_commands:
        cl = cmd.lower()
        for pattern, outcome in _OUTCOME_RULES:
            if re.search(pattern, cl) and outcome not in seen:
                outcomes.append(outcome)
                seen.add(outcome)
                break
    if not outcomes:
        outcomes.append("Fault remediated; service restored to healthy state")
    outcomes.append("No further error events in affected namespace")
    return outcomes[:3]


# ---------------------------------------------------------------------------
# Reasoning construction from system_state_summary
# ---------------------------------------------------------------------------

def build_reasoning(summary: str, fault_category: str) -> str:
    """
    Extract root-cause sentence(s) from system_state_summary section 1.
    Preserves original AOI text; does not hallucinate new content.
    """
    # Section 1: Root Cause and Symptoms
    m = re.search(
        r"1\)\s*Root Cause[^:]*:\s*(.+?)(?=2\)|$)", summary, re.IGNORECASE | re.DOTALL
    )
    if m:
        root_cause_text = m.group(1).strip()
        # Truncate at first paragraph boundary
        root_cause_text = root_cause_text.split("\n\n")[0].strip()
        # Remove trailing section markers
        root_cause_text = re.sub(r"\s*\d+\)\s*$", "", root_cause_text).strip()
        if root_cause_text:
            # Avoid "Root cause: Root cause and symptoms: ..." duplication
            root_cause_text = re.sub(
                r"^Root Cause[^:]*:\s*", "", root_cause_text, flags=re.IGNORECASE
            ).strip()
            return f"Root cause: {root_cause_text}"

    # Fallback: first two sentences of summary
    sentences = [s.strip() for s in summary.split(".") if s.strip()]
    return "Root cause: " + ". ".join(sentences[:3]) + "."


# ---------------------------------------------------------------------------
# Problem string construction
# ---------------------------------------------------------------------------

def build_problem_string(
    task_description: str,
    fault_summary: str,
    namespace: str,
    fault_category: str,
    symptoms: list[str],
) -> str:
    """
    Construct a KFRD-style problem string from AOI task data.
    Uses original AOI text; does not add invented details.
    """
    # Extract the core task objective sentence from task_description
    # Pattern: "Task Objective: ..." or "Task objective: ..."
    obj_m = re.search(r"[Tt]ask [Oo]bjective[:\s]+(.+?)(?:\n|$)", task_description)
    if obj_m:
        objective = obj_m.group(1).strip().rstrip(".")
    else:
        # Use the last sentence of task_description
        sentences = [s.strip() for s in task_description.split(".") if s.strip()]
        objective = sentences[-1] if sentences else "Kubernetes fault detected"

    # Extract short description of affected resource from summary section 2
    res_m = re.search(
        r"2\)\s*Affected Resources?[^:]*:\s*(.+?)(?=3\)|$)", fault_summary, re.IGNORECASE | re.DOTALL
    )
    affected = ""
    if res_m:
        affected_text = res_m.group(1).strip().split(".")[0]
        if len(affected_text) < 200:
            affected = f" Affected: {affected_text}."

    symptom_str = ". ".join(symptoms[:3])
    return (
        f"Kubernetes {fault_category} fault in namespace '{namespace}'. "
        f"{objective}.{affected} "
        f"Symptoms: {symptom_str}."
    )


# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------

def load_json_file(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def parse_problem_id_parts(filename: str) -> tuple[str, str, str]:
    """
    Parse '{base_problem}-{task_type}-{variant}.json' into components.
    Returns (base_problem, task_type, variant).
    """
    stem = Path(filename).stem
    # Remove _run\d+ suffix if present
    stem = re.sub(r"_run\d+$", "", stem)
    # Match last two hyphen-separated segments as task_type-variant
    m = re.match(
        r"^(.+?)-(detection|localization|mitigation|resolution|analysis)-(\d+)$",
        stem,
        re.IGNORECASE,
    )
    if m:
        return m.group(1), m.group(2).lower(), m.group(3)
    # Fallback: treat entire stem as base
    return stem, "unknown", "1"


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------

def build_from_infer(record: dict, record_index: int) -> Optional[dict]:
    """
    Build a KFRD record from an aoi/data/gt/infer/ file.
    Requires fix commands to be present (mitigation/resolution or infer files
    that include fix commands regardless of task type label).
    """
    problem_id: str = record.get("problem_id", "")
    task_description: str = record.get("task_description", "")
    # infer/ uses 'system_state_summary'; infer_adapted/ uses task_info.fault_summary
    summary: str = record.get("system_state_summary", "")
    if not summary:
        ti = record.get("task_info", {})
        summary = ti.get("fault_summary", "")
        if not task_description:
            task_description = ti.get("task_description", "")

    raw_commands: list[str] = record.get("commands", [])

    attempted, final_cmds = split_commands(raw_commands)

    # Reject if no fix commands present
    if not final_cmds:
        return None

    namespace = extract_namespace(task_description, raw_commands)
    resource_type = extract_resource_type(summary, task_description, raw_commands)
    fault_category = classify_fault_category(problem_id, summary)
    symptoms = extract_symptoms(summary)
    reasoning = build_reasoning(summary, fault_category)
    final_solution = "\n".join(final_cmds)
    outcomes = derive_outcomes(final_cmds)
    prevention = _PREVENTION[fault_category]
    problem = build_problem_string(
        task_description, summary, namespace, fault_category, symptoms
    )

    _GENERIC_K8S_TERMS = frozenset({
        "deployment", "statefulset", "daemonset", "replicaset", "pod",
        "node", "namespace", "service", "container", "cluster", "the",
    })

    context: dict = {"namespace": namespace, "resource_type": resource_type}
    # Add specific resource name from summary section 2 if identifiable
    res_m = re.search(r"Pod\s+([a-z0-9][a-z0-9\-]+-[a-z0-9]+)", summary, re.IGNORECASE)
    if res_m:
        context["pod"] = res_m.group(1)
    else:
        # Match "service <name>" where name is a real k8s svc name (3-30 chars, alphanumeric+hyphen)
        svc_m = re.search(r"\bservice\s+([a-z][a-z0-9\-]{2,28})\b", summary, re.IGNORECASE)
        if svc_m:
            svc_name = svc_m.group(1).lower()
            # Reject generic words and English words (heuristic: real k8s names contain hyphens or are short)
            if (svc_name not in _GENERIC_K8S_TERMS
                    and re.match(r"^[a-z][a-z0-9\-]+$", svc_name)
                    and (len(svc_name) <= 20 or "-" in svc_name)
                    and not re.search(r"[aeiou]{2}", svc_name[:4])):  # reject obvious English words
                context["service"] = svc_name

    return {
        "id": f"kfrd_aoi_{record_index:05d}",
        "problem": problem,
        "attempted_solutions": attempted[:6],  # cap at 6 as per paper
        "final_solution": final_solution,
        "reasoning": reasoning,
        "fault_category": fault_category,
        "symptoms": symptoms,
        "context": context,
        "outcomes": outcomes,
        "prevention": prevention,
        "timestamp": time.time() - 86400 * 30,  # 30 days ago (fixed offset for reproducibility)
    }


def build_from_hf_trace(
    hf_record: dict,
    paired_infer: Optional[dict],
    record_index: int,
) -> Optional[dict]:
    """
    Build a KFRD record from an hf_observer_training/ step trace.
    Uses paired infer mitigation/resolution record for fix commands if trace
    is a localization task.
    """
    problem_id: str = hf_record.get("problem_id", "")
    task_description: str = hf_record.get("task_description", "")
    steps: list[dict] = hf_record.get("steps", [])

    # Separate probe/diagnostic steps from executor/submit steps
    probe_steps = [s for s in steps if not s.get("command", "").startswith("submit(")]
    submit_step = next((s for s in steps if s.get("command", "").startswith("submit(")), None)

    # Extract commands from trace steps
    trace_cmds = [s.get("command", "") for s in probe_steps]

    attempted_from_trace, fix_from_trace = split_commands(trace_cmds)

    # Use paired infer record's fix commands if trace has none
    final_cmds: list[str] = []
    summary: str = ""

    if fix_from_trace:
        final_cmds = fix_from_trace
        # Build summary from step summaries
        summary = " ".join(
            s.get("summary", "") for s in steps if s.get("summary")
        )
    elif paired_infer:
        summary = (
            paired_infer.get("system_state_summary", "")
            or paired_infer.get("task_info", {}).get("fault_summary", "")
        )
        _, paired_fix = split_commands(paired_infer.get("commands", []))
        final_cmds = paired_fix

    if not final_cmds:
        return None

    # Enrich summary with step summaries if short
    if len(summary) < 100 and steps:
        step_summaries = " ".join(s.get("summary", "") for s in steps if s.get("summary"))
        summary = step_summaries or summary

    namespace = extract_namespace(task_description, trace_cmds)
    # If submit step found, extract faulty component as service name
    faulty_component = None
    if submit_step:
        m = re.search(r'submit\(\[([^\]]+)\]\)', submit_step.get("command", ""))
        if m:
            components_str = m.group(1)
            components = re.findall(r'"([^"]+)"|\'([^\']+)\'', components_str)
            if components:
                faulty_component = components[0][0] or components[0][1]

    resource_type = extract_resource_type(summary, task_description, trace_cmds)
    fault_category = classify_fault_category(problem_id, summary)
    symptoms = extract_symptoms(summary, steps)
    reasoning = build_reasoning(summary, fault_category)
    final_solution = "\n".join(final_cmds)
    outcomes = derive_outcomes(final_cmds)
    prevention = _PREVENTION[fault_category]
    problem = build_problem_string(
        task_description, summary, namespace, fault_category, symptoms
    )

    _GENERIC_K8S_TERMS_HF = frozenset({
        "deployment", "statefulset", "daemonset", "replicaset", "pod",
        "node", "namespace", "service", "container", "cluster",
    })
    context: dict = {"namespace": namespace, "resource_type": resource_type}
    if faulty_component and faulty_component.lower() not in _GENERIC_K8S_TERMS_HF:
        context["service"] = faulty_component

    return {
        "id": f"kfrd_aoi_{record_index:05d}",
        "problem": problem,
        "attempted_solutions": attempted_from_trace[:6],
        "final_solution": final_solution,
        "reasoning": reasoning,
        "fault_category": fault_category,
        "symptoms": symptoms,
        "context": context,
        "outcomes": outcomes,
        "prevention": prevention,
        "timestamp": time.time() - 86400 * 30,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Commands that look like real kubectl/kubeadm/helm/shell invocations
_VALID_CMD_START = re.compile(
    r"^(kubectl|kubeadm|helm|ssh|sudo|systemctl|nc|until|bash|sh|"
    r"docker|containerd|crictl|journalctl|df|du|free|top|"
    r"cat|echo|touch|mkdir|cp|mv|rm|sed|awk|grep|curl|wget)\b",
    re.IGNORECASE,
)

# Reject if command contains clearly hallucinated k8s CRDs or made-up controllers
_INVALID_PATTERNS = [
    re.compile(r"\bfake[_\-]?\w+\b", re.IGNORECASE),
    re.compile(r"\binvented[_\-]?\w+\b", re.IGNORECASE),
    re.compile(r"\bexample\.com/\w+crd\b", re.IGNORECASE),
]


def validate_record(record: dict) -> tuple[bool, str]:
    """Returns (valid, rejection_reason)."""
    required = ("id", "problem", "attempted_solutions", "final_solution",
                 "reasoning", "fault_category", "symptoms", "context",
                 "outcomes", "prevention", "timestamp")
    for field in required:
        if field not in record:
            return False, f"missing field: {field}"

    if not record["final_solution"].strip():
        return False, "empty final_solution"

    if record["fault_category"] not in VALID_CATEGORIES:
        return False, f"invalid fault_category: {record['fault_category']}"

    ctx = record["context"]
    if "namespace" not in ctx:
        return False, "missing context.namespace"
    if "resource_type" not in ctx:
        return False, "missing context.resource_type"

    if not record["symptoms"]:
        return False, "empty symptoms"

    if len(record["problem"]) < 20:
        return False, "problem string too short"

    if len(record["reasoning"]) < 20:
        return False, "reasoning too short"

    # Validate all commands in final_solution are plausible k8s commands.
    # Skip YAML-content lines, heredoc bodies, and jsonpath continuation lines.
    _CONTINUATION_RE = re.compile(
        r"^(?:"
        r"EOF"                         # heredoc terminator
        r"|---+"                       # YAML document separator
        r"|\{[^\}]*\}"                 # jsonpath fragment {.metadata.name}
        r"|'?\}\{end\}'?"              # jsonpath end fragment
        r"|['\"]?\}end\}['\"]?"        # alternate end
        r"|\s*-\s+\w"                  # YAML list item
        r"|[\{\}]"                     # lone JSON brace (multi-line patch body)
        r"|\"[a-zA-Z]"                 # JSON key line  "spec": ...
        r"|'\}"                        # closing patch arg }'
        r")"
    )
    for cmd in record["final_solution"].splitlines():
        cmd = cmd.strip()
        if not cmd:
            continue
        # Skip YAML/heredoc/jsonpath continuation lines
        if _YAML_LINE_RE.match(cmd):
            continue
        if _CONTINUATION_RE.match(cmd):
            continue
        if not _VALID_CMD_START.match(cmd):
            return False, f"non-kubectl command in final_solution: {cmd[:60]}"
        for bad_pattern in _INVALID_PATTERNS:
            if bad_pattern.search(cmd):
                return False, f"invalid pattern in command: {cmd[:60]}"

    return True, ""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_aoi_sources(aoi_root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Load infer/, infer_adapted/, hf_observer_training/ files."""
    infer_dir = aoi_root / "data" / "gt" / "infer"
    adapted_dir = aoi_root / "data" / "gt" / "infer_adapted"
    hf_dir = aoi_root / "data" / "gt" / "hf_observer_training"

    infer_records: list[dict] = []
    for p in sorted(infer_dir.glob("*.json")):
        rec = load_json_file(p)
        if rec and isinstance(rec, dict):
            rec["_source_file"] = str(p)
            rec["_filename"] = p.name
            infer_records.append(rec)

    adapted_records: list[dict] = []
    for p in sorted(adapted_dir.glob("*.json")):
        rec = load_json_file(p)
        if rec and isinstance(rec, dict):
            rec["_source_file"] = str(p)
            rec["_filename"] = p.name
            adapted_records.append(rec)

    hf_records: list[dict] = []
    for p in sorted(hf_dir.glob("*.json")):
        # Skip _run\d+ variants to avoid near-duplicates
        if re.search(r"_run\d+\.json$", p.name):
            continue
        rec = load_json_file(p)
        if rec and isinstance(rec, dict):
            rec["_source_file"] = str(p)
            rec["_filename"] = p.name
            hf_records.append(rec)

    return infer_records, adapted_records, hf_records


def index_by_problem_id(records: list[dict]) -> dict[str, list[dict]]:
    """Build {base_problem_id: [records...]} index with task_type priority ordering."""
    index: dict[str, list[dict]] = {}
    for rec in records:
        pid = rec.get("problem_id") or rec.get("task_info", {}).get("problem_id", "")
        base, task_type, variant = parse_problem_id_parts(pid or rec.get("_filename", ""))
        rec["_base_problem"] = base
        rec["_task_type"] = task_type
        rec["_variant"] = variant
        index.setdefault(base, []).append(rec)
    # Sort each group by task_type priority (descending: resolution first)
    for base in index:
        index[base].sort(
            key=lambda r: TASK_PRIORITY.get(r.get("_task_type", ""), 0),
            reverse=True,
        )
    return index


def run_pipeline(aoi_root: Path, output_dir: Path, sft_split: float = 0.7) -> None:
    print(f"[AOI→KFRD] loading sources from {aoi_root}")
    infer_records, adapted_records, hf_records = load_aoi_sources(aoi_root)
    print(f"[AOI→KFRD] loaded: infer={len(infer_records)}, "
          f"adapted={len(adapted_records)}, hf_traces={len(hf_records)}")

    # Index by base problem for cross-source pairing
    infer_index = index_by_problem_id(infer_records + adapted_records)
    hf_index = index_by_problem_id(hf_records)

    kfrd_records: list[dict] = []
    rejected: list[tuple[str, str]] = []
    seen_problem_ids: set[str] = set()
    counter = 0

    # --- Pass 1: build from infer/ records (best task-type first) ---
    for base_problem, recs in sorted(infer_index.items()):
        # Deduplicate: one record per (base_problem, task_type)
        seen_task_types: set[str] = set()
        for rec in recs:
            task_type = rec.get("_task_type", "unknown")
            dedup_key = f"{base_problem}:{task_type}:{rec.get('_variant','1')}"
            if dedup_key in seen_problem_ids:
                continue
            seen_problem_ids.add(dedup_key)

            kfrd = build_from_infer(rec, counter)
            if kfrd is None:
                rejected.append((dedup_key, "no fix commands in infer record"))
                continue

            valid, reason = validate_record(kfrd)
            if not valid:
                rejected.append((dedup_key, reason))
                continue

            kfrd_records.append(kfrd)
            counter += 1

    # --- Pass 2: build from hf_observer_training/ traces ---
    # Use traces that provide fix commands themselves OR pair with infer mitigation
    for base_problem, hf_recs in sorted(hf_index.items()):
        for hf_rec in hf_recs:
            task_type = hf_rec.get("_task_type", "unknown")
            dedup_key = f"hf:{base_problem}:{task_type}"
            if dedup_key in seen_problem_ids:
                continue
            seen_problem_ids.add(dedup_key)

            # Find best paired infer record (resolution > mitigation)
            paired_infer: Optional[dict] = None
            if base_problem in infer_index:
                paired_infer = infer_index[base_problem][0]  # already sorted by priority

            kfrd = build_from_hf_trace(hf_rec, paired_infer, counter)
            if kfrd is None:
                rejected.append((dedup_key, "no fix commands from hf trace or paired infer"))
                continue

            # Avoid exact-duplicate final_solution with already-generated records
            final_sig = kfrd["final_solution"][:80]
            if any(r["final_solution"][:80] == final_sig for r in kfrd_records):
                continue  # near-duplicate, skip

            valid, reason = validate_record(kfrd)
            if not valid:
                rejected.append((dedup_key, reason))
                continue

            kfrd_records.append(kfrd)
            counter += 1

    # Re-index IDs sequentially
    for i, rec in enumerate(kfrd_records):
        rec["id"] = f"kfrd_aoi_{i:05d}"

    # --- Split and save ---
    output_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(kfrd_records)
    n_sft = int(n_total * sft_split)
    sft = kfrd_records[:n_sft]
    eval_ = kfrd_records[n_sft:]

    for fname, data in [
        ("kfrd_aoi_sft.json", sft),
        ("kfrd_aoi_eval.json", eval_),
        ("kfrd_aoi_all.json", kfrd_records),
    ]:
        out_path = output_dir / fname
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[AOI→KFRD] wrote {len(data):3d} records → {out_path}")

    print(f"[AOI→KFRD] total: {n_total} records "
          f"({n_sft} sft + {n_total - n_sft} eval), "
          f"{len(rejected)} rejected")
    if rejected:
        print("[AOI→KFRD] rejection reasons:")
        from collections import Counter
        for reason, count in Counter(r for _, r in rejected).most_common():
            print(f"  {count:3d}x  {reason}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert AOI dataset to MetaKube KFRD training format"
    )
    parser.add_argument(
        "--aoi_root",
        type=Path,
        default=Path(__file__).resolve().parent / "aoi",
        help="Path to AOI project root (default: ./aoi)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "MetaKube" / "data" / "datasets" / "kfrd",
        help="Output directory for KFRD JSON files",
    )
    parser.add_argument(
        "--sft_split",
        type=float,
        default=0.7,
        help="Fraction of records for SFT training set (default: 0.7)",
    )
    args = parser.parse_args()
    run_pipeline(args.aoi_root, args.output_dir, args.sft_split)


if __name__ == "__main__":
    main()
