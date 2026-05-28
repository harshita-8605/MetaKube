"""Quantitative base-vs-LoRA evaluation for KubeLLM adapters.

This is a lightweight offline eval for small validation runs. It does not use
Ollama and does not require the full MetaKube orchestrator.
"""

from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kubellm.qwen import QwenKubeLLM


@dataclass
class EvalCase:
    case_id: str
    category: str
    query: str
    causal_chains: list[list[str]]
    required_terms: list[str]
    expected_commands: list[str]


CASES = [
    EvalCase(
        case_id="resource_oom",
        category="Resource",
        query="Pod is OOMKilled repeatedly in production. Diagnose root cause and provide kubectl commands.",
        causal_chains=[["MemoryLeak", "OOMKilled", "IncreaseMemLimit"]],
        required_terms=["oomkilled", "memory", "limit", "pod", "kubectl"],
        expected_commands=["kubectl describe pod", "kubectl top pod", "kubectl set resources"],
    ),
    EvalCase(
        case_id="network_dns",
        category="Network",
        query="Services cannot resolve DNS names and CoreDNS pods are crashing. Diagnose and fix.",
        causal_chains=[["CoreDNSCrash", "DNSFailure", "RestartCoreDNS"]],
        required_terms=["dns", "coredns", "service", "logs", "rollout"],
        expected_commands=["kubectl get pods", "kubectl logs", "kubectl rollout restart"],
    ),
    EvalCase(
        case_id="config_configmap",
        category="Configuration",
        query="Pod is in CrashLoopBackOff after deployment because a ConfigMap may be missing.",
        causal_chains=[["ConfigMapMount", "CrashLoopBackOff", "FixConfigMount"]],
        required_terms=["crashloopbackoff", "configmap", "logs", "deployment", "restart"],
        expected_commands=["kubectl logs", "kubectl create configmap", "kubectl rollout restart"],
    ),
    EvalCase(
        case_id="image_pull",
        category="Image",
        query="Deployment fails with ImagePullBackOff for a private registry image.",
        causal_chains=[["RegistryAuth", "ImagePullBackOff", "FixRegistryCreds"]],
        required_terms=["imagepullbackoff", "registry", "secret", "imagepullsecrets", "deployment"],
        expected_commands=["kubectl create secret docker-registry", "kubectl patch deployment", "kubectl rollout status"],
    ),
    EvalCase(
        case_id="scheduling_taint",
        category="Scheduling",
        query="Pod is stuck Pending because nodes have taints and the deployment has no toleration.",
        causal_chains=[["TaintToleration", "PodPending", "AddToleration"]],
        required_terms=["pending", "taint", "toleration", "node", "deployment"],
        expected_commands=["kubectl describe node", "kubectl patch deployment", "kubectl get pod"],
    ),
]


def generate_outputs(args, lora_path: str | None) -> dict[str, str]:
    model = QwenKubeLLM(
        model_id=args.model_id,
        load_in_4bit=args.load_in_4bit,
        lora_path=lora_path,
        max_new_tokens=args.max_new_tokens,
        precision=args.precision,
        device_map=args.device_map,
    )
    outputs = {}
    for case in CASES:
        outputs[case.case_id] = model.generate_analytical(
            query=case.query,
            patterns=[],
            causal_chains=case.causal_chains,
        )
    return outputs


def unload_cuda() -> None:
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def contains_command(output: str, command_prefix: str) -> bool:
    normalized = " ".join(output.lower().split())
    return command_prefix.lower() in normalized


def repetition_ratio(output: str) -> float:
    tokens = re.findall(r"\w+", output.lower())
    if not tokens:
        return 0.0
    return 1.0 - (len(set(tokens)) / len(tokens))


def section_coverage(output: str) -> float:
    lower = output.lower()
    sections = [
        ("root", "cause"),
        ("resolution",),
        ("kubectl",),
        ("prevention",),
    ]
    hits = 0
    for terms in sections:
        if all(term in lower for term in terms):
            hits += 1
    return hits / len(sections)


def score_output(output: str, case: EvalCase) -> dict:
    lower = output.lower()
    commands = re.findall(r"kubectl[^\n`]*", output)
    term_hits = sum(1 for term in case.required_terms if term.lower() in lower)
    command_hits = sum(1 for cmd in case.expected_commands if contains_command(output, cmd))
    sec_cov = section_coverage(output)
    rep = repetition_ratio(output)

    term_score = term_hits / len(case.required_terms)
    command_score = command_hits / len(case.expected_commands)
    command_presence = min(len(commands), 3) / 3
    repetition_penalty = min(rep, 0.60) / 0.60

    total = (
        35.0 * term_score
        + 30.0 * command_score
        + 20.0 * sec_cov
        + 15.0 * command_presence
        - 10.0 * repetition_penalty
    )
    total = max(0.0, min(100.0, total))

    return {
        "score": round(total, 2),
        "term_hits": term_hits,
        "term_total": len(case.required_terms),
        "term_coverage": round(term_score, 3),
        "expected_command_hits": command_hits,
        "expected_command_total": len(case.expected_commands),
        "expected_command_coverage": round(command_score, 3),
        "kubectl_command_count": len(commands),
        "section_coverage": round(sec_cov, 3),
        "repetition_ratio": round(rep, 3),
        "commands": commands[:5],
    }


def summarize(rows: list[dict]) -> dict:
    def avg(key: str, side: str) -> float:
        return round(sum(row[f"{side}_{key}"] for row in rows) / len(rows), 2)

    return {
        "cases": len(rows),
        "base_avg_score": avg("score", "base"),
        "lora_avg_score": avg("score", "lora"),
        "score_delta": round(avg("score", "lora") - avg("score", "base"), 2),
        "base_avg_expected_command_coverage": avg("expected_command_coverage", "base"),
        "lora_avg_expected_command_coverage": avg("expected_command_coverage", "lora"),
        "base_avg_section_coverage": avg("section_coverage", "base"),
        "lora_avg_section_coverage": avg("section_coverage", "lora"),
        "base_avg_repetition_ratio": avg("repetition_ratio", "base"),
        "lora_avg_repetition_ratio": avg("repetition_ratio", "lora"),
        "avg_base_lora_similarity": round(sum(row["base_lora_similarity"] for row in rows) / len(rows), 3),
    }


def write_reports(out_dir: Path, rows: list[dict], summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lora_quant_eval.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    csv_fields = [
        "case_id",
        "category",
        "base_score",
        "lora_score",
        "score_delta",
        "base_expected_command_coverage",
        "lora_expected_command_coverage",
        "base_section_coverage",
        "lora_section_coverage",
        "base_repetition_ratio",
        "lora_repetition_ratio",
        "base_lora_similarity",
    ]
    with open(out_dir / "lora_quant_eval.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in csv_fields})

    with open(out_dir / "lora_quant_eval.md", "w") as f:
        f.write("# LoRA Quantitative Evaluation\n\n")
        f.write("## Summary\n\n")
        for key, value in summary.items():
            f.write(f"- `{key}`: {value}\n")
        f.write("\n## Per-Case Results\n\n")
        f.write("| Case | Category | Base | LoRA | Delta | Cmd Cov Base | Cmd Cov LoRA | Similarity |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['case_id']} | {row['category']} | {row['base_score']} | "
                f"{row['lora_score']} | {row['score_delta']} | "
                f"{row['base_expected_command_coverage']} | {row['lora_expected_command_coverage']} | "
                f"{row['base_lora_similarity']} |\n"
            )
        f.write("\n## Outputs\n\n")
        for row in rows:
            f.write(f"### {row['case_id']}\n\n")
            f.write(f"Prompt: {row['query']}\n\n")
            f.write(f"Base score: `{row['base_score']}`  LoRA score: `{row['lora_score']}`  Delta: `{row['score_delta']}`\n\n")
            f.write("#### Base Output\n\n")
            f.write(row["base_output"].strip() + "\n\n")
            f.write("#### LoRA Output\n\n")
            f.write(row["lora_output"].strip() + "\n\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--lora_path", default="data/models/kubellm")
    parser.add_argument("--output_dir", default="eval_outputs/lora_quant_eval")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args()

    print("[1/3] Generating base outputs...")
    base_outputs = generate_outputs(args, lora_path=None)
    unload_cuda()

    print("[2/3] Generating LoRA outputs...")
    lora_outputs = generate_outputs(args, lora_path=args.lora_path)
    unload_cuda()

    print("[3/3] Scoring outputs...")
    rows = []
    for case in CASES:
        base_output = base_outputs[case.case_id]
        lora_output = lora_outputs[case.case_id]
        base_metrics = score_output(base_output, case)
        lora_metrics = score_output(lora_output, case)
        similarity = SequenceMatcher(None, base_output, lora_output).ratio()
        row = {
            "case_id": case.case_id,
            "category": case.category,
            "query": case.query,
            "base_output": base_output,
            "lora_output": lora_output,
            "base_lora_similarity": round(similarity, 3),
            "score_delta": round(lora_metrics["score"] - base_metrics["score"], 2),
        }
        for key, value in base_metrics.items():
            row[f"base_{key}"] = value
        for key, value in lora_metrics.items():
            row[f"lora_{key}"] = value
        rows.append(row)

    summary = summarize(rows)
    write_reports(Path(args.output_dir), rows, summary)

    print(json.dumps(summary, indent=2))
    print(f"\nReports written to: {args.output_dir}")


if __name__ == "__main__":
    main()
