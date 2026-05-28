"""Smoke-test a saved LoRA adapter without requiring Ollama."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kubellm.qwen import QwenKubeLLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--lora_path", default="data/models/kubellm")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument(
        "--query",
        default="Pod is OOMKilled repeatedly in production. Provide root cause and kubectl commands.",
    )
    args = parser.parse_args()

    model = QwenKubeLLM(
        model_id=args.model_id,
        load_in_4bit=args.load_in_4bit,
        lora_path=args.lora_path,
        max_new_tokens=args.max_new_tokens,
        precision=args.precision,
        device_map=args.device_map,
    )
    response = model.generate_analytical(
        query=args.query,
        patterns=[],
        causal_chains=[["MemoryLeak", "OOMKilled", "IncreaseMemLimit"]],
    )
    print(response)


if __name__ == "__main__":
    main()
