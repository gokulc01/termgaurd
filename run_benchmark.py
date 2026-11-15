#!/usr/bin/env python3
# run_benchmark.py
import argparse
import hashlib
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

# Force HuggingFace offline for fast, reproducible benchmarking
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from evaluate_benchmark import ALL_CONFIGS, run_ablation_matrix  # noqa: E402
from llm_client import QwenClientConfig, QwenStructuredClient  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description="TermGuard-Agent Benchmark Runner")
    ap.add_argument(
        "--model",
        required=True,
        help="e.g. qwen2.5-coder:3b-instruct or termguard-qwen-2.5-3B-coder-ft",
    )
    ap.add_argument(
        "--backend",
        default="ollama",
        choices=["ollama", "vllm_api", "vllm_offline", "hf_outlines"],
    )
    ap.add_argument("--api-base", default="http://localhost:11434")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--tasks", default="benchmarks/tasks_schema.json")
    ap.add_argument(
        "--corpus",
        default=None,
        help="tldr_corpus.json (default: $TERMGUARD_CORPUS, then next to the code)",
    )
    ap.add_argument(
        "--prompt-style",
        choices=["simple", "detailed", "legacy"],
        default="simple",
        help="use the SAME style the fine-tuned model was trained with",
    )
    ap.add_argument(
        "--configs", nargs="+", default=list(ALL_CONFIGS), choices=list(ALL_CONFIGS)
    )
    ap.add_argument(
        "--rag-supported-only",
        action="store_true",
        help="index only utilities the static gate supports",
    )
    ap.add_argument(
        "--action-format",
        choices=["commands", "legacy"],
        default="commands",
        help="legacy = single utility/args (the format the QLoRA model was trained on)",
    )
    ap.add_argument(
        "--strip-duplicated-utility",
        action="store_true",
        help="drop args[0] when it repeats the utility name (counted in results)",
    )
    ap.add_argument(
        "--no-guided", action="store_true", help="disable schema-constrained decoding"
    )
    ap.add_argument("--output", required=True, help="path to save the JSON results")
    args = ap.parse_args()

    client_cfg = QwenClientConfig(
        model_name_or_path=args.model,
        lora_adapter_path=args.lora,
        backend=args.backend,
        api_base_url=args.api_base,
        enable_guided_decoding=not args.no_guided,
        action_format=args.action_format,
    )
    client = QwenStructuredClient(client_cfg)

    retriever = None
    if any(c in ("C1", "C2") for c in args.configs):
        from hybrid_retriever import HybridDocRetriever

        print("Loading hybrid retriever and semantic index...")
        retriever = HybridDocRetriever(
            args.corpus, supported_only=args.rag_supported_only
        )

    tasks_path = Path(args.tasks)
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "client": client.describe(),
        "tasks_file": str(tasks_path),
        "tasks_sha256": sha256_file(tasks_path),
        "corpus_sha": getattr(retriever, "corpus_sha", None),
        "rag_supported_only": args.rag_supported_only,
        "configs": args.configs,
        "python": platform.python_version(),
    }

    print(
        f"Running {args.configs} | model={args.model} ({args.backend}) | prompt={args.prompt_style}"
    )
    print(f"Results -> {args.output}")

    summary = run_ablation_matrix(
        tasks_path=tasks_path,
        llm_generate_fn=client.generate,
        retriever=retriever,
        output_report_path=Path(args.output),
        prompt_style=args.prompt_style,
        configs_to_run=args.configs,
        manifest=manifest,
        strip_duplicated_utility=args.strip_duplicated_utility,
    )

    print("\n================ BENCHMARK SUMMARY ================")
    for config_id, metrics in summary.items():
        print(f"\nConfiguration: {config_id}")
        for metric_name, val in metrics.items():
            print(f"  {metric_name:<26}: {val}")
    print(f"\nNext: python ablation_report.py {args.output}")


if __name__ == "__main__":
    main()
