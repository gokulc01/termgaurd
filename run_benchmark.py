#!/usr/bin/env python3
# run_benchmark.py
import sys
import os
import argparse
from pathlib import Path

# Force HuggingFace offline for fast benchmarking
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from llm_client import QwenClientConfig, QwenStructuredClient
from evaluate_benchmark import run_ablation_matrix
from hybrid_retriever import HybridDocRetriever


def main():
    parser = argparse.ArgumentParser(description="TermGuard-Agent Benchmark Runner")
    parser.add_argument(
        "--output",
        type=str,
        default="ft_3.5b_simple_prompt_results.json",
        help="Path to save the JSON results",
    )
    args = parser.parse_args()

    # Configuration targeting vanilla 3B baseline via Ollama
    client_cfg = QwenClientConfig(
        # model_name_or_path="qwen2.5-coder:3b-instruct",
        model_name_or_path="termguard-qwen-2.5-3B-coder-ft",
        lora_adapter_path=None,
        backend="ollama",
        api_base_url="http://localhost:11434",
        enable_guided_decoding=True,
    )
    client = QwenStructuredClient(client_cfg)

    print("Loading Hybrid Retriever and semantic index...")
    corpus_path = os.path.expanduser(
        "~/Documents/iiith/sem3/lma/grp_prj/termgaurd_agent/tldr_corpus.json"
    )
    retriever = HybridDocRetriever(corpus_path)

    print(f"Starting C0-C3 benchmark execution over 60 tasks...")
    print(
        f"Target Model: {client_cfg.model_name_or_path} (Backend: {client_cfg.backend})"
    )
    print(f"Results will be saved to: {args.output}")

    summary = run_ablation_matrix(
        tasks_path=Path("benchmarks/tasks_schema.json"),
        llm_generate_fn=client.generate,
        retriever=retriever,
        output_report_path=Path(args.output),
    )

    print("\n================ BENCHMARK SUMMARY ================")
    for config_id, metrics in summary.items():
        print(f"\nConfiguration: {config_id}")
        for metric_name, val in metrics.items():
            print(f"  {metric_name:<26}: {val}")


if __name__ == "__main__":
    main()
