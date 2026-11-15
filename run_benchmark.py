#!/usr/bin/env python3
# run_benchmark.py
import sys
import argparse
from pathlib import Path
from llm_client import QwenClientConfig, QwenStructuredClient
from evaluate_benchmark import run_ablation_matrix


def main():
    parser = argparse.ArgumentParser(description="TermGuard-Agent Benchmark Runner")
    parser.add_argument(
        "--output",
        type=str,
        default="base_3.5b_results.json",
        help="Path to save the JSON results",
    )
    args = parser.parse_args()

    # Configuration targeting vanilla 3B baseline via Ollama
    client_cfg = QwenClientConfig(
        model_name_or_path="qwen2.5-coder:3b-instruct",
        lora_adapter_path=None,
        backend="ollama",
        api_base_url="http://localhost:11434",
        enable_guided_decoding=True,
    )
    client = QwenStructuredClient(client_cfg)

    print(f"Starting C0-C3 benchmark execution over 60 tasks...")
    print(
        f"Target Model: {client_cfg.model_name_or_path} (Backend: {client_cfg.backend})"
    )
    print(f"Results will be saved to: {args.output}")

    summary = run_ablation_matrix(
        tasks_path=Path("benchmarks/tasks_schema.json"),
        llm_generate_fn=client.generate,
        output_report_path=Path(args.output),
    )

    print("\n================ BENCHMARK SUMMARY ================")
    for config_id, metrics in summary.items():
        print(f"\nConfiguration: {config_id}")
        for metric_name, val in metrics.items():
            print(f"  {metric_name:<26}: {val}")


if __name__ == "__main__":
    main()
