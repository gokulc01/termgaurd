# run_benchmark.py
from pathlib import Path
from llm_client import QwenClientConfig, QwenStructuredClient
from evaluate_benchmark import run_ablation_matrix

# Initialize Ollama client
client_cfg = QwenClientConfig(
    model_name_or_path="qwen2.5-coder:7b-instruct",
    backend="ollama",
    api_base_url="http://localhost:11434",
    enable_guided_decoding=True,
    timeout_sec=60.0,
)
client = QwenStructuredClient(client_cfg)

print("Starting C0-C3 benchmark execution over 60 tasks...")
summary = run_ablation_matrix(
    tasks_path=Path("benchmarks/tasks_schema.json"),
    llm_generate_fn=client.generate,
    output_report_path=Path("benchmarks/ablation_report.json"),
)

print("\n================ BENCHMARK SUMMARY ================")
for config_id, metrics in summary.items():
    print(f"\nConfiguration: {config_id}")
    for metric_name, val in metrics.items():
        print(f"  {metric_name:<26}: {val}")
