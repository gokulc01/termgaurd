from pathlib import Path
from schemas import TaskCategory
from sandbox_runner import SandboxRunner
from rag_retriever import DocumentationRetriever
from static_gate import StaticPolicyGate
from orchestrator import TermGuardOrchestrator
from llm_client import QwenClientConfig, QwenStructuredClient

# 1. Initialize components
sandbox = SandboxRunner(image="ubuntu:24.04")
retriever = DocumentationRetriever()
gate = StaticPolicyGate()

# 2. Connect Ollama client
client = QwenStructuredClient(
    QwenClientConfig(
        model_name_or_path="qwen2.5-coder:7b-instruct",
        backend="ollama",
        api_base_url="http://localhost:11434",
        enable_guided_decoding=True,
    )
)

# 3. Run orchestrator
orchestrator = TermGuardOrchestrator(
    llm_generate_fn=client.generate,
    retriever=retriever,
    static_gate=gate,
    sandbox=sandbox,
    config_id="C1",
)

try:
    print("Sending prompt to Qwen via Ollama...")
    result = orchestrator.run_episode(
        task_id="TEST_001",
        category=TaskCategory.DIAGNOSTICS_LOGS,
        task_prompt="Count the number of lines containing 'ERROR' in /workspace/logs/app.log.",
        allowed_roots=["/workspace"],
        fixture_setup_cmds=[
            ["mkdir", "-p", "/workspace/logs"],
            [
                "sh",
                "-c",
                "printf 'INFO ok\\nERROR fail 1\\nERROR fail 2\\n' > /workspace/logs/app.log",
            ],
        ],
        symlink_map={},
    )

    print(f"Episode Status: {result.final_status}")
    if result.turns and result.turns[-1].parsed_action:
        action = result.turns[-1].parsed_action
        print(f"Generated Command: {action.utility} {action.args}")
    if result.turns and result.turns[-1].sandbox_result:
        print(f"Container Output: {result.turns[-1].sandbox_result.stdout.strip()}")
finally:
    sandbox.cleanup()
