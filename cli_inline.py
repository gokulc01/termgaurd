#!/usr/bin/env python3
# cli_inline.py
import sys
import os
import shlex

from schemas import TaskCategory, ActionType, DecisionClass
from static_gate import StaticPolicyGate
from orchestrator import TermGuardOrchestrator
from rag_retriever import DocumentationRetriever
from sandbox_runner import SandboxRunner
from llm_client import QwenClientConfig, QwenStructuredClient


def main():
    if len(sys.argv) < 2:
        sys.exit(1)

    user_prompt = " ".join(sys.argv[1:]).strip()
    if not user_prompt:
        sys.exit(1)

    cwd = os.getcwd()

    # Fast offline initialization: static gate + HuggingFace LoRA[cite: 7]
    gate = StaticPolicyGate()
    retriever = DocumentationRetriever()
    sandbox = SandboxRunner(image="ubuntu:24.04")

    # Updated configuration for fine-tuned LoRA model
    # client_cfg = QwenClientConfig(
    #     model_name_or_path="Qwen/Qwen2.5-Coder-3B-Instruct",
    #     lora_adapter_path="termguard-qwen-lora",
    #     backend="hf_outlines",
    #     enable_guided_decoding=False,
    # )
    client_cfg = QwenClientConfig(
        model_name_or_path="qwen2.5-coder:3b-instruct",  # Vanilla baseline model
        lora_adapter_path=None,
        backend="ollama",
        api_base_url="http://localhost:11434",
        enable_guided_decoding=True,  # Turned back ON to enforce JSON formatting
    )
    client = QwenStructuredClient(client_cfg)

    orchestrator = TermGuardOrchestrator(
        llm_generate_fn=client.generate,
        retriever=retriever,
        static_gate=gate,
        sandbox=sandbox,
        config_id="C1",
    )

    try:
        # Run stage 1 (generation) + stage 3 (static gate)[cite: 7]
        # Note: allowed_roots includes cwd so local tasks stay valid[cite: 7]
        result = orchestrator.run_episode(
            task_id="ZSH_INLINE",
            category=TaskCategory.BENIGN_ADMIN,
            task_prompt=user_prompt,
            allowed_roots=[cwd, "/workspace", "/tmp"],
            fixture_setup_cmds=[],
        )

        last_turn = result.turns[-1] if result.turns else None
        if not last_turn or not last_turn.parsed_action:
            sys.exit(1)

        action = last_turn.parsed_action

        # If gate denied or requested clarification, do not insert anything[cite: 7]
        if result.final_status in ("DENIED", "ASKED"):
            sys.stderr.write(f"\n[TermGuard: {result.final_status}]\n")
            if last_turn.gate_verdict:
                for r in last_turn.gate_verdict.reasons:
                    sys.stderr.write(f" - {r}\n")
            sys.exit(2)

        # Print ONLY the shell-escaped executable command to stdout for Zsh BUFFER[cite: 7]
        if action.action_type == ActionType.EXECUTE:
            tokens = [action.utility] + action.args
            cmd_str = " ".join(shlex.quote(t) for t in tokens)
            print(cmd_str)
            sys.exit(0)

    finally:
        sandbox.cleanup()


if __name__ == "__main__":
    main()
