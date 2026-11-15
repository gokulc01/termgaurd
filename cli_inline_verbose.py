#!/usr/bin/env python3
# cli_inline_verbose.py
import sys
import os
import shlex
import logging

from schemas import TaskCategory, ActionType
from static_gate import StaticPolicyGate
from orchestrator import TermGuardOrchestrator
from rag_retriever import DocumentationRetriever
from sandbox_runner import SandboxRunner
from llm_client import QwenClientConfig, QwenStructuredClient


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("TermGuard-Verbose")


def main():
    if len(sys.argv) < 2:
        logger.error("No prompt provided. Usage: python cli_inline_verbose.py <prompt>")
        sys.exit(1)

    user_prompt = " ".join(sys.argv[1:]).strip()
    cwd = os.getcwd()

    logger.info("=== 1. INITIALIZATION ===")
    logger.info(f"Target Directory : {cwd}")
    logger.info(f"User Prompt      : '{user_prompt}'")

    gate = StaticPolicyGate()
    retriever = DocumentationRetriever()
    sandbox = SandboxRunner(image="ubuntu:24.04")

    logger.info("Initializing QwenClient (qwen2.5-coder:3b-instruct via Ollama)")
    client_cfg = QwenClientConfig(
        model_name_or_path="qwen2.5-coder:3b-instruct",
        lora_adapter_path=None,
        backend="ollama",
        api_base_url="http://localhost:11434",
        enable_guided_decoding=True,
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
        logger.info("\n=== 2. ORCHESTRATOR PIPELINE START ===")
        result = orchestrator.run_episode(
            task_id="ZSH_INLINE_LOGGED",
            category=TaskCategory.BENIGN_ADMIN,
            task_prompt=user_prompt,
            allowed_roots=[cwd, "/workspace", "/tmp"],
            fixture_setup_cmds=[],
        )

        last_turn = result.turns[-1] if result.turns else None
        if not last_turn:
            logger.error("Orchestrator failed to generate a turn.")
            sys.exit(1)

        logger.info("\n=== 3. RAG RETRIEVAL ===")
        if last_turn.retrieval_log:
            logger.info(f"Utility Identified : {last_turn.retrieval_log.utility}")
            logger.info(
                f"Coverage Score     : {last_turn.retrieval_log.coverage_score:.4f}"
            )
        else:
            logger.info("No external documentation retrieved.")

        logger.info("\n=== 4. LLM GENERATION ===")
        sys.stderr.write("-" * 50 + "\n")
        sys.stderr.write(f"{last_turn.raw_model_output}\n")
        sys.stderr.write("-" * 50 + "\n")

        if not last_turn.parsed_action:
            logger.error("Failed to parse model output into StructuredAction JSON.")
            sys.exit(1)

        action = last_turn.parsed_action
        logger.info(f"Parsed Action Type : {action.action_type}")
        if action.expected_effect:
            logger.info(f"Expected Effect    : {action.expected_effect}")

        logger.info("\n=== 5. STATIC GATE EVALUATION ===")
        if last_turn.gate_verdict:
            logger.info(f"Final Decision     : {last_turn.gate_verdict.final_decision}")
            for idx, reason in enumerate(last_turn.gate_verdict.reasons, 1):
                logger.info(f"Reason {idx}           : {reason}")
        else:
            logger.warning("No gate verdict was recorded.")

        logger.info("\n=== 6. EXECUTION OUTCOME ===")
        if result.final_status in ("DENIED", "ASKED"):
            logger.warning(f"Process halted. System returned: {result.final_status}")
            sys.exit(2)

        if action.action_type == ActionType.EXECUTE:
            # Reconstruct the sequence safely
            cmd_str = action.to_command_preview()
            logger.info(f"Command synthesized successfully.")

            # Print ONLY the executable command to stdout for Zsh to capture
            print(cmd_str)
            sys.exit(0)

    except Exception as e:
        logger.exception("A fatal error occurred during pipeline execution.")
        sys.exit(1)
    finally:
        logger.info("\n=== 7. CLEANUP ===")
        sandbox.cleanup()
        logger.info("Sandbox isolated and dismantled.")


if __name__ == "__main__":
    main()
