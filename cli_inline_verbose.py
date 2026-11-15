#!/usr/bin/env python3
import sys
import os
import logging

from schemas import TaskCategory, ActionType
from static_gate import StaticPolicyGate
from orchestrator import TermGuardOrchestrator
from hybrid_retriever import HybridDocRetriever
from sandbox_runner import SandboxRunner
from llm_client import QwenClientConfig, QwenStructuredClient

# Configure detailed logging to standard error (so stdout stays clean for the final command)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("TermGuard-Verbose")


def main():
    if len(sys.argv) < 2:
        logger.error(
            "No prompt provided. Usage: python cli_inline_verbose_2.py <prompt>"
        )
        sys.exit(1)

    user_prompt = " ".join(sys.argv[1:]).strip()
    cwd = os.getcwd()

    logger.info("=== 1. INITIALIZATION ===")
    logger.info(f"Target Directory : {cwd}")
    logger.info(f"Initial Prompt   : '{user_prompt}'")

    gate = StaticPolicyGate()
    logger.info("Loading Hybrid Retriever and semantic index...")
    corpus_path = os.path.expanduser(
        "~/Documents/iiith/sem3/lma/grp_prj/termgaurd_agent/tldr_corpus.json"
    )
    retriever = HybridDocRetriever(corpus_path)
    sandbox = SandboxRunner(image="ubuntu:24.04")

    logger.info("Initializing model via Ollama")
    client_cfg = QwenClientConfig(
        # model_name_or_path="termguard-qwen-2.5-3B-coder-ft",
        model_name_or_path="qwen2.5-coder:3b-instruct",
        # model_name_or_path="phi4-mini:3.8b",
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
        while True:
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
            # Adjusted logging since Hybrid Retriever replaces the explicit utility/coverage log
            logger.info("Hybrid Semantic RAG context applied.")

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
                logger.info(
                    f"Final Decision     : {last_turn.gate_verdict.final_decision}"
                )
                for idx, reason in enumerate(last_turn.gate_verdict.reasons, 1):
                    logger.info(f"Reason {idx}           : {reason}")
            else:
                logger.warning("No gate verdict was recorded.")

            logger.info("\n=== 6. EXECUTION OUTCOME ===")

            # Handle explicit DENY
            if result.final_status == "DENIED":
                logger.error(
                    "Process halted. Static Policy Gate or Model denied the action."
                )
                sys.exit(2)

            # Handle ASK (Disambiguation/Clarification)
            if result.final_status == "ASKED":
                logger.warning(
                    "INTERVENTION REQUIRED: Scope is ambiguous or potentially destructive."
                )

                question = "Please clarify your request."
                if action.clarification_question:
                    question = action.clarification_question
                elif last_turn.gate_verdict and last_turn.gate_verdict.reasons:
                    question = last_turn.gate_verdict.reasons[0]

                sys.stderr.write(f"\n[TermGuard-Agent] {question}\n> ")
                sys.stderr.flush()

                try:
                    user_reply = input()
                except (KeyboardInterrupt, EOFError):
                    sys.stderr.write("\nAborted by user.\n")
                    sys.exit(1)

                if not user_reply.strip():
                    logger.error("Empty response. Aborting.")
                    sys.exit(2)

                user_prompt = f"{user_prompt}\nUser clarification: {user_reply.strip()}"
                logger.info("Resubmitting task with user clarification...")
                continue

            # Handle successful EXECUTE
            if action.action_type == ActionType.EXECUTE:
                cmd_str = action.to_command_preview()
                logger.info(f"Command synthesized successfully.")

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
