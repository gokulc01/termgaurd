#!/usr/bin/env python3
"""
Inline shell copilot. Prints ONE gate-validated command on stdout (logs go to stderr).

The command is meant for the user's own shell, so nothing is executed here: the
orchestrator runs in dry-run mode (generate + static gate), the gate checks paths
against the real current directory with host symlink resolution, and the printed
string is exactly the one the gate parsed.

Exit codes: 0 = command printed, 1 = error / invalid model output, 2 = denied or aborted.
"""

import argparse
import logging
import os
import sys

from llm_client import QwenClientConfig, QwenStructuredClient
from orchestrator import TermGuardOrchestrator
from schemas import DecisionClass, TaskCategory
from static_gate import StaticPolicyGate

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("TermGuard-Verbose")


def host_resolver(paths):
    return [os.path.realpath(p) for p in paths]


def ask_user(question: str) -> str:
    sys.stderr.write(f"\n[TermGuard-Agent] {question}\n> ")
    sys.stderr.flush()
    try:
        return input().strip()
    except (KeyboardInterrupt, EOFError):
        sys.stderr.write("\nAborted by user.\n")
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description="TermGuard inline shell copilot")
    ap.add_argument("prompt", nargs="+")
    ap.add_argument(
        "--model",
        default=os.environ.get("TERMGUARD_MODEL", "qwen2.5-coder:3b-instruct"),
    )
    ap.add_argument(
        "--api-base",
        default=os.environ.get("TERMGUARD_API_BASE", "http://localhost:11434"),
    )
    ap.add_argument(
        "--corpus",
        default=None,
        help="tldr_corpus.json (default: $TERMGUARD_CORPUS, then next to this script)",
    )
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument(
        "--prompt-style", choices=["simple", "detailed", "legacy"], default="simple"
    )
    ap.add_argument(
        "--action-format", choices=["commands", "legacy"], default="commands"
    )
    ap.add_argument("--strip-duplicated-utility", action="store_true")
    ap.add_argument(
        "--max-rounds",
        type=int,
        default=3,
        help="clarification rounds before giving up",
    )
    args = ap.parse_args()

    user_prompt = " ".join(args.prompt).strip()
    cwd = os.path.realpath(os.getcwd())

    logger.info("=== 1. INITIALIZATION ===")
    logger.info(f"Working directory : {cwd}")
    logger.info(f"Prompt            : '{user_prompt}'")

    retriever = None
    if not args.no_rag:
        logger.info("Loading hybrid retriever...")
        from hybrid_retriever import HybridDocRetriever

        try:
            retriever = HybridDocRetriever(args.corpus, supported_only=True)
        except FileNotFoundError as exc:
            logger.warning(f"{exc} Continuing without RAG.")

    client = QwenStructuredClient(
        QwenClientConfig(
            model_name_or_path=args.model,
            backend="ollama",
            api_base_url=args.api_base,
            enable_guided_decoding=True,
            action_format=args.action_format,
        )
    )
    orchestrator = TermGuardOrchestrator(
        llm_generate_fn=client.generate,
        retriever=retriever,
        static_gate=StaticPolicyGate(),
        sandbox=None,
        config_id="C1",
        prompt_style=args.prompt_style,
        strip_duplicated_utility=args.strip_duplicated_utility,
    )

    try:
        for round_idx in range(args.max_rounds):
            logger.info(f"=== 2. PIPELINE (round {round_idx + 1}) ===")
            result = orchestrator.run_episode(
                task_id="CLI_INLINE",
                category=TaskCategory.BENIGN_ADMIN,
                task_prompt=user_prompt,
                allowed_roots=[cwd],
                fixture_setup_cmds=[],
                working_dir=cwd,
                dry_run=True,
                path_resolver=host_resolver,
            )
            for t in result.turns:
                sys.stderr.write(
                    "-" * 50 + f"\n[attempt {t.attempt_index}] {t.raw_model_output}\n"
                )
                if t.gate_verdict:
                    logger.info(
                        f"Gate: {t.gate_verdict.final_decision.value} | "
                        + " ".join(t.gate_verdict.reasons)
                    )
            sys.stderr.write("-" * 50 + "\n")

            last = result.turns[-1] if result.turns else None
            action = last.parsed_action if last else None
            verdict = last.gate_verdict if last else None
            logger.info(f"=== 3. OUTCOME: {result.final_status} ===")

            if result.final_status == "PROPOSED":
                print(verdict.rendered_command)
                sys.exit(0)

            if result.final_status == "DENIED":
                logger.error("Denied by the static policy gate or the model.")
                sys.exit(2)

            if result.final_status != "ASKED":
                logger.error("Could not obtain a valid command from the model.")
                sys.exit(1)

            # ASKED with a concrete, gate-checked command: confirm it, don't re-generate.
            if (
                action
                and action.commands
                and verdict
                and verdict.static_decision != DecisionClass.DENY
            ):
                sys.stderr.write(f"\nProposed: {verdict.rendered_command}\n")
                for r in verdict.reasons:
                    sys.stderr.write(f"  - {r}\n")
                if action.clarification_question:
                    sys.stderr.write(f"  - {action.clarification_question}\n")
                if ask_user("Run this command? [y/N]").lower() in ("y", "yes"):
                    print(verdict.rendered_command)
                    sys.exit(0)
                logger.info("Not confirmed.")
                sys.exit(2)

            # ASKED without commands: the model needs disambiguation.
            question = (
                action.clarification_question if action else None
            ) or "Please clarify your request."
            reply = ask_user(question)
            if not reply:
                logger.error("Empty response. Aborting.")
                sys.exit(2)
            user_prompt = f"{user_prompt}\nUser clarification: {reply}"

        logger.error(
            "No executable command after the maximum number of clarification rounds."
        )
        sys.exit(2)
    except SystemExit:
        raise
    except Exception:
        logger.exception("A fatal error occurred during pipeline execution.")
        sys.exit(1)


if __name__ == "__main__":
    main()
