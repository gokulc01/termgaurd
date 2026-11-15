#!/usr/bin/env python3
"""
Convert the existing termguard_sft.jsonl into a dataset that matches the CURRENT system.

Input records (old format):
  {"source", "task_prompt", "allowed_roots", "rag_context", "feedback",
   "target_action": "<JSON string with top-level utility/args>"}

What it does:
  1. Target: legacy {"utility","args"} -> current {"commands":[{"utility","args","pipe_to_next"}]},
     validated by StructuredAction and serialized in schema field order (the order the
     constrained decoder enforces at inference).
  2. Prompt: rebuilt with TermGuardOrchestrator._build_prompt using the SAME prompt style
     as the benchmark, plus the client's system prompt, as chat messages.
  3. RAG context: kept from the record, re-retrieved with the hybrid retriever (--corpus),
     and/or replaced by "RAG Disabled." for a share of examples (matches C3 serving).
  4. Gate consistency: EXECUTE examples labelled Allow must be allowed by the static gate,
     otherwise they are dropped (the label contradicts the policy the system enforces).
  5. Balance: refusals (ASK / Deny with no commands) are downsampled to --max-refusal-share.
  6. Leakage: prompts too similar to any benchmark task are dropped.
  7. Report: label distribution before/after, drop reasons, rationale diversity.

Usage (run from the repo root):
  python training/convert_sft_dataset.py --input termguard_sft.jsonl \
      --benchmark benchmarks/tasks_schema.json --out-dir training/data \
      [--corpus tldr_corpus.json] [--prompt-style simple]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import ValidationError  # noqa: E402

from llm_client import QwenStructuredClient  # noqa: E402
from orchestrator import TermGuardOrchestrator  # noqa: E402
from schemas import ActionType, DecisionClass, StructuredAction  # noqa: E402
from static_gate import StaticPolicyGate  # noqa: E402

TOKEN_RE = re.compile(r"[a-z0-9_./\-]+")


def tokens(text: str) -> set:
    return set(TOKEN_RE.findall(text.lower()))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def label_of(action: StructuredAction) -> str:
    if action.commands:
        if (
            action.action_type == ActionType.EXECUTE
            and action.semantic_risk == DecisionClass.ALLOW
        ):
            return "EXECUTE"
        return "ESCALATE"  # concrete command, but the model asks first
    return "DENY" if action.semantic_risk == DecisionClass.DENY else "ASK"


def parse_target(raw) -> StructuredAction:
    data = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(data, dict):
        raise ValueError("target_action is not a JSON object")
    return StructuredAction.model_validate(data)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--benchmark", default="benchmarks/tasks_schema.json")
    ap.add_argument("--out-dir", default="training/data")
    ap.add_argument(
        "--prompt-style",
        choices=["simple", "detailed"],
        default="simple",
        help="must match the --prompt-style you benchmark with",
    )
    ap.add_argument(
        "--corpus",
        default=None,
        help="re-retrieve RAG context with the hybrid retriever (matches C1 serving)",
    )
    ap.add_argument(
        "--rag-disabled-share",
        type=float,
        default=0.5,
        help="share of examples whose context is 'RAG Disabled.' (matches C3 serving)",
    )
    ap.add_argument(
        "--max-refusal-share",
        type=float,
        default=0.35,
        help="cap on ASK+DENY (no-command) examples as a share of the output",
    )
    ap.add_argument(
        "--leak-threshold",
        type=float,
        default=0.6,
        help="drop prompts whose token Jaccard with a benchmark prompt is >= this",
    )
    ap.add_argument("--val-share", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gate = StaticPolicyGate()
    orch = TermGuardOrchestrator(
        lambda p: "", None, None, None, "C1", prompt_style=args.prompt_style
    )
    system_prompt = QwenStructuredClient.SYSTEM_PROMPT

    retriever = None
    if args.corpus:
        from hybrid_retriever import HybridDocRetriever

        retriever = HybridDocRetriever(args.corpus)

    bench = []
    if args.benchmark and os.path.exists(args.benchmark):
        bench = [tokens(t["prompt"]) for t in json.load(open(args.benchmark))]
    else:
        print(
            f"WARNING: benchmark file '{args.benchmark}' not found; leakage filter disabled."
        )

    drops = collections.Counter()
    before = collections.Counter()
    kept: List[Tuple[str, Dict, Dict]] = []
    seen_prompts = set()

    with open(args.input, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                drops["bad_json_line"] += 1
                continue

            task = (rec.get("task_prompt") or "").strip()
            if not task:
                drops["no_task_prompt"] += 1
                continue
            key = " ".join(task.lower().split())
            if key in seen_prompts:
                drops["duplicate_prompt"] += 1
                continue

            try:
                action = parse_target(rec.get("target_action"))
            except (ValidationError, ValueError, json.JSONDecodeError):
                drops["target_invalid_for_current_schema"] += 1
                continue
            label = label_of(action)
            before[label] += 1

            if (
                bench
                and max(jaccard(tokens(task), b) for b in bench) >= args.leak_threshold
            ):
                drops["benchmark_leakage"] += 1
                continue

            roots = rec.get("allowed_roots") or ["/workspace"]
            wd = action.working_dir_id or roots[0]

            if label in ("EXECUTE", "ESCALATE"):
                v = gate.evaluate(
                    action, roots, rec.get("symlink_map") or {}, default_working_dir=wd
                )
                dec = v.static_decision
                if label == "EXECUTE" and dec != DecisionClass.ALLOW:
                    drops[f"gate_{dec.value.lower()}_contradicts_allow_label"] += 1
                    continue
                if label == "ESCALATE" and dec == DecisionClass.DENY:
                    drops["gate_deny_contradicts_escalate_label"] += 1
                    continue

            # Context: re-retrieve, keep, or disable, so training covers C1 and C3 serving.
            if rng.random() < args.rag_disabled_share:
                rag = "RAG Disabled."
            elif retriever is not None:
                rag = retriever.retrieve(task, top_k=2)
            else:
                rag = rec.get("rag_context") or "RAG Disabled."

            feedback = (rec.get("feedback") or "None (Initial Turn).").strip()
            orch._feedback = lambda history, fb=feedback: (
                fb
            )  # keep the record's feedback text
            user_prompt = orch._build_prompt(task, roots, wd, rag, [])

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": action.model_dump_json()},
            ]
            meta = {
                "label": label,
                "source": rec.get("source"),
                "line": line_no,
                "recovery": feedback != "None (Initial Turn).",
            }
            kept.append(
                (label, {"messages": messages, "meta": meta}, action.model_dump())
            )
            seen_prompts.add(key)

    # Balance: cap no-command refusals at max_refusal_share of the output.
    acting = [k for k in kept if k[0] in ("EXECUTE", "ESCALATE")]
    refusing = [k for k in kept if k[0] in ("ASK", "DENY")]
    s = args.max_refusal_share
    cap = int(len(acting) * s / max(1e-9, 1 - s)) if s < 1 else len(refusing)
    if len(refusing) > cap:
        rng.shuffle(refusing)
        drops["refusal_downsampled"] += len(refusing) - cap
        refusing = refusing[:cap]
    final = acting + refusing
    rng.shuffle(final)

    n_val = max(1, int(len(final) * args.val_share)) if len(final) > 20 else 0
    val, train = final[:n_val], final[n_val:]
    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as out:
            for _, row, _ in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

    after = collections.Counter(k[0] for k in final)
    rationales = collections.Counter(a["rationale"] for _, _, a in final)
    top5 = sum(c for _, c in rationales.most_common(5))
    stats = {
        "input_labels": dict(before),
        "output_labels": dict(after),
        "output_total": len(final),
        "train": len(train),
        "val": len(val),
        "dropped": dict(drops),
        "distinct_rationales": len(rationales),
        "top5_rationale_share": round(top5 / max(1, len(final)), 3),
        "top_rationales": rationales.most_common(5),
        "prompt_style": args.prompt_style,
        "rag": "hybrid" if retriever else "recorded",
        "rag_disabled_share": args.rag_disabled_share,
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w") as out:
        json.dump(stats, out, indent=2)

    print(
        json.dumps({k: v for k, v in stats.items() if k != "top_rationales"}, indent=2)
    )
    print("top rationales:")
    for r, c in stats["top_rationales"]:
        print(f"  {c:>5}  {r[:90]}")
    if stats["top5_rationale_share"] > 0.5:
        print(
            "WARNING: over half the targets share 5 rationale strings; the model will learn "
            "templated rationales. Labels matter more, but consider diversifying them."
        )
    if after.get("EXECUTE", 0) < 0.5 * max(1, len(final)):
        print("WARNING: fewer than half the examples are EXECUTE; expect over-refusal.")


if __name__ == "__main__":
    main()
