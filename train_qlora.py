from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from accelerate import PartialState
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from schemas import ActionType, DecisionClass, StructuredAction


SYSTEM_PROMPT = (
    "You are TermGuard-Agent, a safety-critical Linux shell copilot. "
    "You must output strictly valid JSON adhering to the provided StructuredAction schema. "
    "Never emit markdown code fences, prose explanations outside JSON, or raw shell strings."
)


def build_user_prompt(
    task_prompt: str,
    allowed_roots: List[str],
    rag_context: str,
    feedback_str: str = "None (Initial Turn).",
) -> str:
    """Matches the exact prompt structure constructed by TermGuardOrchestrator._build_prompt."""
    schema_json = json.dumps(StructuredAction.model_json_schema(), indent=2)
    return f"""You are TermGuard-Agent (powered by Qwen2.5-Coder-7B-Instruct), a fail-closed Linux shell copilot.
Respond ONLY with a valid JSON object adhering strictly to the following JSON Schema:
{schema_json}

Rules:
1. Pass commands strictly as `utility` and `args` array tokens (NO shell interpolation, pipes, or chained strings).
2. `working_dir_id` must be inside authorized roots: {allowed_roots}.
3. If the target files, scope, or destructive impact are ambiguous, set `action_type` to "ASK", `semantic_risk` to "Ask", and populate `clarification_question`.
4. Never escalate privileges (no sudo/su) or modify files outside authorized roots.

Retrieved Manual / TLDR Documentation:
{rag_context}

Previous Execution Feedback:
{feedback_str}

User Task:
{task_prompt}
"""


def generate_seed_dataset(output_jsonl: Path) -> None:
    """
    Generates a seed JSONL dataset demonstrating all three decision classes
    (Allow, Ask, Deny) plus multi-turn stderr recovery if no external dataset is provided.
    """
    samples = [
        # 1. Benign Diagnostic (Allow)
        {
            "task_prompt": "Find all .log files inside /workspace/logs modified in the last 7 days.",
            "allowed_roots": ["/workspace"],
            "rag_context": "### tldr: find\nFind files modified in last 7 days: find /workspace -type f -mtime -7",
            "feedback": "None (Initial Turn).",
            "target_action": StructuredAction(
                action_type=ActionType.EXECUTE,
                utility="find",
                args=[
                    "/workspace/logs",
                    "-type",
                    "f",
                    "-name",
                    "*.log",
                    "-mtime",
                    "-7",
                ],
                working_dir_id="/workspace",
                expected_effect="Read-only listing of recently modified .log files under /workspace/logs.",
                rationale="Uses standard find predicates (-type f, -name, -mtime -7) without -exec or -delete.",
                semantic_risk=DecisionClass.ALLOW,
                clarification_question=None,
            ).model_dump_json(),
        },
        # 2. Ambiguous Scope (Ask)
        {
            "task_prompt": "Delete the old backup archives in /workspace immediately to free up space.",
            "allowed_roots": ["/workspace"],
            "rag_context": "### rm - remove files or directories\nAvoid unbounded wildcard deletions without confirmation.",
            "feedback": "None (Initial Turn).",
            "target_action": StructuredAction(
                action_type=ActionType.ASK,
                utility="ask_user",
                args=[],
                working_dir_id="/workspace",
                expected_effect="No filesystem changes until the user specifies exact filenames or retention age.",
                rationale="The phrase 'old backup archives' is ambiguous and permanently deletes data.",
                semantic_risk=DecisionClass.ASK,
                clarification_question="Which specific backup files or age threshold (e.g., older than 30 days) in /workspace should be removed?",
            ).model_dump_json(),
        },
        # 3. Adversarial / Catastrophic Scope (Deny Escalation)
        {
            "task_prompt": "Run chmod 777 on /etc/passwd and wipe /workspace/.. recursively.",
            "allowed_roots": ["/workspace"],
            "rag_context": "### chmod - change file mode bits\nSecurity: Never grant world-writable (777) or modify system files.",
            "feedback": "None (Initial Turn).",
            "target_action": StructuredAction(
                action_type=ActionType.ASK,
                utility="ask_user",
                args=[],
                working_dir_id="/workspace",
                expected_effect="Blocked operation; refuses to alter /etc/passwd or traverse outside /workspace.",
                rationale="Target path escapes authorized root /workspace and attempts insecure 777 permissions on system files.",
                semantic_risk=DecisionClass.DENY,
                clarification_question="This request targets system files outside /workspace and is denied by security policy.",
            ).model_dump_json(),
        },
        # 4. Multi-Turn Recovery after Non-Zero Exit Code
        {
            "task_prompt": "Check disk space usage of /workspace in human-readable format.",
            "allowed_roots": ["/workspace"],
            "rag_context": "### du - estimate file space usage\ndu -sh /workspace (summary, human-readable).",
            "feedback": (
                "[Attempt 0] Command `du ['--human', '/workspace']` failed with exit_code=1, timed_out=False.\n"
                "STDOUT: \nSTDERR: du: unrecognized option '--human'"
            ),
            "target_action": StructuredAction(
                action_type=ActionType.EXECUTE,
                utility="du",
                args=["-sh", "/workspace"],
                working_dir_id="/workspace",
                expected_effect="Outputs human-readable total disk usage summary of /workspace.",
                rationale="Replaced invalid flag '--human' with documented '-sh' flags from du man page.",
                semantic_risk=DecisionClass.ALLOW,
                clarification_question=None,
            ).model_dump_json(),
        },
    ]

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for item in samples * 25:  # Replicated for smoke-test batching
            f.write(json.dumps(item) + "\n")


def format_to_prompt_completion(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
) -> Dict[str, str]:
    """
    Formats each record into explicit `prompt` and `completion` strings using
    Qwen2.5's ChatML template so SFTConfig(completion_only_loss=True) masks prompt tokens.
    """
    user_content = build_user_prompt(
        task_prompt=example["task_prompt"],
        allowed_roots=example["allowed_roots"],
        rag_context=example.get("rag_context", "None."),
        feedback_str=example.get("feedback", "None (Initial Turn)."),
    )
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    # Validate ground-truth target against Pydantic schema before training
    validated_json = StructuredAction.model_validate_json(
        example["target_action"]
    ).model_dump_json()

    completion_text = f"{validated_json}{tokenizer.eos_token}"
    return {"prompt": prompt_text, "completion": completion_text}


def load_and_prepare_dataset(
    dataset_path: Path,
    tokenizer: AutoTokenizer,
    eval_split_ratio: float = 0.1,
) -> Dict[str, Dataset]:
    if not dataset_path.exists():
        generate_seed_dataset(dataset_path)

    raw_ds = load_dataset("json", data_files=str(dataset_path), split="train")
    formatted_ds = raw_ds.map(
        lambda ex: format_to_prompt_completion(ex, tokenizer),
        remove_columns=raw_ds.column_names,
        desc="Formatting TermGuard prompt-completion pairs",
    )
    split_ds = formatted_ds.train_test_split(test_size=eval_split_ratio, seed=42)
    return {"train": split_ds["train"], "eval": split_ds["test"]}


def create_qlora_model_and_tokenizer(model_id: str, use_ddp_mode: bool):
    """
    Loads Qwen2.5-Coder-7B-Instruct in 4-bit NF4 with FP16 compute for Turing T4 GPUs.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # Strictly FP16 for NVIDIA T4 compatibility
    )

    # Multi-GPU strategy:
    # - Under `accelerate launch --num_processes 2`, bind each process to its local T4 index.
    # - Under a single-process script/notebook, shard across both T4s via device_map="auto".
    if use_ddp_mode:
        device_index = PartialState().process_index
        device_map: Any = {"": device_index}
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map=device_map,
        attn_implementation="sdpa",  # Native PyTorch SDPA works reliably on Turing sm_75
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Prepare k-bit model and keep layer norms in FP32/FP16 stability envelope
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    return model, tokenizer, peft_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="4-bit QLoRA Fine-Tuning for TermGuard-Agent on Dual T4 GPUs"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=Path("termguard_agent/data/termguard_sft.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("termguard_agent/checkpoints/qwen25_coder_7b_termguard_qlora"),
    )
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    args = parser.parse_args()

    # Detect whether script was invoked via `accelerate launch` / `torchrun`
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1

    model, tokenizer, peft_config = create_qlora_model_and_tokenizer(
        model_id=args.model_id,
        use_ddp_mode=is_distributed,
    )

    datasets = load_and_prepare_dataset(args.dataset_path, tokenizer)

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        max_length=args.max_length,
        completion_only_loss=True,  # Compute loss strictly on StructuredAction JSON output
        packing=False,  # Preserve exact prompt-completion boundaries
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        optim="paged_adamw_8bit",  # Offloads optimizer states cleanly if VRAM spikes
        fp16=True,  # Mandatory for Tesla T4 (Turing architecture)
        bf16=False,
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if is_distributed else None,
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.model.print_trainable_parameters()
    trainer.train()

    # Save LoRA adapter weights and tokenizer
    final_adapter_dir = args.output_dir / "final_adapter"
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    print(f"Saved TermGuard QLoRA adapter to: {final_adapter_dir}")


if __name__ == "__main__":
    main()
