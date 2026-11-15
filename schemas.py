from __future__ import annotations
from enum import Enum
from typing import List, Optional
import copy
from pydantic import BaseModel, Field, field_validator, model_validator
import shlex


class ActionType(str, Enum):
    EXECUTE = "EXECUTE"
    ASK = "ASK"


class DecisionClass(str, Enum):
    ALLOW = "Allow"
    ASK = "Ask"
    DENY = "Deny"


class TaskCategory(str, Enum):
    DIAGNOSTICS_LOGS = "diagnostics_logs"
    BENIGN_ADMIN = "benign_administration"
    CATASTROPHIC_SCOPE = "catastrophic_scope"
    OBFUSCATED_DYNAMIC = "obfuscated_dynamic"
    PRIVILEGE_PROBES = "privilege_probes"
    AMBIGUOUS_INTENT = "ambiguous_intent"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


BENIGN_CATEGORIES = {
    TaskCategory.DIAGNOSTICS_LOGS.value,
    TaskCategory.BENIGN_ADMIN.value,
}


class CommandNode(BaseModel):
    utility: str = Field(
        description="Bare Linux executable name (e.g., 'find', 'grep', 'mkdir', 'cd'). No paths."
    )
    args: List[str] = Field(
        default_factory=list,
        description="Command arguments as discrete array tokens, never interpolated shell strings.",
    )
    pipe_to_next: bool = Field(
        default=False,
        description=(
            "true = pipe this command's stdout into the next command (|). "
            "false = run the next command only if this one succeeds (&&). Ignored on the last command."
        ),
    )

    @field_validator("utility")
    @classmethod
    def validate_utility_token(cls, v: str) -> str:
        v = v.strip()
        if not v or any(
            ch in v for ch in (" ", ";", "|", "&", "$", "`", ">", "<", "=", "\n")
        ):
            raise ValueError(
                "Utility must be a single binary token without shell metacharacters."
            )
        return v


class StructuredAction(BaseModel):
    """Schema forced upon Qwen2.5-Coder generation."""

    action_type: ActionType = Field(
        description="EXECUTE to run a command, or ASK if target/mutation scope is ambiguous."
    )
    commands: List[CommandNode] = Field(
        default_factory=list,
        description="Ordered list of commands. Can be empty if action_type is ASK.",
    )
    working_dir_id: str = Field(
        default="/workspace",
        description="Normalized working directory path inside the authorized task root.",
    )
    expected_effect: str = Field(
        description="Concise statement of expected read/write impact on container state."
    )
    rationale: str = Field(
        description="Short reasoning explaining flag choices or why disambiguation is required."
    )
    semantic_risk: DecisionClass = Field(
        default=DecisionClass.ALLOW,
        description="Model's self-assessed risk level (Allow, Ask, or Deny).",
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="Required when action_type == ASK or semantic_risk == Ask.",
    )

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_single_command(cls, data):
        """
        Legacy format (what the QLoRA data was generated with):
            {"action_type": ..., "utility": "df", "args": ["-h"], ...}
        is converted to the current {"commands": [{"utility": "df", "args": ["-h"]}]}.
        The result goes through exactly the same validation and gate.
        """
        if isinstance(data, dict) and "commands" not in data and "utility" in data:
            data = dict(data)
            util = data.pop("utility")
            args = data.pop("args", None) or []
            data["commands"] = [{"utility": util, "args": args}] if util else []
        return data

    @model_validator(mode="after")
    def execute_needs_commands(self) -> "StructuredAction":
        # EXECUTE with nothing to execute is a malformed tool call, not a refusal.
        if self.action_type == ActionType.EXECUTE and not self.commands:
            raise ValueError(
                "action_type is EXECUTE but `commands` is empty or missing; include at least "
                "one command, or use action_type ASK with a clarification_question."
            )
        return self

    def render_command_line(self) -> str:
        """
        The ONE canonical rendering of this action. The static gate parses exactly
        this string, the sandbox executes exactly this string (via `sh -c`), and the
        CLI prints exactly this string. Every token is shlex-quoted, so the only shell
        operators that can appear are the `|` / `&&` joiners emitted here.
        """
        if not self.commands:
            return ""
        parts: List[str] = []
        for i, cmd in enumerate(self.commands):
            tokens = [cmd.utility] + [str(a) for a in cmd.args]
            parts.append(" ".join(shlex.quote(t) for t in tokens))
            if i < len(self.commands) - 1:
                parts.append("|" if cmd.pipe_to_next else "&&")
        return " ".join(parts)

    # Backwards-compatible name used by older callers.
    def to_command_preview(self) -> str:
        return self.render_command_line()


def generation_json_schema() -> dict:
    """
    StructuredAction schema for constrained decoding with EVERY property required.
    Pydantic omits defaulted fields (commands, working_dir_id, semantic_risk, ...) from
    `required`, and grammar-constrained decoders then let small models skip them.
    Nullable fields (clarification_question) stay nullable, so the model can emit null.
    """
    schema = copy.deepcopy(StructuredAction.model_json_schema())
    for obj in [schema, *schema.get("$defs", {}).values()]:
        if obj.get("type") == "object" and "properties" in obj:
            obj["required"] = list(obj["properties"].keys())
    return schema


def legacy_generation_json_schema() -> dict:
    """Single-command schema matching termguard_sft.jsonl targets (all fields required)."""
    risk = {"type": "string", "enum": [d.value for d in DecisionClass]}
    props = {
        "action_type": {"type": "string", "enum": [a.value for a in ActionType]},
        "utility": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "args": {"type": "array", "items": {"type": "string"}},
        "working_dir_id": {"type": "string"},
        "expected_effect": {"type": "string"},
        "rationale": {"type": "string"},
        "semantic_risk": risk,
        "clarification_question": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    }
    return {
        "title": "StructuredAction",
        "type": "object",
        "properties": props,
        "required": list(props.keys()),
    }


class RetrievalLog(BaseModel):
    utility: str
    supported_utility: bool
    matched_passages: List[str]
    coverage_score: float


class GateVerdict(BaseModel):
    static_decision: DecisionClass
    semantic_decision: DecisionClass
    final_decision: DecisionClass
    reasons: List[str] = Field(default_factory=list)
    normalized_paths: List[str] = Field(default_factory=list)
    # True when the denial is about an unsupported form (e.g. unknown utility, shell
    # builtin, uninspectable sed script) rather than a dangerous capability. Only
    # recoverable denials are fed back to the model for another attempt; the retry
    # is gated again from scratch.
    recoverable: bool = False
    rendered_command: Optional[str] = None


class SandboxResult(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    oom_killed: bool = False
    output_truncated: bool = False
    wall_time_ms: float


class TurnTelemetry(BaseModel):
    attempt_index: int
    raw_model_output: str
    schema_valid: bool
    parsed_action: Optional[StructuredAction] = None
    retrieval_log: Optional[RetrievalLog] = None
    gate_verdict: Optional[GateVerdict] = None
    sandbox_result: Optional[SandboxResult] = None
    # Pydantic error for schema-invalid output, fed back to the model on retry.
    validation_error: Optional[str] = None
    # Number of commands where a repeated utility name was stripped from args[0]
    # (orchestrator strip_duplicated_utility option).
    stripped_duplicate_utility: int = 0
    # Set when the orchestrator flipped a model ASK into EXECUTE (see orchestrator).
    auto_corrected_ask: bool = False


class EpisodeResult(BaseModel):
    task_id: str
    config_id: str
    category: TaskCategory
    turns: List[TurnTelemetry]
    final_status: str
    # Prohibited side effect observed by an independent validator (proposal's UER).
    unsafe_executed: bool = False
    # Any command reached the sandbox on this task (stricter, reported separately).
    executed_any_cmd: bool = False
    post_condition_passed: bool = False
    recovered_after_failure: bool = False
    error: Optional[str] = None
