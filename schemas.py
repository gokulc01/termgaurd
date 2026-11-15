from __future__ import annotations
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
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


class CommandNode(BaseModel):
    utility: str = Field(
        description="Primary Linux executable name (e.g., 'find', 'grep', 'mkdir', 'cd')."
    )
    args: List[str] = Field(
        default_factory=list,
        description="Command arguments as discrete array tokens, never interpolated shell strings.",
    )

    @field_validator("utility")
    @classmethod
    def validate_utility_token(cls, v: str) -> str:
        v = v.strip()
        if not v or any(ch in v for ch in (" ", ";", "|", "&", "$", "`", ">", "<")):
            raise ValueError(
                "Utility must be a single binary token without shell metacharacters."
            )
        return v


class StructuredAction(BaseModel):
    """Schema forced upon Qwen2.5-Coder-7B-Instruct generation."""

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

    def to_command_preview(self) -> str:
        if not self.commands:
            return ""

        # Standard Linux utilities that typically consume stdin via a pipe
        pipe_consumers = {
            "wc",
            "grep",
            "egrep",
            "awk",
            "sed",
            "sort",
            "uniq",
            "head",
            "tail",
            "cut",
            "tr",
            "less",
            "more",
        }

        cmd_strings = []
        for cmd in self.commands:
            tokens = [cmd.utility] + [str(a) for a in cmd.args]
            cmd_strings.append(" ".join(shlex.quote(t) for t in tokens))

        if len(cmd_strings) == 1:
            return cmd_strings[0]

        # Smart joiner logic
        result = cmd_strings[0]
        for i in range(1, len(self.commands)):
            next_util = self.commands[i].utility

            if next_util in pipe_consumers:
                result += f" | {cmd_strings[i]}"
            else:
                result += f" && {cmd_strings[i]}"

        return result


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


class SandboxResult(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    oom_killed: bool = False
    wall_time_ms: float


class TurnTelemetry(BaseModel):
    attempt_index: int
    raw_model_output: str
    schema_valid: bool
    parsed_action: Optional[StructuredAction] = None
    retrieval_log: Optional[RetrievalLog] = None
    gate_verdict: Optional[GateVerdict] = None
    sandbox_result: Optional[SandboxResult] = None


class EpisodeResult(BaseModel):
    task_id: str
    config_id: str
    category: TaskCategory
    turns: List[TurnTelemetry]
    final_status: str
    unsafe_executed: bool = False
    post_condition_passed: bool = False
    recovered_after_failure: bool = False
