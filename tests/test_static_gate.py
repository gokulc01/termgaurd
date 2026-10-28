from __future__ import annotations
import pytest

from schemas import DecisionClass, StructuredAction
from static_gate import StaticPolicyGate


@pytest.fixture
def gate() -> StaticPolicyGate:
    return StaticPolicyGate()


@pytest.fixture
def sample_symlink_map() -> dict[str, str]:
    return {
        "/workspace/shadow_link": "/etc/shadow",
        "/workspace/etc_escape": "/etc",
        "/workspace/loop_a": "/workspace/loop_b",
        "/workspace/loop_b": "/workspace/loop_a",
    }


def test_allow_read_only_diagnostic(gate: StaticPolicyGate):
    action = StructuredAction(
        action_type="EXECUTE",
        utility="grep",
        args=["-rn", "ERROR", "/workspace/logs"],
        working_dir_id="/workspace",
        expected_effect="Read-only pattern search in logs.",
        rationale="Safe diagnostic command.",
        semantic_risk=DecisionClass.ALLOW,
    )
    verdict = gate.evaluate(action=action, allowed_roots=["/workspace"], symlink_map={})
    assert verdict.final_decision == DecisionClass.ALLOW
    assert verdict.static_decision == DecisionClass.ALLOW


def test_allow_bounded_mutation(gate: StaticPolicyGate):
    action = StructuredAction(
        action_type="EXECUTE",
        utility="mkdir",
        args=["-p", "/workspace/releases/v1"],
        working_dir_id="/workspace",
        expected_effect="Create nested release directory.",
        rationale="Bounded mutation inside authorized root.",
        semantic_risk=DecisionClass.ALLOW,
    )
    verdict = gate.evaluate(action=action, allowed_roots=["/workspace"], symlink_map={})
    assert verdict.final_decision == DecisionClass.ALLOW
    assert verdict.static_decision == DecisionClass.ALLOW


def test_deny_prohibited_utility(gate: StaticPolicyGate):
    action = StructuredAction(
        action_type="EXECUTE",
        utility="sudo",
        args=["cat", "/etc/shadow"],
        working_dir_id="/workspace",
        expected_effect="Privilege escalation attempt.",
        rationale="Trying sudo.",
        semantic_risk=DecisionClass.ALLOW,
    )
    verdict = gate.evaluate(action=action, allowed_roots=["/workspace"], symlink_map={})
    assert verdict.final_decision == DecisionClass.DENY
    assert any("prohibited" in r.lower() for r in verdict.reasons)


def test_deny_unsupported_utility(gate: StaticPolicyGate):
    action = StructuredAction(
        action_type="EXECUTE",
        utility="nmap",
        args=["-sS", "127.0.0.1"],
        working_dir_id="/workspace",
        expected_effect="Network scan.",
        rationale="Testing unsupported binary.",
        semantic_risk=DecisionClass.ALLOW,
    )
    verdict = gate.evaluate(action=action, allowed_roots=["/workspace"], symlink_map={})
    assert verdict.final_decision == DecisionClass.DENY
    assert any("unsupported utility" in r.lower() for r in verdict.reasons)


def test_deny_symlink_traversal_escape(
    gate: StaticPolicyGate, sample_symlink_map: dict[str, str]
):
    action = StructuredAction(
        action_type="EXECUTE",
        utility="cat",
        args=["/workspace/shadow_link"],
        working_dir_id="/workspace",
        expected_effect="Read sensitive shadow file via symlink.",
        rationale="Symlink escape probe.",
        semantic_risk=DecisionClass.ALLOW,
    )
    verdict = gate.evaluate(
        action=action,
        allowed_roots=["/workspace"],
        symlink_map=sample_symlink_map,
    )
    assert verdict.final_decision == DecisionClass.DENY
    assert any(
        "resolves outside allowed roots" in r.lower() or "symlink" in r.lower()
        for r in verdict.reasons
    )


def test_deny_symlink_loop(gate: StaticPolicyGate, sample_symlink_map: dict[str, str]):
    action = StructuredAction(
        action_type="EXECUTE",
        utility="ls",
        args=["/workspace/loop_a"],
        working_dir_id="/workspace",
        expected_effect="Traverse symlink loop.",
        rationale="Testing loop resolution.",
        semantic_risk=DecisionClass.ALLOW,
    )
    verdict = gate.evaluate(
        action=action,
        allowed_roots=["/workspace"],
        symlink_map=sample_symlink_map,
    )
    assert verdict.final_decision == DecisionClass.DENY


def test_deny_compound_commands(gate: StaticPolicyGate):
    action = StructuredAction(
        action_type="EXECUTE",
        utility="grep",
        args=["ERROR", "/workspace/app.log", "|", "wc", "-l"],
        working_dir_id="/workspace",
        expected_effect="Piped command execution.",
        rationale="Attempting compound pipeline in atomic action.",
        semantic_risk=DecisionClass.ALLOW,
    )
    verdict = gate.evaluate(action=action, allowed_roots=["/workspace"], symlink_map={})
    assert verdict.final_decision == DecisionClass.DENY
    assert any(
        "pipeline" in r.lower() or "compound" in r.lower() for r in verdict.reasons
    )


def test_deny_obfuscated_and_dynamic(gate: StaticPolicyGate):
    action = StructuredAction(
        action_type="EXECUTE",
        utility="sh",
        args=["-c", "rm -rf /workspace/*"],
        working_dir_id="/workspace",
        expected_effect="Dynamic shell wrapper execution.",
        rationale="Obfuscated shell command.",
        semantic_risk=DecisionClass.ALLOW,
    )
    verdict = gate.evaluate(action=action, allowed_roots=["/workspace"], symlink_map={})
    assert verdict.final_decision == DecisionClass.DENY


def test_ask_find_delete(gate: StaticPolicyGate):
    action = StructuredAction(
        action_type="EXECUTE",
        utility="find",
        args=["/workspace", "-name", "*.tmp", "-delete"],
        working_dir_id="/workspace",
        expected_effect="Delete temporary files.",
        rationale="Using find -delete.",
        semantic_risk=DecisionClass.ALLOW,
    )
    verdict = gate.evaluate(action=action, allowed_roots=["/workspace"], symlink_map={})
    assert verdict.final_decision == DecisionClass.ASK
    assert any(
        "find -delete" in r.lower()
        or "requires explicit user confirmation" in r.lower()
        for r in verdict.reasons
    )
