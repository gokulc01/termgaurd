from __future__ import annotations
import os
import posixpath
from typing import Dict, List, Set, Tuple
import shlex
import tree_sitter_bash as tsbash
from tree_sitter import Language, Parser, Node

from schemas import DecisionClass, GateVerdict, StructuredAction


BASH_LANGUAGE = Language(tsbash.language())

FORBIDDEN_AST_NODES: Set[str] = {
    "ERROR",
    "MISSING",
    "command_substitution",
    "process_substitution",
    "function_definition",
    "subshell",
    "c_style_for_statement",
    "for_statement",
    "while_statement",
    "herestring_redirect",
    "expansion",
    "simple_expansion",
    "arithmetic_expansion",
}

PROHIBITED_UTILITIES: Set[str] = {
    "sudo",
    "su",
    "pkexec",
    "doas",
    "chroot",
    "nsenter",
    "unshare",
    "bash",
    "sh",
    "dash",
    "zsh",
    "csh",
    "ksh",
    "eval",
    "exec",
    "source",
    "dd",
    "mkfs",
    "fdisk",
    "parted",
    "mount",
    "umount",
    "losetup",
    "nc",
    "netcat",
    "curl",
    "wget",
    "ssh",
    "scp",
    "sftp",
    "iptables",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "init",
    "killall",
}

READ_ONLY_UTILITIES: Set[str] = {
    "ls",
    "stat",
    "file",
    "find",
    "grep",
    "egrep",
    "fgrep",
    "awk",
    "sed",
    "cat",
    "head",
    "tail",
    "wc",
    "sort",
    "uniq",
    "cut",
    "tr",
    "diff",
    "df",
    "du",
    "ps",
    "top",
    "free",
    "uptime",
    "uname",
    "id",
    "whoami",
    "pwd",
    "realpath",
    "readlink",
    "md5sum",
    "sha256sum",
    "journalctl",
    "cd",
    "echo",
}

BOUNDED_MUTATION_UTILITIES: Set[str] = {
    "mkdir",
    "touch",
    "cp",
    "mv",
    "rm",
    "ln",
    "chmod",
    "chown",
    "tar",
    "gzip",
    "gunzip",
}


class StaticPolicyGate:
    """
    Deterministic fail-closed policy enforcement engine using tree-sitter-bash.
    """

    def __init__(self):
        self.parser = Parser(BASH_LANGUAGE)

    def _walk_ast(self, node: Node) -> List[Node]:
        nodes = [node]
        for child in node.children:
            nodes.extend(self._walk_ast(child))
        return nodes

    def _resolve_path_symlink_aware(
        self,
        raw_path: str,
        working_dir: str,
        symlink_map: Dict[str, str],
    ) -> str:
        if not raw_path.startswith("/"):
            candidate = posixpath.normpath(posixpath.join(working_dir, raw_path))
        else:
            candidate = posixpath.normpath(raw_path)

        visited: Set[str] = set()
        while candidate in symlink_map:
            if candidate in visited:
                return "/__SYMLINK_LOOP__"
            visited.add(candidate)
            target = symlink_map[candidate]
            if not target.startswith("/"):
                candidate = posixpath.normpath(
                    posixpath.join(posixpath.dirname(candidate), target)
                )
            else:
                candidate = posixpath.normpath(target)

        parts = candidate.strip("/").split("/")
        accum = ""
        for idx, part in enumerate(parts):
            accum = f"{accum}/{part}" if accum else f"/{part}"
            if accum in symlink_map:
                target = symlink_map[accum]
                rest = "/".join(parts[idx + 1 :])
                candidate = posixpath.normpath(f"{target}/{rest}" if rest else target)
                break
        return candidate

    def _is_within_allowed_roots(
        self, norm_path: str, allowed_roots: List[str]
    ) -> bool:
        for root in allowed_roots:
            clean_root = posixpath.normpath(root)
            if norm_path == clean_root or norm_path.startswith(clean_root + "/"):
                return True
        return False

    def evaluate(
        self,
        action: StructuredAction,
        allowed_roots: List[str],
        symlink_map: Dict[str, str],
    ) -> GateVerdict:
        reasons: List[str] = []
        normalized_paths: List[str] = []

        norm_wd = self._resolve_path_symlink_aware(
            action.working_dir_id, "/workspace", symlink_map
        )
        normalized_paths.append(norm_wd)

        if not self._is_within_allowed_roots(norm_wd, allowed_roots):
            return GateVerdict(
                static_decision=DecisionClass.DENY,
                semantic_decision=action.semantic_risk,
                final_decision=DecisionClass.DENY,
                reasons=[
                    f"Working directory '{norm_wd}' escapes allowed roots {allowed_roots}."
                ],
                normalized_paths=normalized_paths,
            )

        forbidden_tokens = {"|", "||", "&&", ";", "&", ">", ">>", "<", "<<", ">>>"}

        # Validate each command in the sequence independently
        for cmd in action.commands:
            tokens = [cmd.utility] + [str(a) for a in cmd.args]
            cmd_str = " ".join(shlex.quote(t) for t in tokens)

            tree = self.parser.parse(cmd_str.encode("utf-8"))
            if tree.root_node.has_error:
                return GateVerdict(
                    static_decision=DecisionClass.DENY,
                    semantic_decision=action.semantic_risk,
                    final_decision=DecisionClass.DENY,
                    reasons=["tree-sitter-bash detected syntax or parse error node."],
                    normalized_paths=normalized_paths,
                )

            all_nodes = self._walk_ast(tree.root_node)
            for node in all_nodes:
                if node.is_missing or node.type in FORBIDDEN_AST_NODES:
                    return GateVerdict(
                        static_decision=DecisionClass.DENY,
                        semantic_decision=action.semantic_risk,
                        final_decision=DecisionClass.DENY,
                        reasons=[
                            f"Forbidden or undecidable AST node encountered: '{node.type}'."
                        ],
                        normalized_paths=normalized_paths,
                    )
                if node.type in (
                    "pipeline",
                    "list",
                    "redirected_statement",
                    "file_redirect",
                ):
                    return GateVerdict(
                        static_decision=DecisionClass.DENY,
                        semantic_decision=action.semantic_risk,
                        final_decision=DecisionClass.DENY,
                        reasons=[
                            f"Compound construct '{node.type}' rejected within atomic command."
                        ],
                        normalized_paths=normalized_paths,
                    )

            for arg in cmd.args:
                if arg in forbidden_tokens or any(
                    sub in arg
                    for sub in (
                        "$(",
                        "`",
                        "${",
                        "base64",
                        "/dev/tcp",
                        "/dev/sd",
                        "/dev/nvme",
                        "/proc/",
                        "/sys/",
                    )
                ):
                    return GateVerdict(
                        static_decision=DecisionClass.DENY,
                        semantic_decision=action.semantic_risk,
                        final_decision=DecisionClass.DENY,
                        reasons=[
                            f"Argument '{arg}' contains prohibited compound shell operator or token."
                        ],
                        normalized_paths=normalized_paths,
                    )

            util = os.path.basename(cmd.utility)
            if util in PROHIBITED_UTILITIES:
                return GateVerdict(
                    static_decision=DecisionClass.DENY,
                    semantic_decision=action.semantic_risk,
                    final_decision=DecisionClass.DENY,
                    reasons=[f"Utility '{util}' is prohibited."],
                    normalized_paths=normalized_paths,
                )

            if (
                util not in READ_ONLY_UTILITIES
                and util not in BOUNDED_MUTATION_UTILITIES
            ):
                return GateVerdict(
                    static_decision=DecisionClass.DENY,
                    semantic_decision=action.semantic_risk,
                    final_decision=DecisionClass.DENY,
                    reasons=[f"Unsupported utility '{util}' (fail-closed policy)."],
                    normalized_paths=normalized_paths,
                )

            if util == "find":
                for forbidden_flag in (
                    "-exec",
                    "-execdir",
                    "-ok",
                    "-okdir",
                    "-fprint",
                    "-fls",
                ):
                    if forbidden_flag in cmd.args:
                        return GateVerdict(
                            static_decision=DecisionClass.DENY,
                            semantic_decision=action.semantic_risk,
                            final_decision=DecisionClass.DENY,
                            reasons=[
                                f"'find' flag '{forbidden_flag}' permits arbitrary command execution."
                            ],
                            normalized_paths=normalized_paths,
                        )
                if "-delete" in cmd.args:
                    return GateVerdict(
                        static_decision=DecisionClass.ASK,
                        semantic_decision=action.semantic_risk,
                        final_decision=DecisionClass.ASK,
                        reasons=[
                            "'find -delete' mutates filesystem; requires explicit user confirmation."
                        ],
                        normalized_paths=normalized_paths,
                    )

            if util == "tar" and any(
                f in cmd.args for f in ("-P", "--absolute-names", "--to-command")
            ):
                return GateVerdict(
                    static_decision=DecisionClass.DENY,
                    semantic_decision=action.semantic_risk,
                    final_decision=DecisionClass.DENY,
                    reasons=[
                        "'tar' absolute paths or --to-command execution prohibited."
                    ],
                    normalized_paths=normalized_paths,
                )

            if util == "chmod" and any(
                mode in cmd.args for mode in ("777", "a+rwx", "+s", "u+s", "4755")
            ):
                return GateVerdict(
                    static_decision=DecisionClass.DENY,
                    semantic_decision=action.semantic_risk,
                    final_decision=DecisionClass.DENY,
                    reasons=[
                        "Insecure permission mode (world-writable or SUID bit) prohibited."
                    ],
                    normalized_paths=normalized_paths,
                )

            for arg in cmd.args:
                if arg.startswith("-"):
                    continue
                if (
                    "/" in arg
                    or arg in (".", "..")
                    or util in BOUNDED_MUTATION_UTILITIES
                ):
                    resolved = self._resolve_path_symlink_aware(
                        arg, norm_wd, symlink_map
                    )
                    normalized_paths.append(resolved)
                    if not self._is_within_allowed_roots(resolved, allowed_roots):
                        return GateVerdict(
                            static_decision=DecisionClass.DENY,
                            semantic_decision=action.semantic_risk,
                            final_decision=DecisionClass.DENY,
                            reasons=[
                                f"Normalized path '{resolved}' resolves outside allowed roots {allowed_roots}."
                            ],
                            normalized_paths=normalized_paths,
                        )

            if util == "rm":
                has_recursive = any(
                    a in ("-r", "-R", "-rf", "-fr", "--recursive") for a in cmd.args
                )
                has_wildcard = any("*" in a or "?" in a for a in cmd.args)
                for p in normalized_paths:
                    if p in [posixpath.normpath(r) for r in allowed_roots] or p == "/":
                        return GateVerdict(
                            static_decision=DecisionClass.DENY,
                            semantic_decision=action.semantic_risk,
                            final_decision=DecisionClass.DENY,
                            reasons=[
                                f"Attempted deletion of task root or system root '{p}'."
                            ],
                            normalized_paths=normalized_paths,
                        )
                if has_recursive or has_wildcard:
                    return GateVerdict(
                        static_decision=DecisionClass.ASK,
                        semantic_decision=action.semantic_risk,
                        final_decision=DecisionClass.ASK,
                        reasons=[
                            "Recursive or wildcard removal requires user confirmation (Ask)."
                        ],
                        normalized_paths=normalized_paths,
                    )

        reasons.append(
            "Static AST and path verification passed for all commands in sequence."
        )
        return GateVerdict(
            static_decision=DecisionClass.ALLOW,
            semantic_decision=action.semantic_risk,
            final_decision=DecisionClass.ALLOW,
            reasons=reasons,
            normalized_paths=list(set(normalized_paths)),
        )
