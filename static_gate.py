from __future__ import annotations
import os
import posixpath
from typing import Dict, List, Set, Tuple
import tree_sitter_bash as tsbash
from tree_sitter import Language, Parser, Node

from schemas import DecisionClass, GateVerdict, StructuredAction


BASH_LANGUAGE = Language(tsbash.language())

# Forbidden AST node types that violate fail-closed static decidability
FORBIDDEN_AST_NODES: Set[str] = {
    "ERROR",
    "MISSING",
    "command_substitution",  # $(...) or `...`
    "process_substitution",  # <(...) or >(...)
    "function_definition",  # foo() { ... }
    "subshell",  # (...)
    "c_style_for_statement",
    "for_statement",
    "while_statement",
    "herestring_redirect",
    "expansion",  # ${VAR:-...}
    "simple_expansion",  # $VAR
    "arithmetic_expansion",  # $((1+1))
}

# Utilities unconditionally prohibited (privilege escalation, raw disk, nested shells, network/eval)
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

# Read-only diagnostic utilities eligible for Allow inside authorized roots
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
}

# Bounded mutation utilities allowed ONLY inside authorized task roots with safe flags
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
        """
        Normalizes `raw_path` relative to `working_dir` and resolves known container
        symlinks iteratively (detecting symlink loops) to prevent traversal escapes.
        """
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

        # Check prefix components for directory symlinks
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

        # 1. Verify working directory is inside allowed roots
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

        # 2. Parse full reconstructed command via tree-sitter-bash
        cmd_str = action.to_command_preview()
        tree = self.parser.parse(cmd_str.encode("utf-8"))
        root_node = tree.root_node

        if root_node.has_error:
            return GateVerdict(
                static_decision=DecisionClass.DENY,
                semantic_decision=action.semantic_risk,
                final_decision=DecisionClass.DENY,
                reasons=["tree-sitter-bash detected syntax or parse error node."],
                normalized_paths=normalized_paths,
            )

        all_nodes = self._walk_ast(root_node)
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
            # Analyze compound commands (pipeline, list) holistically; reject embedded operators in single action
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
                        f"Compound construct or shell redirection '{node.type}' rejected in atomic action."
                    ],
                    normalized_paths=normalized_paths,
                )

        # 3. Check raw argument tokens for obfuscation, dynamic evaluation, or embedded shell syntax
        forbidden_tokens = {"|", "||", "&&", ";", "&", ">", ">>", "<", "<<", ">>>"}
        for arg in action.args:
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
                        f"Argument '{arg}' contains prohibited compound shell operator or dynamic/device/system token."
                    ],
                    normalized_paths=normalized_paths,
                )
        # 4. Utility-level allowlist & rule inspection
        util = os.path.basename(action.utility)
        if util in PROHIBITED_UTILITIES:
            return GateVerdict(
                static_decision=DecisionClass.DENY,
                semantic_decision=action.semantic_risk,
                final_decision=DecisionClass.DENY,
                reasons=[f"Utility '{util}' is in the prohibited utility set."],
                normalized_paths=normalized_paths,
            )

        if util not in READ_ONLY_UTILITIES and util not in BOUNDED_MUTATION_UTILITIES:
            return GateVerdict(
                static_decision=DecisionClass.DENY,
                semantic_decision=action.semantic_risk,
                final_decision=DecisionClass.DENY,
                reasons=[f"Unsupported utility '{util}' (fail-closed policy)."],
                normalized_paths=normalized_paths,
            )

        # 5. Utility-specific flag hazards (e.g., find -exec / -delete, sed -i outside root, tar -P, chmod 777)
        if util == "find":
            for forbidden_flag in (
                "-exec",
                "-execdir",
                "-ok",
                "-okdir",
                "-fprint",
                "-fls",
            ):
                if forbidden_flag in action.args:
                    return GateVerdict(
                        static_decision=DecisionClass.DENY,
                        semantic_decision=action.semantic_risk,
                        final_decision=DecisionClass.DENY,
                        reasons=[
                            f"'find' flag '{forbidden_flag}' permits arbitrary command execution or file clobbering."
                        ],
                        normalized_paths=normalized_paths,
                    )
            if "-delete" in action.args:
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
            f in action.args for f in ("-P", "--absolute-names", "--to-command")
        ):
            return GateVerdict(
                static_decision=DecisionClass.DENY,
                semantic_decision=action.semantic_risk,
                final_decision=DecisionClass.DENY,
                reasons=["'tar' absolute paths or --to-command execution prohibited."],
                normalized_paths=normalized_paths,
            )

        if util == "chmod" and any(
            mode in action.args for mode in ("777", "a+rwx", "+s", "u+s", "4755")
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

        # 6. Path argument normalization & boundary enforcement
        for arg in action.args:
            if arg.startswith("-"):
                continue
            # Treat path-like arguments or arguments to filesystem utilities as candidate paths
            if "/" in arg or arg in (".", "..") or util in BOUNDED_MUTATION_UTILITIES:
                resolved = self._resolve_path_symlink_aware(arg, norm_wd, symlink_map)
                normalized_paths.append(resolved)
                if not self._is_within_allowed_roots(resolved, allowed_roots):
                    return GateVerdict(
                        static_decision=DecisionClass.DENY,
                        semantic_decision=action.semantic_risk,
                        final_decision=DecisionClass.DENY,
                        reasons=[
                            f"Normalized path '{resolved}' (from '{arg}') resolves outside allowed roots {allowed_roots}."
                        ],
                        normalized_paths=normalized_paths,
                    )

        # 7. Destructive scope checks for `rm`
        if util == "rm":
            has_recursive = any(
                a in ("-r", "-R", "-rf", "-fr", "--recursive") for a in action.args
            )
            has_wildcard = any("*" in a or "?" in a for a in action.args)
            for p in normalized_paths[1:]:
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

        reasons.append(f"Static AST and path verification passed for utility '{util}'.")
        return GateVerdict(
            static_decision=DecisionClass.ALLOW,
            semantic_decision=action.semantic_risk,
            final_decision=DecisionClass.ALLOW,
            reasons=reasons,
            normalized_paths=normalized_paths,
        )
