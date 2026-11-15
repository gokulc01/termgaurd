from __future__ import annotations
import posixpath
import re
from typing import Callable, Dict, List, Optional, Set

import tree_sitter_bash as tsbash
from tree_sitter import Language, Node, Parser

from schemas import DecisionClass, GateVerdict, StructuredAction


BASH_LANGUAGE = Language(tsbash.language())

# Batch resolver: maps absolute (possibly un-normalized) paths to their real paths,
# e.g. `realpath -m` inside the sandbox container, or os.path.realpath on the host.
PathResolver = Callable[[List[str]], List[str]]

SYMLINK_LOOP = "/__SYMLINK_LOOP__"

# The rendered command line is built only from shlex-quoted tokens joined by `|` / `&&`,
# so this whitelist is exactly what a legitimate rendering can contain. Anything else
# (variable_assignment, redirections, substitutions, ...) means something slipped past
# rendering and is denied.
ALLOWED_AST_NODES: Set[str] = {
    "program",
    "list",
    "pipeline",
    "command",
    "command_name",
    "word",
    "number",
    "raw_string",
    "concatenation",  # shlex.quote("it's") -> 'it'"'"'s'
    "string",
    "string_content",
    '"',
    "|",
    "&&",
}

PROHIBITED_UTILITIES: Set[str] = {
    # privilege / namespace
    "sudo", "su", "pkexec", "doas", "chroot", "nsenter", "unshare", "setpriv", "capsh",
    # shells and interpreters (arbitrary code)
    "bash", "sh", "dash", "zsh", "csh", "tcsh", "ksh", "fish", "busybox",
    "python", "python3", "perl", "ruby", "node", "php", "lua", "tclsh", "expect",
    # command wrappers (would let a prohibited command hide behind an allowed one)
    "env", "xargs", "nohup", "setsid", "timeout", "nice", "ionice", "stdbuf",
    "script", "watch", "strace", "ltrace", "gdb",
    # shell builtins that evaluate code
    "eval", "exec", "source", ".", "trap", "alias",
    # disks and devices
    "dd", "mkfs", "fdisk", "parted", "mount", "umount", "losetup", "wipefs", "shred",
    # network
    "nc", "netcat", "ncat", "socat", "curl", "wget", "ssh", "scp", "sftp", "rsync",
    "telnet", "ftp",
    "iptables", "nft",
    # system state
    "shutdown", "reboot", "halt", "poweroff", "init", "systemctl", "service",
    "kill", "killall", "pkill", "crontab", "at",
    "chattr", "docker", "kubectl",
}

# Builtins that need shell state the agent does not model.
UNSUPPORTED_BUILTINS: Set[str] = {
    "export", "set", "unset", "pushd", "popd", "ulimit", "read", "declare", "local",
}

READ_ONLY_UTILITIES: Set[str] = {
    "ls", "stat", "file", "find", "grep", "egrep", "fgrep", "awk", "sed", "cat",
    "head", "tail", "wc", "sort", "uniq", "cut", "tr", "diff", "df", "du", "ps",
    "free", "uptime", "uname", "id", "whoami", "pwd", "realpath", "readlink",
    "md5sum", "sha256sum", "journalctl", "echo", "date",
}

BOUNDED_MUTATION_UTILITIES: Set[str] = {
    "mkdir", "touch", "cp", "mv", "rm", "ln", "chmod", "chown", "tar", "gzip", "gunzip",
}

# Utilities with dedicated handlers below.
SPECIAL_UTILITIES: Set[str] = {"cd", "umask", "git", "top"}

SUPPORTED_UTILITIES: Set[str] = (
    READ_ONLY_UTILITIES | BOUNDED_MUTATION_UTILITIES | SPECIAL_UTILITIES
)

FORBIDDEN_SUBSTRINGS = (
    "$(", "`", "${", "base64", "/dev/tcp", "/dev/udp", "/dev/sd", "/dev/nvme",
    "/dev/mem", "/proc/", "/sys/",
)
# Bare shell operators passed as arguments. They are quoted (inert) when rendered, but
# they mean the model tried to compose shell syntax inside one command.
SHELL_OPERATOR_TOKENS = {"|", "||", "&&", ";", "&", ">", ">>", "<", "<<", "<<<", ">|"}

# Options whose *next* argument is a pattern/value rather than a filesystem path.
PATTERN_VALUE_OPTIONS: Dict[str, Set[str]] = {
    "grep": {"-e", "--regexp", "-m", "--max-count", "-A", "-B", "-C"},
    "egrep": {"-e", "--regexp", "-m", "-A", "-B", "-C"},
    "fgrep": {"-e", "--regexp", "-m", "-A", "-B", "-C"},
    "sed": {"-e", "--expression"},
    "awk": {"-v", "-F", "-e", "--source"},
    "find": {"-name", "-iname", "-path", "-ipath", "-regex", "-iregex", "-wholename",
             "-iwholename", "-lname", "-ilname", "-type", "-mtime", "-mmin", "-atime",
             "-amin", "-ctime", "-cmin", "-size", "-maxdepth", "-mindepth", "-perm",
             "-user", "-group", "-printf"},
    "cut": {"-d", "-f", "-c", "-b"},
    "sort": {"-t", "-k"},
    "head": {"-n", "-c"},
    "tail": {"-n", "-c"},
    "date": {"-d", "--date"},
    "journalctl": {"-u", "-n", "-p", "--since", "--until"},
    "du": {"-d", "--max-depth"},
}
# Utilities whose first positional argument is a pattern/script (unless -e/-f is used).
PATTERN_FIRST_UTILITIES = {"grep", "egrep", "fgrep", "awk", "sed"}
# Utilities whose positional args are never filesystem paths.
NON_PATH_UTILITIES = {"echo", "tr", "date", "umask", "uname", "id", "whoami"}

# Flags that make a utility follow symlinks out of the checked path set.
FOLLOW_SYMLINK_FLAGS: Dict[str, Set[str]] = {
    "find": {"-L", "-follow"},
    "grep": {"--dereference-recursive"},
    "egrep": {"--dereference-recursive"},
    "fgrep": {"--dereference-recursive"},
    "tar": {"--dereference"},
    "cp": {"--dereference"},
}
# Same thing as letters inside short-option clusters (e.g. grep -Ri).
FOLLOW_SYMLINK_SHORT: Dict[str, str] = {
    "grep": "R", "egrep": "R", "fgrep": "R", "tar": "h", "cp": "L",
}

FIND_EXEC_FLAGS = ("-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0",
                   "-fprintf", "-fls")

TAR_FORBIDDEN_LONG = (
    "--absolute-names", "--to-command", "--checkpoint-action",
    "--use-compress-program", "--rsh-command", "--info-script",
    "--new-volume-script", "--rmt-command",
)
TAR_FORBIDDEN_SHORT = set("PIF")  # -P absolute names, -I compress program, -F info script

GIT_READONLY_SUBCOMMANDS = {
    "status", "log", "diff", "show", "ls-files", "rev-parse", "blame", "shortlog",
    "describe", "grep", "branch", "tag",
}
GIT_FORBIDDEN_ARG_PREFIXES = (
    "--output", "--ext-diff", "--textconv", "--open-files-in-pager", "-O",
    "--exec", "--upload-pack", "--receive-pack",
)
GIT_BRANCH_LIST_FLAGS = {
    "-a", "--all", "-r", "--remotes", "-v", "-vv", "--verbose", "-l", "--list",
    "--show-current", "--merged", "--no-merged", "--no-color", "--color",
}

# --- sed: only a whitelisted, side-effect-free subset (no e / w / W / r / R) --------
_SED_ADDR_ATOM = r"(?:\d+|\$|/(?:\\.|[^/\\\n])*/I?)"
_SED_ADDR = rf"(?:{_SED_ADDR_ATOM}(?:\s*,\s*(?:{_SED_ADDR_ATOM}|[+~]\d+))?\s*!?\s*)"
_SED_S = r"s(?P<d>[^\\\n\sa-zA-Z0-9])(?:\\.|(?!(?P=d)).)*(?P=d)(?:\\.|(?!(?P=d)).)*(?P=d)[gGiIpPmM0-9]*"
_SED_Y = r"y/(?:\\.|[^/\\\n])*/(?:\\.|[^/\\\n])*/"
_SED_SIMPLE = r"[pPdDqQ=nNgGhHxlz]"
_SED_CMD = rf"{_SED_ADDR}?(?:{_SED_S}|{_SED_Y}|{_SED_SIMPLE})"
_SED_CMD_RE = re.compile(rf"\s*{_SED_CMD}\s*", re.S)


def sed_script_supported(script: str) -> bool:
    """Match one whitelisted command at a time, separated by ';' or newlines."""
    pos = 0
    while True:
        m = _SED_CMD_RE.match(script, pos)
        if not m:
            return False
        pos = m.end()
        if pos == len(script):
            return True
        if script[pos] not in ";\n":
            return False
        pos += 1
        if script[pos:].strip() == "":
            return True

AWK_PRINT_REDIRECT_RE = re.compile(r"\bprintf?\b[^;}]*>")

CHMOD_OPTIONS = {"-R", "-v", "-c", "-f", "--recursive", "--verbose", "--changes",
                 "--silent", "--quiet"}
CHMOD_SYMBOLIC_RE = re.compile(r"^([ugoa]*)((?:[-+=][rwxXstugo]*)+)$")


class _Deny(Exception):
    def __init__(self, reason: str, recoverable: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.recoverable = recoverable


def _is_short_cluster(arg: str) -> bool:
    return arg.startswith("-") and not arg.startswith("--") and len(arg) > 1


class StaticPolicyGate:
    """
    Deterministic fail-closed policy engine.

    What it actually enforces (describe it this way in the report):
      1. Utility allowlist + prohibited list (bare names only).
      2. Per-utility argument rules (find/tar/chmod/rm/git/awk/sed/...).
      3. Path confinement of every path-like operand, including option values
         (--opt=/x, -o/x), resolved per command with the working directory as it is
         at that point in the sequence (cd is tracked) and with symlinks created
         earlier in the sequence (ln -s is tracked). If a resolver is given
         (realpath in the sandbox, or on the host), real symlinks are resolved too.
      4. A tree-sitter whitelist over the exact command line that will be executed
         and shown, so only quoted words joined by `|` / `&&` can get through.
    """

    def __init__(self):
        self.parser = Parser(BASH_LANGUAGE)

    # ------------------------------------------------------------------ helpers
    def _walk_ast(self, node: Node) -> List[Node]:
        nodes = [node]
        for child in node.children:
            nodes.extend(self._walk_ast(child))
        return nodes

    def _resolve_with_map(self, path: str, symlink_map: Dict[str, str], depth: int = 0) -> str:
        """Component-wise resolution (realpath semantics: `link/..` follows the link first)."""
        if depth > 40:
            return SYMLINK_LOOP
        comps = [c for c in path.split("/") if c not in ("", ".")]
        cur = "/"
        for c in comps:
            if c == "..":
                cur = posixpath.dirname(cur)
                continue
            nxt = posixpath.join(cur, c)
            if nxt in symlink_map:
                target = symlink_map[nxt]
                base = target if target.startswith("/") else posixpath.join(cur, target)
                cur = self._resolve_with_map(base, symlink_map, depth + 1)
                if cur == SYMLINK_LOOP:
                    return cur
            else:
                cur = nxt
        return cur

    def _resolve_many(
        self,
        raw_paths: List[str],
        working_dir: str,
        symlink_map: Dict[str, str],
        resolver: Optional[PathResolver],
    ) -> List[List[str]]:
        """Return, per raw path, every resolution that must lie within the roots."""
        joined = [
            p if p.startswith("/") else posixpath.join(working_dir, p) for p in raw_paths
        ]
        out = [[self._resolve_with_map(j, symlink_map)] for j in joined]
        if resolver is not None and joined:
            try:
                real = resolver(joined)
            except Exception as exc:  # fail closed
                raise _Deny(f"Path resolution failed: {exc}")
            if len(real) != len(joined):
                raise _Deny("Path resolution returned an unexpected number of results.")
            for lst, r in zip(out, real):
                lst.append(posixpath.normpath(r))
        return out

    @staticmethod
    def _within(path: str, roots: List[str]) -> bool:
        if path == SYMLINK_LOOP:
            return False
        for root in roots:
            if path == root or path.startswith(root.rstrip("/") + "/"):
                return True
        return False

    def _check_paths(
        self,
        raw_paths: List[str],
        working_dir: str,
        symlink_map: Dict[str, str],
        resolver: Optional[PathResolver],
        roots: List[str],
        normalized_paths: List[str],
    ) -> List[List[str]]:
        resolved = self._resolve_many(raw_paths, working_dir, symlink_map, resolver)
        for raw, options in zip(raw_paths, resolved):
            for r in options:
                normalized_paths.append(r)
                if not self._within(r, roots):
                    raise _Deny(
                        f"Path '{raw}' resolves to '{r}', outside allowed roots {roots}."
                    )
        return resolved

    def _path_operands(self, util: str, args: List[str]) -> List[str]:
        """Collect every argument (or option value) that may be read as a path."""
        if util in NON_PATH_UTILITIES:
            return []
        if util == "chmod":
            return self._chmod_operands(args)
        value_opts = PATTERN_VALUE_OPTIONS.get(util, set())
        has_explicit_script = any(
            a in ("-e", "--regexp", "--expression", "-f", "--file", "--source")
            or a.startswith(("--regexp=", "--expression=", "--file=", "--source="))
            for a in args
        )
        skip_first_positional = util in PATTERN_FIRST_UTILITIES and not has_explicit_script
        if util == "chown":
            skip_first_positional = True  # owner[:group]

        operands: List[str] = []
        end_of_opts = False
        skip_next = False
        for arg in args:
            if skip_next:
                skip_next = False
                continue
            if not end_of_opts and arg == "--":
                end_of_opts = True
                continue
            if not end_of_opts and arg.startswith("-") and arg != "-":
                if arg in value_opts:
                    skip_next = True
                elif arg.startswith("--") and "=" in arg:
                    name, value = arg.split("=", 1)
                    if name not in value_opts:
                        operands.append(value)
                elif _is_short_cluster(arg) and len(arg) > 2 and ("/" in arg or ".." in arg):
                    if arg[:2] not in value_opts:  # -F/ or -d/ are separators, not paths
                        operands.append(arg[2:])  # glued value, e.g. -o/etc/passwd, -C../x
                continue
            if skip_first_positional:
                skip_first_positional = False
                continue
            operands.append(arg)
        return operands

    @staticmethod
    def _chmod_operands(args: List[str]) -> List[str]:
        """chmod modes may start with '-' (chmod -w f), so identify the mode explicitly."""
        operands: List[str] = []
        mode_seen = False
        for a in args:
            if a in CHMOD_OPTIONS:
                continue
            if a.startswith("--"):
                if "=" in a:
                    operands.append(a.split("=", 1)[1])  # --reference=FILE
                    mode_seen = True
                continue
            if not mode_seen and (CHMOD_SYMBOLIC_RE.match(a) or re.fullmatch(r"[0-7]{1,4}", a)
                                  or not a.startswith("-")):
                mode_seen = True
                continue
            operands.append(a)
        return operands

    @staticmethod
    def _positionals(args: List[str], option_values: Set[str] = frozenset()) -> List[str]:
        pos: List[str] = []
        end_of_opts = False
        skip_next = False
        for a in args:
            if skip_next:
                skip_next = False
                continue
            if not end_of_opts and a == "--":
                end_of_opts = True
                continue
            if not end_of_opts and a.startswith("-") and a != "-":
                if a in option_values:
                    skip_next = True
                continue
            pos.append(a)
        return pos

    # -------------------------------------------------------- per-utility rules
    def _check_find(self, args: List[str], asks: List[str]) -> None:
        for flag in FIND_EXEC_FLAGS:
            if flag in args:
                raise _Deny(f"'find' flag '{flag}' permits command execution or file writes.")
        if "-delete" in args:
            asks.append("'find -delete' mutates the filesystem; requires explicit confirmation.")

    def _check_tar(self, args: List[str]) -> None:
        for i, a in enumerate(args):
            if a.startswith("--"):
                name = a.split("=", 1)[0]
                if len(name) > 4 and any(f.startswith(name) for f in TAR_FORBIDDEN_LONG):
                    raise _Deny(f"'tar' option '{a}' (absolute paths or command execution) prohibited.")
            elif _is_short_cluster(a) or (i == 0 and re.fullmatch(r"[A-Za-z]+", a)):
                letters = a.lstrip("-")
                bad = TAR_FORBIDDEN_SHORT.intersection(letters)
                if bad:
                    raise _Deny(f"'tar' flag(s) {sorted(bad)} (absolute paths or command execution) prohibited.")

    def _check_chmod(self, args: List[str], asks: List[str]) -> None:
        mode = None
        for a in args:
            if a in CHMOD_OPTIONS or a.startswith("--reference"):
                continue
            if CHMOD_SYMBOLIC_RE.match(a) or re.fullmatch(r"[0-7]{1,4}", a):
                mode = a
                break
            if a.startswith("-"):
                continue
            mode = a
            break
        if mode is None:
            if any(a.startswith("--reference") for a in args):
                asks.append("'chmod --reference' copies an unknown mode; confirm.")
                return
            raise _Deny("'chmod' without a mode.", recoverable=True)

        if re.fullmatch(r"[0-7]{1,4}", mode):
            v = int(mode, 8)
            if v & 0o6000:
                raise _Deny(f"chmod mode '{mode}' sets SUID/SGID; prohibited.")
            if v & 0o002:
                raise _Deny(f"chmod mode '{mode}' is world-writable; prohibited.")
            return

        for clause in mode.split(","):
            m = CHMOD_SYMBOLIC_RE.match(clause)
            if not m:
                raise _Deny(f"Unparseable chmod mode '{mode}'.", recoverable=True)
            who, ops = m.group(1), m.group(2)
            for op, perms in re.findall(r"([-+=])([rwxXstugo]*)", ops):
                if op == "-":
                    continue
                if "s" in perms:
                    raise _Deny(f"chmod '{clause}' sets SUID/SGID; prohibited.")
                world = (not who) or ("o" in who) or ("a" in who)
                if "w" in perms and ("o" in who or "a" in who):
                    raise _Deny(f"chmod '{clause}' grants world write; prohibited.")
                if world and ("w" in perms or set(perms) & set("ugo")):
                    asks.append(f"chmod '{clause}' may widen write access; confirm.")

    def _check_rm(
        self,
        args: List[str],
        resolved_targets: List[List[str]],
        roots: List[str],
        asks: List[str],
    ) -> None:
        if any(a == "--no-preserve-root" for a in args):
            raise _Deny("'rm --no-preserve-root' prohibited.")
        for options in resolved_targets:
            for p in options:
                if p == "/" or p in roots or any(r.startswith(p.rstrip("/") + "/") for r in roots):
                    raise _Deny(f"Attempted deletion of task root, its ancestor, or system root '{p}'.")
        recursive = False
        for a in args:
            if a == "--":
                break
            if a == "--recursive" or (_is_short_cluster(a) and set(a[1:]) & {"r", "R"}):
                recursive = True
        wildcard = any(ch in a for a in self._positionals(args) for ch in "*?[")
        if recursive or wildcard:
            asks.append("Recursive or wildcard removal requires user confirmation (Ask).")

    def _check_git(self, args: List[str]) -> None:
        if not args:
            raise _Deny("Bare 'git' with no subcommand.", recoverable=True)
        if args[0].startswith("-"):
            raise _Deny(
                "git global options (-c, -C, --git-dir, --exec-path, ...) are not allowed.",
                recoverable=True,
            )
        sub = args[0]
        if sub not in GIT_READONLY_SUBCOMMANDS:
            raise _Deny(
                f"git subcommand '{sub}' is not on the read-only allowlist "
                f"{sorted(GIT_READONLY_SUBCOMMANDS)}."
            )
        for a in args[1:]:
            if a.startswith(GIT_FORBIDDEN_ARG_PREFIXES):
                raise _Deny(f"git argument '{a}' can write files or run external programs.")
        rest = args[1:]
        if sub == "branch" and any(a not in GIT_BRANCH_LIST_FLAGS for a in rest):
            raise _Deny("Only listing forms of 'git branch' are allowed.")
        if sub == "tag" and rest and rest[0] not in ("-l", "--list"):
            raise _Deny("Only 'git tag' / 'git tag -l [pattern]' are allowed.")

    def _check_awk(self, args: List[str]) -> None:
        for a in args:
            if a in ("-f", "--file", "-E", "--exec", "-i", "--include", "-l", "--load") or \
                    a.startswith(("--file=", "--exec=", "--include=", "--load=")):
                raise _Deny(f"awk option '{a}' loads uninspectable program code.")
        programs: List[str] = []
        it = iter(range(len(args)))
        explicit = False
        for i in it:
            a = args[i]
            if a in ("-e", "--source") and i + 1 < len(args):
                programs.append(args[i + 1])
                explicit = True
                next(it, None)
            elif a in ("-v", "-F") and i + 1 < len(args):
                next(it, None)
        if not explicit:
            pos = self._positionals(args, {"-v", "-F"})
            if not pos:
                raise _Deny("awk without a program.", recoverable=True)
            programs.append(pos[0])
        for prog in programs:
            if re.search(r"\bsystem\b|\bgetline\b|@", prog):
                raise _Deny("awk program uses system()/getline/@load; prohibited.")
            if "|" in prog or AWK_PRINT_REDIRECT_RE.search(prog):
                raise _Deny(
                    "awk pipes or print redirections are not supported; use pipe_to_next "
                    "and plain print instead.",
                    recoverable=True,
                )

    def _check_sed(self, args: List[str]) -> None:
        scripts: List[str] = []
        i = 0
        explicit = False
        while i < len(args):
            a = args[i]
            if a in ("-f", "--file") or a.startswith("--file="):
                raise _Deny("sed -f loads an uninspectable script.")
            if a in ("-e", "--expression") and i + 1 < len(args):
                scripts.append(args[i + 1])
                explicit = True
                i += 2
                continue
            if a.startswith("--expression="):
                scripts.append(a.split("=", 1)[1])
                explicit = True
            i += 1
        if not explicit:
            pos = self._positionals(args)
            if not pos:
                raise _Deny("sed without a script.", recoverable=True)
            scripts.append(pos[0])
        for s in scripts:
            if not sed_script_supported(s):
                raise _Deny(
                    f"sed script '{s}' is outside the supported subset (s///, y///, "
                    "p/d/q with line or /regex/ addresses; no e/w/r commands or flags).",
                    recoverable=True,
                )

    def _check_follow_flags(self, util: str, args: List[str]) -> None:
        longs = FOLLOW_SYMLINK_FLAGS.get(util, set())
        short = FOLLOW_SYMLINK_SHORT.get(util)
        for a in args:
            if a == "--":
                break
            if a in longs or (short and _is_short_cluster(a) and util != "find" and short in a[1:]):
                raise _Deny(
                    f"'{util} {a}' follows symlinks beyond the checked paths; "
                    "use the non-dereferencing form.",
                    recoverable=True,
                )

    # ------------------------------------------------------------------- main
    def evaluate(
        self,
        action: StructuredAction,
        allowed_roots: List[str],
        symlink_map: Dict[str, str],
        default_working_dir: str = "/workspace",
        path_resolver: Optional[PathResolver] = None,
    ) -> GateVerdict:
        normalized_paths: List[str] = []
        asks: List[str] = []
        rendered = action.render_command_line()

        def verdict(decision: DecisionClass, reasons: List[str], recoverable=False) -> GateVerdict:
            return GateVerdict(
                static_decision=decision,
                semantic_decision=action.semantic_risk,
                final_decision=decision,
                reasons=reasons,
                normalized_paths=sorted(set(normalized_paths)),
                recoverable=recoverable,
                rendered_command=rendered,
            )

        try:
            roots = sorted({posixpath.normpath(r) for r in allowed_roots})
            if path_resolver is not None:
                try:
                    roots = sorted(set(roots) | {posixpath.normpath(r) for r in path_resolver(roots)})
                except Exception as exc:
                    raise _Deny(f"Could not resolve allowed roots: {exc}")

            # 1. Whole-line syntax whitelist over the exact string that will run.
            tree = self.parser.parse(rendered.encode("utf-8"))
            if tree.root_node.has_error:
                raise _Deny("tree-sitter-bash reported a parse error on the rendered command.")
            for node in self._walk_ast(tree.root_node):
                if node.is_missing or node.type not in ALLOWED_AST_NODES:
                    raise _Deny(f"Disallowed shell construct '{node.type}' in rendered command.")

            local_map = dict(symlink_map)
            wd_opts = self._check_paths(
                [action.working_dir_id], default_working_dir, local_map,
                path_resolver, roots, normalized_paths,
            )
            wd = wd_opts[-1][-1]

            n = len(action.commands)
            for idx, cmd in enumerate(action.commands):
                util = cmd.utility
                args = [str(a) for a in cmd.args]
                in_pipeline = cmd.pipe_to_next or (idx > 0 and action.commands[idx - 1].pipe_to_next)

                if "/" in util:
                    raise _Deny(f"Utility '{util}' must be a bare name, not a path.", recoverable=True)
                if util in PROHIBITED_UTILITIES:
                    raise _Deny(f"Utility '{util}' is prohibited.")
                if util in UNSUPPORTED_BUILTINS:
                    raise _Deny(f"Shell builtin '{util}' is not supported.", recoverable=True)
                if util not in SUPPORTED_UTILITIES:
                    raise _Deny(
                        f"Unsupported utility '{util}' (fail-closed). Supported: "
                        f"{sorted(SUPPORTED_UTILITIES)}.",
                        recoverable=True,
                    )
                if idx == n - 1 and cmd.pipe_to_next:
                    raise _Deny("The last command cannot set pipe_to_next.", recoverable=True)

                for arg in args:
                    if any(sub in arg for sub in FORBIDDEN_SUBSTRINGS):
                        raise _Deny(f"Argument '{arg}' contains a prohibited token.")
                    if arg in SHELL_OPERATOR_TOKENS:
                        raise _Deny(
                            f"Argument '{arg}' is a shell operator; use separate commands "
                            "with pipe_to_next instead.",
                            recoverable=True,
                        )

                self._check_follow_flags(util, args)
                if util == "find":
                    self._check_find(args, asks)
                elif util == "tar":
                    self._check_tar(args)
                elif util == "chmod":
                    self._check_chmod(args, asks)
                elif util == "git":
                    self._check_git(args)
                elif util == "awk":
                    self._check_awk(args)
                elif util == "sed":
                    self._check_sed(args)
                elif util == "top":
                    letters = "".join(a[1:] for a in args if _is_short_cluster(a))
                    if "b" not in letters or "n" not in letters:
                        raise _Deny("'top' must be non-interactive: use 'top -b -n 1'.", recoverable=True)
                elif util == "umask":
                    pos = self._positionals(args)
                    if len(pos) != 1 or not re.fullmatch(r"[0-7]{3,4}", pos[0]):
                        raise _Deny("'umask' needs exactly one octal mask.", recoverable=True)
                    if not int(pos[0], 8) & 0o002:
                        raise _Deny(f"umask '{pos[0]}' leaves new files world-writable.")

                # Path confinement of every path-like operand at the current wd.
                operands = self._path_operands(util, args)
                resolved = self._check_paths(
                    operands, wd, local_map, path_resolver, roots, normalized_paths
                )

                if util == "rm":
                    targets = self._positionals(args)
                    target_res = self._resolve_many(targets, wd, local_map, path_resolver)
                    self._check_rm(args, target_res, roots, asks)

                if util == "ln" and any(
                    a in ("-s", "--symbolic") or (_is_short_cluster(a) and "s" in a[1:])
                    for a in args
                ):
                    pos = self._positionals(args)
                    if any(a in ("-t", "--target-directory") or a.startswith("--target-directory=") for a in args) \
                            or len(pos) not in (1, 2):
                        raise _Deny("Only 'ln -s TARGET [LINK]' forms are supported.", recoverable=True)
                    target = pos[0]
                    link = pos[1] if len(pos) == 2 else posixpath.basename(target.rstrip("/"))
                    link_abs = self._resolve_with_map(
                        link if link.startswith("/") else posixpath.join(wd, link), local_map
                    )
                    # A relative symlink target is interpreted relative to the link's directory.
                    if not target.startswith("/"):
                        self._check_paths(
                            [target], posixpath.dirname(link_abs), local_map,
                            path_resolver, roots, normalized_paths,
                        )
                    local_map[link_abs] = target

                if util == "cd":
                    if in_pipeline:
                        raise _Deny("'cd' cannot be part of a pipeline.", recoverable=True)
                    pos = self._positionals(args)
                    if len(pos) != 1:
                        raise _Deny("'cd' needs exactly one target directory.", recoverable=True)
                    wd = resolved[0][-1]

        except _Deny as d:
            return verdict(DecisionClass.DENY, [d.reason], recoverable=d.recoverable)

        if asks:
            return verdict(DecisionClass.ASK, asks)
        return verdict(
            DecisionClass.ALLOW,
            ["Static syntax, argument, and path verification passed for all commands."],
        )
