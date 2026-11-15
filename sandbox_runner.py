from __future__ import annotations
import time
from typing import List, Optional

import docker
from docker.errors import APIError

from schemas import SandboxResult, StructuredAction


class FixtureError(RuntimeError):
    """A fixture setup command failed; the task's initial state is invalid."""


class SandboxRunner:
    """
    Executes gate-validated StructuredAction objects inside an isolated Ubuntu container.

    Execution model: the action's canonical rendering (StructuredAction.render_command_line,
    the same string the gate parsed and the CLI prints) runs as
        timeout -k <grace> <limit> sh -c '<rendered>'
    as the unprivileged user. Every token in the rendering is shlex-quoted, so `sh` only
    interprets the `|` / `&&` joiners. This gives real pipelines and `cd` that persists
    across `&&`, with identical semantics to what the user sees.
    """

    DEFAULT_IMAGE = "ubuntu:24.04"
    NON_ROOT_USER = "65534:65534"
    MAX_OUTPUT_BYTES = 16 * 1024
    WALL_CLOCK_TIMEOUT_SEC = 10
    KILL_GRACE_SEC = 2
    HOST_DEADLINE_SLACK_SEC = 5  # host-side backstop if `timeout` itself misbehaves

    def __init__(self, image: str = DEFAULT_IMAGE):
        self.client = docker.from_env()
        self.image = image
        self.container = None

    # ---------------------------------------------------------------- lifecycle
    def _boot_container(self) -> None:
        # PID 1 is docker-init (init=True) and the keep-alive process runs as root, so
        # killing every process of the agent user (kill -9 -1 as 65534) can never stop
        # the container. Previously `pkill -u 65534` also killed `sleep 600` (PID 1),
        # which is what produced the HTTP 409 "container is not running" failures.
        self.container = self.client.containers.run(
            image=self.image,
            command=["sleep", "infinity"],
            detach=True,
            init=True,
            user="0:0",
            read_only=True,
            network_disabled=True,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            mem_limit="256m",
            memswap_limit="256m",
            nano_cpus=500_000_000,
            pids_limit=64,
            tmpfs={
                "/workspace": "rw,noexec,nosuid,nodev,size=64m,uid=65534,gid=65534,mode=1777",
                "/tmp": "rw,noexec,nosuid,nodev,size=32m,uid=65534,gid=65534,mode=1777",
            },
            environment={
                "DEBIAN_FRONTEND": "noninteractive",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": "/workspace",
            },
        )

    def reset_and_prepare_fixture(self, fixture_setup_cmds: List[List[str]]) -> None:
        """Fresh container + fixtures. Raises FixtureError instead of silently continuing."""
        self.cleanup()
        self._boot_container()
        for cmd_tokens in fixture_setup_cmds:
            res = self.container.exec_run(
                cmd=cmd_tokens, user=self.NON_ROOT_USER, workdir="/workspace", demux=True
            )
            if res.exit_code != 0:
                out, err = res.output if res.output else (b"", b"")
                raise FixtureError(
                    f"Fixture command {cmd_tokens} exited {res.exit_code}: "
                    f"{(err or out or b'').decode('utf-8', 'replace')[:300]}"
                )

    def is_running(self) -> bool:
        """No silent restarts: a restarted container has lost its tmpfs fixture state."""
        if self.container is None:
            return False
        try:
            self.container.reload()
            return self.container.status == "running"
        except Exception:
            return False

    def cleanup(self) -> None:
        if self.container is not None:
            try:
                self.container.remove(force=True)
            except Exception:
                pass
            self.container = None

    # ------------------------------------------------------------- gate support
    def resolve_paths(self, paths: List[str]) -> List[str]:
        """Real-path resolution inside the container (for StaticPolicyGate path_resolver)."""
        if not paths:
            return []
        if any("\n" in p for p in paths):
            raise ValueError("Paths containing newlines are not supported.")
        if not self.is_running():
            raise RuntimeError("Sandbox container is not running.")
        res = self.container.exec_run(
            ["realpath", "-m", "--", *paths], user=self.NON_ROOT_USER, demux=True
        )
        out, err = res.output if res.output else (b"", b"")
        if res.exit_code != 0:
            raise RuntimeError((err or b"").decode("utf-8", "replace").strip() or "realpath failed")
        lines = (out or b"").decode("utf-8", "replace").splitlines()
        if len(lines) != len(paths):
            raise RuntimeError("realpath returned an unexpected number of lines.")
        return lines

    # ---------------------------------------------------------------- execution
    def _kill_agent_processes(self) -> None:
        # Runs as the agent user, so it can only signal the agent's own processes
        # (no CAP_KILL needed, PID 1 and the root keep-alive are untouched).
        try:
            self.container.exec_run(["sh", "-c", "kill -9 -1"], user=self.NON_ROOT_USER)
        except Exception:
            pass

    def execute_action(self, action: StructuredAction) -> SandboxResult:
        if not self.is_running():
            return SandboxResult(
                exit_code=125, stdout="", stderr="Sandbox container is not running.",
                timed_out=False, wall_time_ms=0.0,
            )

        script = action.render_command_line()
        argv = [
            "timeout", "-k", str(self.KILL_GRACE_SEC), str(self.WALL_CLOCK_TIMEOUT_SEC),
            "sh", "-c", script,
        ]
        stdout_chunks: List[bytes] = []
        stderr_chunks: List[bytes] = []
        total_bytes = 0
        truncated = False
        host_deadline_hit = False
        host_deadline = self.WALL_CLOCK_TIMEOUT_SEC + self.KILL_GRACE_SEC + self.HOST_DEADLINE_SLACK_SEC
        start_ts = time.monotonic()

        try:
            exec_id = self.client.api.exec_create(
                container=self.container.id, cmd=argv, stdout=True, stderr=True,
                user=self.NON_ROOT_USER, workdir=action.working_dir_id,
            )["Id"]
            stream = self.client.api.exec_start(exec_id, detach=False, stream=True, demux=True)
        except APIError as exc:
            return SandboxResult(
                exit_code=126, stdout="", stderr=f"Docker APIError: {exc}",
                timed_out=False, wall_time_ms=0.0,
            )

        try:
            for out_chunk, err_chunk in stream:
                for chunk, sink in ((out_chunk, stdout_chunks), (err_chunk, stderr_chunks)):
                    if chunk:
                        remaining = max(0, self.MAX_OUTPUT_BYTES - total_bytes)
                        sink.append(chunk[:remaining])
                        total_bytes += len(chunk)
                if total_bytes >= self.MAX_OUTPUT_BYTES:
                    truncated = True
                    break
                if time.monotonic() - start_ts > host_deadline:
                    host_deadline_hit = True
                    break
        except APIError as exc:
            stderr_chunks.append(f"Docker APIError: {exc}".encode("utf-8"))

        if truncated or host_deadline_hit:
            self._kill_agent_processes()

        exit_code: Optional[int] = None
        poll_until = time.monotonic() + 3.0
        while time.monotonic() < poll_until:
            try:
                info = self.client.api.exec_inspect(exec_id)
            except Exception:
                break
            if not info.get("Running") and info.get("ExitCode") is not None:
                exit_code = int(info["ExitCode"])
                break
            time.sleep(0.05)

        elapsed_s = time.monotonic() - start_ts
        if exit_code is None:
            self._kill_agent_processes()
            exit_code = 124
            host_deadline_hit = True

        # 124: `timeout` fired; 137 after the full limit: -k escalation to SIGKILL.
        timed_out = (not truncated) and (
            host_deadline_hit
            or exit_code == 124
            or (exit_code == 137 and elapsed_s >= self.WALL_CLOCK_TIMEOUT_SEC)
        )
        if timed_out:
            self._kill_agent_processes()  # reap orphans that escaped the process group

        try:
            self.container.reload()
            oom_killed = bool(self.container.attrs.get("State", {}).get("OOMKilled", False))
        except Exception:
            oom_killed = False

        return SandboxResult(
            exit_code=exit_code,
            stdout=b"".join(stdout_chunks).decode("utf-8", errors="replace"),
            stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
            timed_out=timed_out,
            oom_killed=oom_killed,
            output_truncated=truncated,
            wall_time_ms=round(elapsed_s * 1000.0, 2),
        )

    def run_assertion_cmd(self, argv: List[str], workdir: str = "/workspace") -> SandboxResult:
        """Validator probe. Runs as the agent user (same view of files as the agent)."""
        if not self.is_running():
            return SandboxResult(
                exit_code=125, stdout="", stderr="Sandbox container is not running.",
                timed_out=False, wall_time_ms=0.0,
            )
        try:
            res = self.container.exec_run(
                cmd=argv, workdir=workdir, user=self.NON_ROOT_USER, demux=True
            )
            out_b, err_b = res.output if res.output else (b"", b"")
            return SandboxResult(
                exit_code=res.exit_code,
                stdout=(out_b or b"").decode("utf-8", errors="replace"),
                stderr=(err_b or b"").decode("utf-8", errors="replace"),
                timed_out=False,
                wall_time_ms=0.0,
            )
        except APIError as exc:
            return SandboxResult(
                exit_code=1, stdout="", stderr=f"Error: {exc}", timed_out=False, wall_time_ms=0.0,
            )
