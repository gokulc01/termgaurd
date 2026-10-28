from __future__ import annotations
import time
from typing import List, Optional
import docker
from docker.errors import APIError, ContainerError

from schemas import SandboxResult, StructuredAction


class SandboxRunner:
    """
    Executes validated StructuredAction objects inside an isolated Ubuntu container.
    Enforces non-root execution, read-only rootfs, cap_drop=['ALL'], no-new-privileges,
    network_disabled=True, PID limits, CPU/Memory limits, and output truncation.
    """

    DEFAULT_IMAGE = "ubuntu:24.04"
    NON_ROOT_USER = "65534:65534"  # nobody:nogroup
    MAX_OUTPUT_BYTES = 16 * 1024  # 16 KB cap on stdout/stderr
    WALL_CLOCK_TIMEOUT_SEC = 10.0  # 10s hard execution timeout

    def __init__(self, image: str = DEFAULT_IMAGE):
        self.client = docker.from_env()
        self.image = image
        self.container = None

    def reset_and_prepare_fixture(self, fixture_setup_cmds: List[List[str]]) -> None:
        """Destroys any prior container, boots a fresh hardened container, and seeds task fixtures."""
        self.cleanup()

        self.container = self.client.containers.run(
            image=self.image,
            command=["sleep", "300"],
            detach=True,
            user=self.NON_ROOT_USER,
            read_only=True,
            network_disabled=True,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            mem_limit="256m",
            memswap_limit="256m",
            nano_cpus=500_000_000,  # 0.5 CPU core
            pids_limit=64,  # Defends against fork bombs
            tmpfs={
                "/workspace": "rw,noexec,nosuid,nodev,size=64m,uid=65534,gid=65534,mode=1777",
                "/tmp": "rw,noexec,nosuid,nodev,size=32m,uid=65534,gid=65534,mode=1777",
            },
            environment={
                "DEBIAN_FRONTEND": "noninteractive",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/local/bin:/usr/bin:/bin",
            },
        )

        # Populate initial task fixtures inside /workspace
        for cmd_tokens in fixture_setup_cmds:
            self.container.exec_run(
                cmd=cmd_tokens,
                user=self.NON_ROOT_USER,
                workdir="/workspace",
            )

    def execute_action(self, action: StructuredAction) -> SandboxResult:
        """
        Executes [action.utility, *action.args] directly via execve in the container
        without invoking an intermediate shell interpreter.
        """
        if self.container is None:
            raise RuntimeError(
                "Sandbox container is not initialized. Call reset_and_prepare_fixture first."
            )

        argv = [action.utility] + [str(a) for a in action.args]
        start_ts = time.monotonic()

        # Run command inside a background exec instance to enforce strict wall-clock timeout
        exec_id = self.client.api.exec_create(
            container=self.container.id,
            cmd=argv,
            stdout=True,
            stderr=True,
            user=self.NON_ROOT_USER,
            workdir=action.working_dir_id,
        )["Id"]

        output_generator = self.client.api.exec_start(
            exec_id, detach=False, stream=True, demux=True
        )

        stdout_chunks: List[bytes] = []
        stderr_chunks: List[bytes] = []
        total_bytes = 0
        timed_out = False

        try:
            for out_chunk, err_chunk in output_generator:
                elapsed = time.monotonic() - start_ts
                if elapsed > self.WALL_CLOCK_TIMEOUT_SEC:
                    timed_out = True
                    break

                if out_chunk:
                    remaining = max(0, self.MAX_OUTPUT_BYTES - total_bytes)
                    stdout_chunks.append(out_chunk[:remaining])
                    total_bytes += len(out_chunk)
                if err_chunk:
                    remaining = max(0, self.MAX_OUTPUT_BYTES - total_bytes)
                    stderr_chunks.append(err_chunk[:remaining])
                    total_bytes += len(err_chunk)

                if total_bytes >= self.MAX_OUTPUT_BYTES:
                    break
        except APIError as exc:
            stderr_chunks.append(f"Docker APIError: {exc}".encode("utf-8"))

        elapsed_ms = (time.monotonic() - start_ts) * 1000.0
        inspect_info = self.client.api.exec_inspect(exec_id)
        exit_code = inspect_info.get("ExitCode")
        if timed_out or exit_code is None:
            exit_code = 124  # Standard timeout exit code
            # Restart fresh state or kill lingering subprocesses
            self.container.exec_run(["pkill", "-u", "65534"], user="0:0")

        self.container.reload()
        oom_killed = bool(self.container.attrs.get("State", {}).get("OOMKilled", False))

        return SandboxResult(
            exit_code=int(exit_code),
            stdout=b"".join(stdout_chunks).decode("utf-8", errors="replace"),
            stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
            timed_out=timed_out,
            oom_killed=oom_killed,
            wall_time_ms=round(elapsed_ms, 2),
        )

    def run_assertion_cmd(
        self, argv: List[str], workdir: str = "/workspace"
    ) -> SandboxResult:
        """Used by the evaluation harness to run deterministic post-condition assertions."""
        res = self.container.exec_run(cmd=argv, workdir=workdir, demux=True)
        out_b, err_b = res.output if res.output else (b"", b"")
        return SandboxResult(
            exit_code=res.exit_code,
            stdout=(out_b or b"").decode("utf-8", errors="replace"),
            stderr=(err_b or b"").decode("utf-8", errors="replace"),
            timed_out=False,
            wall_time_ms=0.0,
        )

    def cleanup(self) -> None:
        if self.container is not None:
            try:
                self.container.remove(force=True)
            except Exception:
                pass
            self.container = None
