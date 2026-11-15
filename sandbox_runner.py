from __future__ import annotations
import time
from typing import List, Optional
import docker
from docker.errors import APIError, ContainerError

from schemas import SandboxResult, StructuredAction


class SandboxRunner:
    """
    Executes validated StructuredAction objects inside an isolated Ubuntu container.
    """

    DEFAULT_IMAGE = "ubuntu:24.04"
    NON_ROOT_USER = "65534:65534"
    MAX_OUTPUT_BYTES = 16 * 1024
    WALL_CLOCK_TIMEOUT_SEC = 10.0

    def __init__(self, image: str = DEFAULT_IMAGE):
        self.client = docker.from_env()
        self.image = image
        self.container = None

    def _boot_container(self) -> None:
        self.container = self.client.containers.run(
            image=self.image,
            command=["sleep", "600"],
            detach=True,
            user=self.NON_ROOT_USER,
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
            },
        )

    def reset_and_prepare_fixture(self, fixture_setup_cmds: List[List[str]]) -> None:
        self.cleanup()
        self._boot_container()
        for cmd_tokens in fixture_setup_cmds:
            try:
                self.container.exec_run(
                    cmd=cmd_tokens, user=self.NON_ROOT_USER, workdir="/workspace"
                )
            except Exception:
                pass

    def ensure_container_running(self) -> bool:
        if self.container is None:
            return False
        try:
            self.container.reload()
            if self.container.status != "running":
                self.container.start()
                self.container.reload()
            return self.container.status == "running"
        except Exception:
            return False

    def execute_action(self, action: StructuredAction) -> SandboxResult:
        """
        Executes sequence sequentially. Short-circuits on first failure (mimicking &&).
        """
        if not self.ensure_container_running():
            return SandboxResult(
                exit_code=1,
                stdout="",
                stderr="Container is stopped.",
                timed_out=False,
                oom_killed=False,
                wall_time_ms=0.0,
            )

        stdout_chunks = []
        stderr_chunks = []
        total_bytes = 0
        final_exit_code = 0
        timed_out = False
        total_time_ms = 0.0

        for cmd in action.commands:
            argv = [cmd.utility] + [str(a) for a in cmd.args]
            start_ts = time.monotonic()

            try:
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
            except APIError as exc:
                stderr_chunks.append(f"Docker APIError: {exc}".encode("utf-8"))
                final_exit_code = 1
                break

            cmd_timed_out = False
            try:
                for out_chunk, err_chunk in output_generator:
                    if (time.monotonic() - start_ts) > self.WALL_CLOCK_TIMEOUT_SEC:
                        cmd_timed_out = True
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
            total_time_ms += elapsed_ms

            try:
                exit_code = self.client.api.exec_inspect(exec_id).get("ExitCode")
            except Exception:
                exit_code = None

            if cmd_timed_out or exit_code is None:
                final_exit_code = 124
                timed_out = True
                try:
                    self.container.exec_run(["pkill", "-u", "65534"], user="0:0")
                except Exception:
                    pass
                break

            final_exit_code = int(exit_code)
            if final_exit_code != 0:
                break

        try:
            self.container.reload()
            oom_killed = bool(
                self.container.attrs.get("State", {}).get("OOMKilled", False)
            )
        except Exception:
            oom_killed = False

        return SandboxResult(
            exit_code=final_exit_code,
            stdout=b"".join(stdout_chunks).decode("utf-8", errors="replace"),
            stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
            timed_out=timed_out,
            oom_killed=oom_killed,
            wall_time_ms=round(total_time_ms, 2),
        )

    def run_assertion_cmd(
        self, argv: List[str], workdir: str = "/workspace"
    ) -> SandboxResult:
        if not self.ensure_container_running():
            return SandboxResult(
                exit_code=1,
                stdout="",
                stderr="Container not running.",
                timed_out=False,
                wall_time_ms=0.0,
            )
        try:
            res = self.container.exec_run(cmd=argv, workdir=workdir, demux=True)
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
                exit_code=1,
                stdout="",
                stderr=f"Error: {exc}",
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
