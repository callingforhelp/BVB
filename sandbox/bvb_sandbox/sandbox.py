"""Thin wrapper around the Docker CLI for one sandbox container.

One container == one task. The agent (host-side) drives the container through
``Sandbox.exec``; whatever it does, the only contract is that
``/workspace/output/result.blend`` exists when it finishes.
"""

from __future__ import annotations

import dataclasses
import subprocess
import uuid
from pathlib import Path


@dataclasses.dataclass
class ExecResult:
    returncode: int
    stdout: str
    stderr: str


class Sandbox:
    def __init__(
        self,
        *,
        image: str,
        video_path: Path | None,
        cpus: float = 2.0,
        memory: str = "4g",
        network: bool = False,
        name: str | None = None,
        platform: str | None = "linux/amd64",
    ) -> None:
        self.image = image
        self.video_path = video_path
        self.cpus = cpus
        self.memory = memory
        self.network = network
        self.name = name or f"bvb_{uuid.uuid4().hex[:12]}"
        self.platform = platform
        self.container_id: str | None = None

    def start(self) -> None:
        command = [
            "docker", "run", "-d", "--rm",
            "--name", self.name,
            f"--cpus={self.cpus}",
            f"--memory={self.memory}",
        ]
        if self.platform:
            command += ["--platform", self.platform]
        if not self.network:
            command += ["--network", "none"]
        if self.video_path is not None:
            command += ["-v", f"{self.video_path}:/workspace/video/video.mp4:ro"]
        command += [self.image]
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        self.container_id = result.stdout.strip()

    def exec(self, bash: str, timeout: float = 300.0) -> ExecResult:
        if self.container_id is None:
            raise RuntimeError("Sandbox not started")
        command = ["docker", "exec", self.name, "bash", "-lc", bash]
        try:
            completed = subprocess.run(
                command, text=True, capture_output=True, check=False, timeout=timeout
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(124, exc.stdout or "", f"command timed out after {timeout}s")
        return ExecResult(completed.returncode, completed.stdout or "", completed.stderr or "")

    def copy_frames_in(self, local_dir: Path) -> None:
        if self.container_id is None:
            raise RuntimeError("Sandbox not started")
        for frame in sorted(local_dir.glob("*")):
            subprocess.run(
                ["docker", "cp", str(frame), f"{self.name}:/workspace/frames/{frame.name}"],
                check=True, capture_output=True, text=True,
            )

    def copy_out(self, container_path: str, local_path: Path) -> bool:
        if self.container_id is None:
            raise RuntimeError("Sandbox not started")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["docker", "cp", f"{self.name}:{container_path}", str(local_path)],
            text=True, capture_output=True, check=False,
        )
        return result.returncode == 0 and local_path.exists()

    def output_exists(self, container_path: str = "/workspace/output/result.blend") -> bool:
        result = self.exec(f"test -f {container_path} && echo EXISTS || echo MISSING")
        return "EXISTS" in result.stdout

    def stop(self) -> None:
        if self.container_id is None:
            return
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, text=True, check=False)
        self.container_id = None

    def __enter__(self) -> "Sandbox":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
