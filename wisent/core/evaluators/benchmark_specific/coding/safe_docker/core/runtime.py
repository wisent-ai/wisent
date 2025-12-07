from __future__ import annotations
import json, os, subprocess, tempfile
from typing import TYPE_CHECKING
from wisent.core.evaluators.benchmark_specific.coding.safe_docker.core.atoms import Result, SandboxExecutor
from wisent.core.errors import DockerRuntimeError

if TYPE_CHECKING:
    from wisent.core.evaluators.benchmark_specific.coding.safe_docker.core.atoms import Job

__all__ = ["DockerSandboxExecutor"]

DEFAULT_IMAGE = "coding/sandbox:polyglot-1.0"

SAFE_FLAGS = [
    "--rm", "--network=none",
    "--pids-limit=256",
    "--read-only",
    "--cap-drop=ALL",
    "--security-opt=no-new-privileges",
]

TMPFS_FLAGS = [
    "--tmpfs", "/tmp:exec,mode=1777,size=134217728",
    "--tmpfs", "/work:exec,mode=1777,size=268435456",
]


class DockerSandboxExecutor(SandboxExecutor):
    """
    Executes a Job inside a Docker container, given a read-only job dir of files.
    """
    def __init__(self, image: str = DEFAULT_IMAGE, runtime: str | None = None):
        self.image = image
        self.runtime = runtime
        # Skip Docker health check if environment variable is set (useful for slow Docker Desktop on macOS)
        if not os.environ.get('SKIP_DOCKER_HEALTH_CHECK', '').lower() == 'true':
            self._check_docker_available()

    def _check_docker_available(self) -> None:
        """
        Check if Docker daemon is running and accessible.

        Raises:
            RuntimeError: If Docker is not available or not running.
        """
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                raise DockerRuntimeError(reason=f"Docker daemon is not running: {result.stderr}")
        except FileNotFoundError:
            raise DockerRuntimeError(reason="Docker command not found. Please install Docker.")
        except subprocess.TimeoutExpired:
            raise DockerRuntimeError(reason="Docker command timed out. Docker daemon may be unresponsive.")

    def run(self, files: dict[str, str], job: Job) -> Result:
        """
        Runs a Job inside a Docker container, given a read-only job dir of files.

        arguments:
            files:
                A mapping of filename to file content, representing the job directory.
            job:
                The Job to execute.

        exceptions:
            Raises subprocess.CalledProcessError if the `docker` command itself fails.

        returns:
            A Result object with the outcome of the execution.

        example (pythonm add function)
        >>> from wisent.core.evaluators.benchmark_specific.coding.safe_docker.core.atoms import Job, Result
        >>> from wisent.core.evaluators.benchmark_specific.coding.safe_docker.core.runtime import DockerSandboxExecutor
        >>> job = Job(
        ...     language="python",
        ...     compile_argv=None,
        ...     run_argv=["python3", "/job/tests.py"],
        ...     cpu_limit_s=2,
        ...     wall_timeout_s=5,
        ...     mem_limit_mb=256,
        ... )
        >>> files = {
        ...     "solution.py": "def add(a,b): return a + b",
        ...     "tests.py": "from solution import add\ndef test_ok(): assert add(1,2)==3",
        ... }
        >>> res: Result = DockerSandboxExecutor().run(files, job)
        >>> res.status
        'ok'
        >>> res.exit_code
        0
        >>> res.stdout
        'test_ok passed'
        >>> res.stderr
        ''
        >>> round(res.elapsed, 2)
        0.23
        """
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = os.path.join(tmp, "job")
            os.makedirs(job_dir, exist_ok=True)
            for name, content in files.items():
                with open(os.path.join(job_dir, name), "w", encoding="utf-8") as f:
                    f.write(content)
            with open(os.path.join(job_dir, "job.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "language": job.language,
                    "compile": {"argv": job.compile_argv} if job.compile_argv else None,
                    "run": {"argv": job.run_argv},
                    "cpu_limit_s": job.cpu_limit_s,
                    "wall_timeout_s": job.wall_timeout_s,
                    "mem_limit_mb": job.mem_limit_mb,
                    "fsize_mb": job.fsize_mb,
                    "nproc": job.nproc,
                    "nofile": job.nofile,
                }, f)
            base = ["docker"]
            if self.runtime:
                base += ["--runtime", self.runtime]
            cmd = base + ["run", "-i", *SAFE_FLAGS, *TMPFS_FLAGS, "-v", f"{job_dir}:/job:ro", self.image]
            p = subprocess.run(cmd, check=False, capture_output=True, text=True)
            out = (p.stdout or "").strip()
            try:
                payload = json.loads(out)
            except json.JSONDecodeError:
                return Result(
                    status="runtime_error",
                    exit_code=p.returncode,
                    stdout=p.stdout or "",
                    stderr=p.stderr or "Failed to parse executor output as JSON.",
                    elapsed=0.0,
                )
            return Result(
                status=payload.get("status","runtime_error"),
                exit_code=int(payload.get("exit_code", p.returncode)),
                stdout=payload.get("stdout",""),
                stderr=payload.get("stderr",""),
                elapsed=float(payload.get("elapsed",0.0)),
            )