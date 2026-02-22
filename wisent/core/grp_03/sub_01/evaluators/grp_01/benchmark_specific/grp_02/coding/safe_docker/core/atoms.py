from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["Job", "Result", "LanguageRecipe", "SandboxExecutor"]

@dataclass(frozen=True)
class Job:
    """How to build + run a submission inside a sandbox.

    attributes:
        language:
            Programming language, e.g. "python", "cpp", "java".
        compile_argv:
            If not None, argv to compile the code (e.g. ["g++", "-o", "program", "solution.cpp"]).
            If None, no compilation step is done.
        run_argv:
            argv to run the code (e.g. ["./program"] or ["python3", "solution.py"]).
        cpu_limit_s:
            CPU time limit in seconds (e.g. 3).
        wall_timeout_s:
            Wall clock timeout in seconds (e.g. 8).
        mem_limit_mb:
            Memory limit in megabytes (e.g. 4096).
        fsize_mb:
            Max file size in megabytes (e.g. 16).
        nproc:
            Max number of processes/threads (e.g. 128).
        nofile:
            Max number of open files (e.g. 512).

    example:
        >>> job = Job(
        >>>     language="python",
        >>>     compile_argv=None,
        >>>     run_argv=["python3", "solution.py"],
        >>>     cpu_limit_s=3,
        >>>     wall_timeout_s=8,
        >>>     mem_limit_mb=4096,
        >>>     fsize_mb=16,
        >>>     nproc=128,
        >>>     nofile=512,
        >>> )
    """
    language: str
    compile_argv: list[str] | None
    run_argv: list[str]
    cpu_limit_s: int = 3
    wall_timeout_s: int = 8
    mem_limit_mb: int = 4096
    fsize_mb: int = 16
    nproc: int = 128
    nofile: int = 512

@dataclass(frozen=True)
class Result:
    """
    Result of running a Job inside a sandbox.

    attributes:
        status:
            One of "ok", "compile_error", "runtime_error", "timeout".
        exit_code:
            Exit code of the program (or compiler), or -1 if killed by timeout or OOM.
        stdout:
            Captured standard output (max 32k chars).
        stderr:
            Captured standard error (max 32k chars).
        elapsed:
            Wall clock time elapsed in seconds (float).

    example:
        >>> res = Result(
        >>>     status="ok",
        >>>     exit_code=0,
        >>>     stdout="Hello, world!",
        >>>     stderr="",
        >>>     elapsed=1.23,
        >>> )
    """
    status: str          
    exit_code: int
    stdout: str
    stderr: str
    elapsed: float

@runtime_checkable
class LanguageRecipe(Protocol):
    """
    Knows how to create a Job for a given language and set of files.

    attributes:
        language:
            The programming language this recipe supports, e.g. "python", "cpp", "java".
    """
    language: str
    def make_job(self, **options) -> Job: ...
    

@runtime_checkable
class SandboxExecutor(Protocol):
    """
    Executes a Job inside a sandbox, given a read-only job dir of files.
    """
    def run(self, files: dict[str, str], job: Job) -> Result: ...
