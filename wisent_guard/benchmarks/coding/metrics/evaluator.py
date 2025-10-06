from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, TYPE_CHECKING

from wisent_guard.benchmarks.coding.safe_docker.core.runtime import DockerSandboxExecutor
from wisent_guard.benchmarks.coding.safe_docker.recipes import RECIPE_REGISTRY
from wisent_guard.benchmarks.coding.metrics.core.atoms import SampleOutcome, Evaluator

from wisent_guard.benchmarks.coding.output_sanitizer.core.atoms import TaskSchema
from wisent_guard.benchmarks.coding.output_sanitizer.python_sanitizer import PythonStandardizer
from wisent_guard.benchmarks.coding.output_sanitizer.cpp_sanitizer import CppStandardizer
from wisent_guard.benchmarks.coding.output_sanitizer.java_sanitizer import JavaStandardizer

if TYPE_CHECKING:
    from wisent_guard.benchmarks.coding.safe_docker.core.atoms import Result
    from wisent_guard.benchmarks.coding.providers.core.atoms import Provider, CodingTask
    from wisent_guard.benchmarks.coding.output_sanitizer.core.atoms import CodeStandardizer

RepairFn = Callable[[str, dict[str,str], str], dict[str,str]]  

@dataclass
class EvaluatorConfig:
    """
    Configuration for CodingEvaluator.
    
    attributes:
        image:
            Docker image to use for code execution (default: "coding/sandbox:polyglot-1.0").
        runtime:
            Optional Docker runtime (e.g., "runsc" for gVisor).
        feedback_max_chars:
            Maximum characters of feedback to pass to the repair function (default: 2000).
        self_repair:
            Whether to perform a single self-repair turn (default: True). It means the we provide feedback to the model for one iteration.
        time_limit_s:
            Time limit in seconds for each code execution (default: 8s).
        cpu_limit_s:
            CPU time limit in seconds for each code execution (default: 3s).
        mem_limit_mb:
            Memory limit in megabytes for each code execution (default: 768MB).
        pre_sanitize:
            Whether to run LLM output through a sanitizer before execution (default: True).
    """
    image: str = "coding/sandbox:polyglot-1.0"
    runtime: Optional[str] = None        
    feedback_max_chars: int = 2000
    self_repair: bool = True             
    time_limit_s: int = 8
    cpu_limit_s: int = 3
    mem_limit_mb: int = 768
    pre_sanitize: bool = True            

_SANITIZERS = {
    "python": PythonStandardizer(),
    "cpp":    CppStandardizer(),
    "java":   JavaStandardizer(),
}

def _default_filename(lang: str) -> str:
    """
    Returns the default source file name for a given programming language.
    
    arguments:
        lang:
            Programming language ("python", "cpp", or "java").
    
    returns:
        Default filename as a string.
    """
    return {"python":"solution.py","cpp":"solution.cpp","java":"Solution.java"}[lang]

def _make_schema(task: CodingTask) -> TaskSchema:
    """
    Constructs a TaskSchema from a CodingTask, using task options or defaults.
    
    arguments:
        task:
            CodingTask containing language and options.
    
    returns:
        TaskSchema with language, file_name, entry_point, java_class, prefer_rename,
        and allow_wrapper set appropriately.
        
    example:
        >>> from wisent_guard.benchmarks.coding.providers.core.atoms import CodingTask
        >>> task = CodingTask(language="python", files={}, options={"entry_point":"add","file_name":"my_solution.py"})
        >>> schema = _make_schema(task)
        >>> schema.language
        'python'
        >>> schema.file_name
        'my_solution.py'
        >>> schema.entry_point
        'add'
        >>> schema.java_class
        'Solution'
        >>> schema.prefer_rename
        True
        >>> schema.allow_wrapper
        True
    """
    entry = str(task.options.get("entry_point", "solve"))
    file_name = str(task.options.get("file_name", _default_filename(task.language)))
    java_class = str(task.options.get("java_class", "Solution"))
    return TaskSchema(language=task.language, file_name=file_name, entry_point=entry,
                      java_class=java_class, prefer_rename=True, allow_wrapper=True)

class CodingEvaluator(Evaluator):
    """
    Evaluator for coding tasks with optional self-repair.
    """
    def __init__(self, provider: Provider, model_fn: Callable[[CodingTask], dict[str,str]],
                 repair_fn: Optional[RepairFn] = None, cfg: EvaluatorConfig = EvaluatorConfig()):
        self.provider = provider
        self.model_fn = model_fn
        self.repair_fn = repair_fn
        self.cfg = cfg
        self.exec = DockerSandboxExecutor(image=cfg.image, runtime=cfg.runtime)

    def _feedback(self, res: "Result") -> str:
        """
        Generates feedback text from a Result object for use in self-repair.
        
        arguments:
            res:
                Result object containing status, stdout, stderr, and elapsed time.
                
        returns:
            Feedback string summarizing the result, truncated to cfg.feedback_max_chars.
            
        examples:
            >>> from wisent_guard.benchmarks.coding.safe_docker.core.atoms import Result
            >>> res = Result(status="timeout", stdout="", stderr="", elapsed=10.0)
            >>> evaluator = CodingEvaluator(provider=None, model_fn=lambda x: {}, cfg=EvaluatorConfig())
            >>> evaluator._feedback(res)
            'Timeout after 10.00s.'
            >>> res = Result(status="compile_error", stdout="", stderr="error: something went wrong", elapsed=1.5)
            >>> evaluator._feedback(res)
            'Compilation failed:\nerror: something went wrong'
            >>> res = Result(status="runtime_error", stdout="test failed", stderr="", elapsed=0.5)
            >>> evaluator._feedback(res)
            'Runtime error:\ntest failed'
        """
        if res.status == "timeout":
            return f"Timeout after {res.elapsed:.2f}s."
        body = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
        if res.status == "compile_error":
            prefix = "Compilation failed:\n"
        else:
            prefix = "Tests failed:\n"
        return (prefix + body)[: self.cfg.feedback_max_chars]

    def _run_once(self, task: CodingTask, files: dict[str,str]) -> Result:
        """
        Runs a single evaluation job for the given task and files.

        arguments:
            task:
                The coding task to evaluate.
            files:
                The files to include in the evaluation.

        returns:
            Result object containing the status, stdout, stderr, and elapsed time.

        examples:
            >>> from wisent_guard.benchmarks.coding.providers.core.atoms import CodingTask
            >>> from wisent_guard.benchmarks.coding.safe_docker.core.atoms import Result
            >>> task = CodingTask(language="python", files={}, options={})
            >>> files = {"solution.py": "def add(a,b): return a + b", "tests.py": "from solution import add\ndef test_ok(): assert add(1,2)==3"}
            >>> evaluator = CodingEvaluator(provider=None, model_fn=lambda x: {})
            >>> res: Result = evaluator._run_once(task, files)
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
        recipe = RECIPE_REGISTRY[task.language]
        job = recipe.make_job(files, **task.options,
                              time_limit_s=self.cfg.time_limit_s,
                              cpu_limit_s=self.cfg.cpu_limit_s,
                              mem_limit_mb=self.cfg.mem_limit_mb)
        return self.exec.run(files, job)

    def _maybe_sanitize(self, task: CodingTask, files: dict[str,str]) -> dict[str,str]:
        """
        Optionally sanitizes the generated files based on the task schema.

        arguments:
            task:
                The coding task containing language and options.
            files:
                The generated files to potentially sanitize.

        returns:
            The sanitized files if pre_sanitize is True and a sanitizer exists for the language; otherwise, the original files.
        
        examples:
            >>> from wisent_guard.benchmarks.coding.providers.core.atoms import CodingTask
            >>> task = CodingTask(language="python", files={}, options={"entry_point":"add","file_name":"my_solution.py"})
            >>> files = {"my_solution.py": "def add(a,b): return a - b  # BUG"}
            >>> evaluator = CodingEvaluator(provider=None, model_fn=lambda x: {}, cfg=EvaluatorConfig(pre_sanitize=True))
            >>> sanitized_files = evaluator._maybe_sanitize(task, files)
            >>> "my_solution.py" in sanitized_files
            True
            >>> sanitized_files["my_solution.py"]
            'def add(a, b):\n    return a + b\n'
        """
        if not self.cfg.pre_sanitize:
            return files
        schema = _make_schema(task)
        sanitizer: "CodeStandardizer" = _SANITIZERS.get(task.language)
        if sanitizer is None:
            return files

        raw = files.get(schema.file_name) or files.get("__raw__")
        if not raw:
            return files

        out = sanitizer.normalize(raw, schema)
        files = {**files, schema.file_name: out.files.get(schema.file_name, raw)}
        return files

    def evaluate(self) -> Iterable[SampleOutcome]:
        """
        Evaluates all tasks from the provider, performing optional self-repair.

        yields:
            SampleOutcome for each task, indicating pass/fail status and elapsed time.

        examples:
            >>> from wisent_guard.benchmarks.coding.providers.core.atoms import CodingTask, Provider
            >>> class DummyProvider:
            ...     name = "dummy"
            ...     def iter_tasks(self):
            ...         yield CodingTask(language="python", files={"tests.py":"from solution import add\ndef test_ok(): assert add(1,2)==3"},
            ...              options={"entry_point":"add","file_name":"solution.py"})
            >>> def model_fn(task: CodingTask) -> Dict[str,str]:
            ...     return {"solution.py": "def add(a,b): return a - b  # BUG"}
            >>> def repair_fn(lang: str, prev_files: Dict[str,str], feedback: str) -> Dict[str,str]:
            ...     fixed = prev_files["solution.py"].replace("a - b", "a + b")
            ...     return {"solution.py": fixed}
            >>> evaluator = CodingEvaluator(provider=DummyProvider(), model_fn=model_fn, repair_fn=repair_fn, cfg=EvaluatorConfig(self_repair=True))
            >>> outcomes = list(evaluator.evaluate())
            >>> len(outcomes)
            1
            >>> outcomes[0].passed
            True
        """
        for idx, task in enumerate(self.provider.iter_tasks()):
            files0 = self.model_fn(task)
            files0 = {**task.files, **files0}
            files0 = self._maybe_sanitize(task, files0)

            r0 = self._run_once(task, files0)
            if r0.status == "ok":
                yield SampleOutcome(task_id=f"{self.provider.name}:{idx}", status=r0.status, passed=True, elapsed=r0.elapsed)
                continue

            if not self.cfg.self_repair or self.repair_fn is None:
                yield SampleOutcome(task_id=f"{self.provider.name}:{idx}", status=r0.status, passed=False, elapsed=r0.elapsed); continue

            fb = self._feedback(r0)
            files1 = self.repair_fn(task.language, files0, fb)
            files1 = {**task.files, **files1}
            files1 = self._maybe_sanitize(task, files1)

            r1 = self._run_once(task, files1)
            passed = (r0.status == "ok") or (r1.status == "ok")
            yield SampleOutcome(task_id=f"{self.provider.name}:{idx}", status=r1.status, passed=passed, elapsed=(r0.elapsed + r1.elapsed))