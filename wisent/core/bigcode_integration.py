"""
BigCode Evaluation Harness integration for Wisent.

This module provides integration with bigcode-evaluation-harness for code generation benchmarks.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from wisent.core.errors import TaskNotFoundError, InsufficientDataError

logger = logging.getLogger(__name__)


class BigCodeTaskLoader:
    """Loads and manages BigCode evaluation tasks."""

    # Mapping of our task names to BigCode task names
    TASK_MAPPING = {
        # === DIRECT MATCHES ===
        "humaneval": "humaneval",
        "mbpp": "mbpp",
        "conala": "conala",
        "concode": "concode",
        "mercury": "mercury",
        # === CORRECTED MAPPINGS ===
        "humaneval_plus": "humanevalplus",
        "instructhumaneval": "instruct-humaneval",
        "mbpp_plus": "mbppplus",
        "apps": "apps-introductory",
        "ds1000": "ds1000-all-completion",
        # === MULTI-LANGUAGE TASKS ===
        "multiple_py": "multiple-py",
        "multiple_js": "multiple-js",
        "multiple_java": "multiple-java",
        "multiple_cpp": "multiple-cljcpp",
        "multiple_rs": "multiple-rs",
        "multiple_go": "multiple-go",
        # === CODE-TO-TEXT TASKS ===
        "codexglue_code_to_text_python": "codexglue_code_to_text-python",
        "codexglue_code_to_text_go": "codexglue_code_to_text-go",
        "codexglue_code_to_text_java": "codexglue_code_to_text-java",
        "codexglue_code_to_text_javascript": "codexglue_code_to_text-javascript",
        "codexglue_code_to_text_php": "codexglue_code_to_text-php",
        "codexglue_code_to_text_ruby": "codexglue_code_to_text-ruby",
        # === FIXED PROBLEMATIC MAPPINGS ===
        "recode": "perturbed-humaneval-natgen-num_seeds_1",
        "humanevalpack": None,  # ❌ REMOVED - no simple mapping exists, only complex variants
    }

    def __init__(self):
        """Initialize BigCode task loader."""
        self._bigcode_available = self._check_bigcode_available()
        self._task_cache = {}

    def _check_bigcode_available(self) -> bool:
        """Check if bigcode-evaluation-harness is available."""
        try:
            import bigcode_eval

            return True
        except ImportError:
            logger.warning("bigcode-evaluation-harness not installed")
            return False

    def is_bigcode_task(self, task_name: str) -> bool:
        """Check if a task is a BigCode task."""
        return task_name in self.TASK_MAPPING

    def load_task(self, task_name: str, limit: Optional[int] = None) -> "BigCodeTask":
        """
        Load a BigCode task.

        Args:
            task_name: Name of the task (our naming convention)
            limit: Optional limit on number of samples

        Returns:
            BigCodeTask object
        """
        if not self._bigcode_available:
            raise ImportError("bigcode-evaluation-harness not installed. Run: pip install bigcode-evaluation-harness")

        if task_name not in self.TASK_MAPPING:
            raise TaskNotFoundError(task_name=task_name, available_tasks=list(self.TASK_MAPPING.keys()))

        bigcode_task_name = self.TASK_MAPPING[task_name]

        # Handle removed tasks with None mapping
        if bigcode_task_name is None:
            raise TaskNotFoundError(task_name=task_name)

        # Check cache
        cache_key = f"{task_name}:{limit}"
        if cache_key in self._task_cache:
            return self._task_cache[cache_key]

        # Create task object
        task = BigCodeTask(task_name, bigcode_task_name, limit)
        self._task_cache[cache_key] = task

        return task


class BigCodeTask:
    """Represents a BigCode evaluation task."""

    def __init__(self, task_name: str, bigcode_task_name: str, limit: Optional[int] = None):
        """
        Initialize BigCode task.

        Args:
            task_name: Our task name
            bigcode_task_name: BigCode's task name
            limit: Optional limit on samples
        """
        self.task_name = task_name
        self.bigcode_task_name = bigcode_task_name
        self.limit = limit
        self._limit = limit  # Store as private attribute too
        self._data = None
        self._task_obj = None
        self._load_data()

    def _load_data(self):
        """Load task data from BigCode."""
        try:
            # Import BigCode modules
            from bigcode_eval.tasks import get_task

            # Get the task
            task = get_task(self.bigcode_task_name)
            self._task_obj = task

            # Get dataset - BigCode uses get_dataset() method
            dataset = task.get_dataset()

            # Convert to list if needed
            if hasattr(dataset, "__iter__"):
                dataset = list(dataset)

            # Apply limit if specified
            if self.limit:
                dataset = dataset[: self.limit]

            self._data = dataset

        except Exception as e:
            logger.error(f"Failed to load BigCode task {self.bigcode_task_name}: {e}")
            # Fallback to loading from files if available
            self._load_from_files()

    # Methods to match lm-eval interface
    def has_validation_docs(self) -> bool:
        """Check if task has validation documents."""
        return False  # BigCode tasks don't have separate validation sets

    def has_test_docs(self) -> bool:
        """Check if task has test documents."""
        return True  # All samples are considered test docs

    def test_docs(self) -> List[Dict[str, Any]]:
        """Get test documents."""
        return self.get_samples()

    def validation_docs(self) -> List[Dict[str, Any]]:
        """Get validation documents."""
        return []  # No separate validation set

    def doc_to_text(self, doc: Dict[str, Any]) -> str:
        """Convert document to text prompt."""
        # Handle different BigCode formats
        if "prompt" in doc:
            return doc["prompt"]
        if "text" in doc:
            return doc["text"]
        if "question" in doc:
            return doc["question"]
        if "problem" in doc:
            return doc["problem"]
        # Fallback - try to use task object if available
        if self._task_obj and hasattr(self._task_obj, "get_prompt"):
            return self._task_obj.get_prompt(doc)
        return str(doc)

    def _load_from_files(self):
        """Load task data from local files as fallback."""
        # Try to load from standard locations
        data_paths = [
            f"~/.cache/bigcode_eval/{self.bigcode_task_name}",
            f"data/{self.bigcode_task_name}",
            f"bigcode_eval/tasks/{self.bigcode_task_name}",
        ]

        for path in data_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                self._load_from_path(expanded_path)
                return

        # If no data found, raise error
        raise InsufficientDataError(reason=f"No data found for task {self.task_name}. Please provide valid benchmark data.")

    def _load_from_path(self, path: str):
        """Load data from a specific path."""
        data = []

        # Look for JSON/JSONL files
        for file in Path(path).glob("*.json*"):
            with open(file) as f:
                if file.suffix == ".jsonl":
                    for line in f:
                        data.append(json.loads(line))
                else:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        data.extend(file_data)
                    else:
                        data.append(file_data)

        if self.limit:
            data = data[: self.limit]

        self._data = data

    def get_samples(self) -> List[Dict[str, Any]]:
        """Get all samples from the task."""
        return self._data if self._data else []

    def __len__(self):
        """Get number of samples."""
        return len(self._data) if self._data else 0

    def __iter__(self):
        """Iterate over samples."""
        return iter(self.get_samples())


class BigCodeEvaluator:
    """Evaluates model outputs on BigCode benchmarks."""

    def __init__(self, docker_executor=None):
        """
        Initialize evaluator.

        Args:
            docker_executor: Optional Docker executor for secure code execution
        """
        self.docker_executor = docker_executor

    def evaluate(self, task: BigCodeTask, generations: List[str], k_values: List[int] = [1, 10, 100]) -> Dict[str, Any]:
        """
        Evaluate generations on a BigCode task.

        Args:
            task: BigCodeTask object
            generations: List of generated code solutions
            k_values: k values for pass@k metric

        Returns:
            Evaluation results dict
        """
        results = {
            "task": task.task_name,
            "num_samples": len(task),
            "num_generations": len(generations),
            "pass_at_k": {},
        }

        # For code generation tasks, we need to execute and test
        if self._is_code_execution_task(task.task_name):
            results["execution_results"] = self._evaluate_code_execution(task, generations)

            # Calculate pass@k
            for k in k_values:
                if k <= len(generations):
                    pass_rate = self._calculate_pass_at_k(results["execution_results"], k)
                    results["pass_at_k"][f"pass@{k}"] = pass_rate

        else:
            # For non-execution tasks (e.g., code-to-text), use BLEU or other metrics
            results["bleu_scores"] = self._evaluate_text_generation(task, generations)

        return results

    def _is_code_execution_task(self, task_name: str) -> bool:
        """Check if task requires code execution."""
        non_execution_tasks = {
            "codexglue_code_to_text",
            "codexglue_code_to_text_python",
            "codexglue_code_to_text_go",
            "codexglue_code_to_text_ruby",
            "codexglue_code_to_text_java",
            "codexglue_code_to_text_javascript",
            "codexglue_code_to_text_php",
        }
        return task_name not in non_execution_tasks

    def _evaluate_code_execution(self, task: BigCodeTask, generations: List[str]) -> List[Dict]:
        """Evaluate code by executing it."""
        results = []

        for i, sample in enumerate(task.get_samples()):
            sample_results = []

            for j, generation in enumerate(generations[i] if i < len(generations) else []):
                result = self._execute_and_test(sample, generation, task.task_name)
                sample_results.append(result)

            results.append({"sample_id": i, "results": sample_results})

        return results

    def _execute_and_test(self, sample: Dict, generation: str, task_name: str) -> Dict:
        """Execute generated code and run tests."""
        if self.docker_executor:
            # Use Docker for secure execution
            return self._execute_in_docker(sample, generation, task_name)
        # Fallback to subprocess (less secure)
        return self._execute_in_subprocess(sample, generation, task_name)

    def _execute_in_docker(self, sample: Dict, generation: str, task_name: str) -> Dict:
        """
        Execute code in Docker container for secure sandboxed execution.
        
        This provides isolation from the host system and prevents malicious
        code from causing damage. Uses resource limits and network isolation.
        
        Args:
            sample: Task sample with test cases
            generation: Generated code to execute
            task_name: Name of the task for language detection
            
        Returns:
            Dict with 'passed', 'error', 'output' keys
        """
        result = {"passed": False, "error": None, "output": None}
        
        try:
            import docker
            from docker.errors import ContainerError, ImageNotFound, APIError
        except ImportError:
            result["error"] = "Docker SDK not installed. Run: pip install docker"
            return result
        
        try:
            client = docker.from_env()
        except Exception as e:
            result["error"] = f"Failed to connect to Docker daemon: {e}"
            return result
        
        # Determine language and Docker image
        language = self._detect_language(task_name)
        image_map = {
            "python": "python:3.10-slim",
            "javascript": "node:18-slim",
            "java": "openjdk:17-slim",
            "cpp": "gcc:12",
            "go": "golang:1.21-alpine",
            "rust": "rust:1.70-slim",
        }
        image = image_map.get(language, "python:3.10-slim")
        
        # Create test script
        test_script = self._create_test_script(sample, generation, task_name)
        
        # Create a temporary directory for the code
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write the test script
            script_path = os.path.join(tmpdir, self._get_script_filename(language))
            with open(script_path, "w") as f:
                f.write(test_script)
            
            # Build the command based on language
            cmd = self._get_docker_command(language, script_path)
            
            try:
                # Pull image if not available
                try:
                    client.images.get(image)
                except ImageNotFound:
                    logger.info(f"Pulling Docker image: {image}")
                    client.images.pull(image)
                
                # Run container with security constraints
                container = client.containers.run(
                    image=image,
                    command=cmd,
                    volumes={tmpdir: {"bind": "/code", "mode": "ro"}},
                    working_dir="/code",
                    mem_limit="256m",
                    cpu_period=100000,
                    cpu_quota=50000,  # 50% CPU limit
                    network_disabled=True,  # No network access
                    read_only=True,  # Read-only filesystem
                    user="nobody",  # Run as unprivileged user
                    detach=True,
                    stderr=True,
                    stdout=True,
                )
                
                # Wait for completion with timeout
                exit_status = container.wait(timeout=30)
                
                # Get output
                stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
                stderr = container.logs(stdout=False, stderr=True).decode("utf-8")
                
                # Clean up container
                container.remove(force=True)
                
                if exit_status["StatusCode"] == 0:
                    result["passed"] = True
                    result["output"] = stdout
                    logger.debug(f"Docker execution PASSED. Output: {stdout[:200]}")
                else:
                    result["error"] = stderr or stdout
                    logger.debug(f"Docker execution FAILED. Error: {result['error'][:500]}")
                    
            except ContainerError as e:
                result["error"] = f"Container error: {e.stderr.decode() if e.stderr else str(e)}"
            except APIError as e:
                result["error"] = f"Docker API error: {e}"
            except Exception as e:
                if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                    result["error"] = "Timeout (30s)"
                else:
                    result["error"] = str(e)
        
        return result
    
    def _detect_language(self, task_name: str) -> str:
        """Detect programming language from task name."""
        task_lower = task_name.lower()
        
        if "py" in task_lower or "python" in task_lower:
            return "python"
        elif "js" in task_lower or "javascript" in task_lower:
            return "javascript"
        elif "java" in task_lower and "javascript" not in task_lower:
            return "java"
        elif "cpp" in task_lower or "c++" in task_lower:
            return "cpp"
        elif "go" in task_lower and "golang" not in task_lower:
            return "go"
        elif "rs" in task_lower or "rust" in task_lower:
            return "rust"
        
        return "python"  # Default
    
    def _get_script_filename(self, language: str) -> str:
        """Get appropriate filename for the language."""
        extensions = {
            "python": "test_script.py",
            "javascript": "test_script.js",
            "java": "TestScript.java",
            "cpp": "test_script.cpp",
            "go": "test_script.go",
            "rust": "test_script.rs",
        }
        return extensions.get(language, "test_script.py")
    
    def _get_docker_command(self, language: str, script_path: str) -> str:
        """Get the command to run the script in Docker."""
        filename = os.path.basename(script_path)
        commands = {
            "python": f"python /code/{filename}",
            "javascript": f"node /code/{filename}",
            "java": f"cd /code && javac {filename} && java TestScript",
            "cpp": f"cd /code && g++ -o /tmp/test {filename} && /tmp/test",
            "go": f"cd /code && go run {filename}",
            "rust": f"cd /code && rustc -o /tmp/test {filename} && /tmp/test",
        }
        return commands.get(language, f"python /code/{filename}")

    def _execute_in_subprocess(self, sample: Dict, generation: str, task_name: str) -> Dict:
        """Execute code in subprocess (less secure)."""
        result = {"passed": False, "error": None, "output": None}

        try:
            # Create test script
            test_script = self._create_test_script(sample, generation, task_name)

            # Write to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(test_script)
                temp_path = f.name

            try:
                # Execute
                proc = subprocess.run([sys.executable, temp_path], capture_output=True, text=True, timeout=10)

                if proc.returncode == 0:
                    result["passed"] = True
                    result["output"] = proc.stdout
                    logger.debug(f"✅ Code execution PASSED. Output: {proc.stdout[:200]}")
                else:
                    result["error"] = proc.stderr or proc.stdout
                    logger.debug(f"❌ Code execution FAILED. Error: {result['error'][:500]}")

            finally:
                # Clean up
                os.unlink(temp_path)

        except subprocess.TimeoutExpired:
            result["error"] = "Timeout"
        except Exception as e:
            result["error"] = str(e)

        return result

    def _create_test_script(self, sample: Dict, generation: str, task_name: str) -> str:
        """Create a test script for the sample."""
        if "humaneval" in task_name:
            script = self._create_humaneval_test_script(sample, generation)
        elif "mbpp" in task_name:
            script = self._create_mbpp_test_script(sample, generation)
        elif "apps" in task_name:
            script = self._create_apps_test_script(sample, generation)
        else:
            # Default format
            script = self._create_humaneval_test_script(sample, generation)

        logger.debug(f"📝 Test script for {task_name}:\n{script}\n")
        return script

    def _create_humaneval_test_script(self, sample: Dict, generation: str) -> str:
        """Create test script for HumanEval format."""
        entry_point = sample.get("entry_point", "solution")
        test_code = sample.get("test", "")
        prompt = sample.get("prompt", "")

        # The prompt contains the function signature, and generation should be the function body
        # We need to combine them properly
        script = f"""
{prompt}{generation}

{test_code}

if __name__ == "__main__":
    check({entry_point})
    print("All tests passed!")
"""
        return script

    def _create_mbpp_test_script(self, sample: Dict, generation: str) -> str:
        """Create test script for MBPP format."""
        test_imports = sample.get("test_imports", [])
        test_list = sample.get("test_list", [])

        # Fix function name mismatch before creating test script
        fixed_generation = self._fix_function_name_mismatch(generation, test_list)

        imports = "\n".join(test_imports)
        tests = "\n    ".join(test_list)

        script = f"""
{imports}

{fixed_generation}

if __name__ == "__main__":
    {tests}
    print("All tests passed!")
"""
        return script

    def _create_apps_test_script(self, sample: Dict, generation: str) -> str:
        """Create test script for APPS format."""
        # APPS has input/output pairs
        io_data = json.loads(sample.get("input_output", "{}"))
        inputs = io_data.get("inputs", [])
        outputs = io_data.get("outputs", [])

        tests = []
        for inp, out in zip(inputs, outputs):
            tests.append(f"assert str(solution({inp})) == '{out}'")

        test_code = "\n    ".join(tests)

        script = f"""
{generation}

if __name__ == "__main__":
    {test_code}
    print("All tests passed!")
"""
        return script

    def _fix_function_name_mismatch(self, code: str, test_list: List[str]) -> str:
        """
        Fix function name mismatches between generated code and test cases.

        Uses wrapper function approach for robustness across different code structures.

        Args:
            code: Generated code that may have wrong function name
            test_list: List of test assertions that specify expected function name

        Returns:
            Fixed code with wrapper function if needed
        """
        import re

        if not test_list or not code.strip():
            return code

        # Extract expected function name from test assertions
        expected_name = None
        # Built-in functions to skip when looking for the target function
        builtin_functions = {
            "set",
            "len",
            "str",
            "int",
            "float",
            "list",
            "tuple",
            "dict",
            "sum",
            "max",
            "min",
            "abs",
            "round",
            "sorted",
            "reversed",
        }

        for test in test_list:
            # Find all function calls in assert statements
            function_calls = re.findall(r"(\w+)\s*\(", test)

            for func_name in function_calls:
                # Skip built-in functions and common test functions
                if func_name not in builtin_functions and func_name not in {
                    "assert",
                    "assertEqual",
                    "assertTrue",
                    "assertFalse",
                }:
                    expected_name = func_name
                    break

            if expected_name:
                break

        if not expected_name:
            return code  # No function name found in tests

        # Extract actual function name from generated code
        actual_name = None
        func_match = re.search(r"def\s+(\w+)\s*\(", code)
        if func_match:
            actual_name = func_match.group(1)

        if not actual_name:
            return code  # No function definition found

        if actual_name == expected_name:
            return code  # Names already match

        logger.debug(f"🔧 Function name mismatch detected: {actual_name} → {expected_name}")
        logger.debug("   Adding wrapper function for compatibility")

        # Add wrapper function to bridge the name gap
        wrapper = f"""
# Wrapper function for test compatibility
def {expected_name}(*args, **kwargs):
    return {actual_name}(*args, **kwargs)
"""

        return code + wrapper

    def _calculate_pass_at_k(self, execution_results: List[Dict], k: int) -> float:
        """Calculate pass@k metric."""
        total_passed = 0
        total_samples = len(execution_results)

        for result in execution_results:
            sample_results = result["results"][:k]
            if any(r["passed"] for r in sample_results):
                total_passed += 1

        return total_passed / total_samples if total_samples > 0 else 0.0

    def _evaluate_text_generation(self, task: BigCodeTask, generations: List[str]) -> Dict[str, Any]:
        """
        Evaluate text generation tasks (e.g., code-to-text) using BLEU and other metrics.
        
        Args:
            task: BigCodeTask with reference texts
            generations: List of generated texts
            
        Returns:
            Dict with BLEU scores and other metrics
        """
        try:
            import sacrebleu
            has_sacrebleu = True
        except ImportError:
            has_sacrebleu = False
            logger.warning("sacrebleu not installed. Using fallback BLEU implementation.")
        
        results = {
            "bleu_scores": [],
            "avg_bleu": 0.0,
            "exact_match": 0.0,
            "f1_scores": [],
            "avg_f1": 0.0,
        }
        
        samples = task.get_samples()
        total_exact_match = 0
        
        for i, sample in enumerate(samples):
            if i >= len(generations):
                break
            
            generation = generations[i] if isinstance(generations[i], str) else generations[i][0]
            reference = self._extract_reference(sample)
            
            if reference is None:
                continue
            
            # BLEU score
            if has_sacrebleu:
                bleu = self._compute_sacrebleu(generation, reference)
            else:
                bleu = self._compute_simple_bleu(generation, reference)
            results["bleu_scores"].append(bleu)
            
            # Exact match
            if self._normalize_text(generation) == self._normalize_text(reference):
                total_exact_match += 1
            
            # F1 score (token-level)
            f1 = self._compute_token_f1(generation, reference)
            results["f1_scores"].append(f1)
        
        # Compute averages
        if results["bleu_scores"]:
            results["avg_bleu"] = sum(results["bleu_scores"]) / len(results["bleu_scores"])
        if results["f1_scores"]:
            results["avg_f1"] = sum(results["f1_scores"]) / len(results["f1_scores"])
        if samples:
            results["exact_match"] = total_exact_match / min(len(samples), len(generations))
        
        return results
    
    def _extract_reference(self, sample: Dict) -> Optional[str]:
        """Extract reference text from a sample."""
        # Try common field names for reference texts
        for field in ["docstring", "reference", "target", "answer", "comment", "description", "nl"]:
            if field in sample and sample[field]:
                ref = sample[field]
                if isinstance(ref, list):
                    return ref[0] if ref else None
                return str(ref)
        return None
    
    def _compute_sacrebleu(self, hypothesis: str, reference: str) -> float:
        """Compute BLEU score using sacrebleu library."""
        import sacrebleu
        
        # sacrebleu expects references as a list of lists
        bleu = sacrebleu.sentence_bleu(hypothesis, [reference])
        return bleu.score / 100.0  # Normalize to [0, 1]
    
    def _compute_simple_bleu(self, hypothesis: str, reference: str, max_n: int = 4) -> float:
        """
        Compute simple BLEU score without external dependencies.
        
        Uses smoothed BLEU to handle short sequences.
        """
        from collections import Counter
        import math
        
        def get_ngrams(text: str, n: int) -> Counter:
            tokens = text.lower().split()
            return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
        
        hyp_tokens = hypothesis.lower().split()
        ref_tokens = reference.lower().split()
        
        if not hyp_tokens or not ref_tokens:
            return 0.0
        
        # Compute n-gram precisions
        precisions = []
        for n in range(1, min(max_n + 1, len(hyp_tokens) + 1)):
            hyp_ngrams = get_ngrams(hypothesis, n)
            ref_ngrams = get_ngrams(reference, n)
            
            # Clipped counts
            clipped_count = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
            total_count = sum(hyp_ngrams.values())
            
            # Smoothing: add 1 to avoid zero precision
            precision = (clipped_count + 1) / (total_count + 1)
            precisions.append(precision)
        
        if not precisions:
            return 0.0
        
        # Geometric mean of precisions
        log_precision = sum(math.log(p) for p in precisions) / len(precisions)
        
        # Brevity penalty
        if len(hyp_tokens) >= len(ref_tokens):
            bp = 1.0
        else:
            bp = math.exp(1 - len(ref_tokens) / len(hyp_tokens))
        
        bleu = bp * math.exp(log_precision)
        return min(1.0, bleu)  # Cap at 1.0
    
    def _compute_token_f1(self, hypothesis: str, reference: str) -> float:
        """Compute token-level F1 score."""
        hyp_tokens = set(hypothesis.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if not hyp_tokens or not ref_tokens:
            return 0.0
        
        common = hyp_tokens & ref_tokens
        
        if not common:
            return 0.0
        
        precision = len(common) / len(hyp_tokens)
        recall = len(common) / len(ref_tokens)
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for exact match comparison."""
        import re
        # Lowercase, remove extra whitespace, remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text


# Main interface for BigCode integration
_loader = None
_evaluator = None


def get_bigcode_loader() -> BigCodeTaskLoader:
    """Get the global BigCode task loader."""
    global _loader
    if _loader is None:
        _loader = BigCodeTaskLoader()
    return _loader


def get_bigcode_evaluator(docker_executor=None) -> BigCodeEvaluator:
    """Get the global BigCode evaluator."""
    global _evaluator
    if _evaluator is None:
        _evaluator = BigCodeEvaluator(docker_executor)
    return _evaluator


def is_bigcode_task(task_name: str) -> bool:
    """Check if a task is from BigCode."""
    return get_bigcode_loader().is_bigcode_task(task_name)


def load_bigcode_task(task_name: str, limit: Optional[int] = None) -> BigCodeTask:
    """Load a BigCode task."""
    return get_bigcode_loader().load_task(task_name, limit)


def evaluate_bigcode_task(task: BigCodeTask, generations: List[str], docker_executor=None) -> Dict[str, Any]:
    """Evaluate generations on a BigCode task."""
    evaluator = get_bigcode_evaluator(docker_executor)
    return evaluator.evaluate(task, generations)
