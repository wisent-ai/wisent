"""Docker execution mixin for BigCode evaluator."""
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from wisent.core.constants import (
    BIGCODE_TEST_TIMEOUT,
    DISPLAY_TRUNCATION_LARGE,
    DISPLAY_TRUNCATION_MEDIUM,
    DOCKER_CPU_PERIOD_US,
    DOCKER_CPU_QUOTA_50PCT_US,
    DEFAULT_TIMEOUT_DOCKER,
)
from wisent.core.utils.core.hardware import docker_bigcode_mem_limit_mb

logger = logging.getLogger(__name__)


class BigCodeDockerMixin:
    """Mixin providing Docker/subprocess execution for BigCodeEvaluator."""

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
                    mem_limit=f"{docker_bigcode_mem_limit_mb()}m",
                    cpu_period=DOCKER_CPU_PERIOD_US,
                    cpu_quota=DOCKER_CPU_QUOTA_50PCT_US,  # 50% CPU limit
                    network_disabled=True,  # No network access
                    read_only=True,  # Read-only filesystem
                    user="nobody",  # Run as unprivileged user
                    detach=True,
                    stderr=True,
                    stdout=True,
                )
                
                # Wait for completion with timeout
                exit_status = container.wait(timeout=DEFAULT_TIMEOUT_DOCKER)
                
                # Get output
                stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
                stderr = container.logs(stdout=False, stderr=True).decode("utf-8")
                
                # Clean up container
                container.remove(force=True)
                
                if exit_status["StatusCode"] == 0:
                    result["passed"] = True
                    result["output"] = stdout
                    logger.debug(f"Docker execution PASSED. Output: {stdout[:DISPLAY_TRUNCATION_MEDIUM]}")
                else:
                    result["error"] = stderr or stdout
                    logger.debug(f"Docker execution FAILED. Error: {result['error'][:DISPLAY_TRUNCATION_LARGE]}")
                    
            except ContainerError as e:
                result["error"] = f"Container error: {e.stderr.decode() if e.stderr else str(e)}"
            except APIError as e:
                result["error"] = f"Docker API error: {e}"
            except Exception as e:
                if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                    result["error"] = f"Timeout ({DEFAULT_TIMEOUT_DOCKER}s)"
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
                proc = subprocess.run([sys.executable, temp_path], capture_output=True, text=True, timeout=BIGCODE_TEST_TIMEOUT)

                if proc.returncode == 0:
                    result["passed"] = True
                    result["output"] = proc.stdout
                    logger.debug(f"Code execution PASSED. Output: {proc.stdout[:DISPLAY_TRUNCATION_MEDIUM]}")
                else:
                    result["error"] = proc.stderr or proc.stdout
                    logger.debug(f"Code execution FAILED. Error: {result['error'][:DISPLAY_TRUNCATION_LARGE]}")

            finally:
                # Clean up
                os.unlink(temp_path)

        except subprocess.TimeoutExpired:
            result["error"] = "Timeout"
        except Exception as e:
            result["error"] = str(e)

        return result

