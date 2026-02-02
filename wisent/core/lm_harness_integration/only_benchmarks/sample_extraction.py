"""Sample extraction functions for benchmark tasks."""

from __future__ import annotations

import os
from typing import Optional

from wisent.core.errors import TaskLoadError, FallbackNotPermittedError

from ..populate_tasks import get_task_samples_for_analysis as _get_task_samples_for_analysis
from .sample_helpers import get_task_samples_direct


__all__ = [
    "get_task_samples_for_analysis",
    "get_task_samples_with_subtasks",
]


def get_task_samples_for_analysis(
    task_name: str, num_samples: int = 5, trust_remote_code: bool = False
) -> dict:
    """Enhanced wrapper for get_task_samples_for_analysis with trust_remote_code."""
    try:
        original_env = {}
        if trust_remote_code:
            env_vars = {
                "HF_ALLOW_CODE_EVAL": "1",
                "TRUST_REMOTE_CODE": "1",
                "HF_DATASETS_TRUST_REMOTE_CODE": "1",
            }
            for key, value in env_vars.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value

        try:
            result = _get_task_samples_for_analysis(task_name, num_samples=num_samples)
            if "error" not in result:
                return result
        except Exception as e:
            print(f"Initial attempt failed: {e}")

        if trust_remote_code:
            try:
                from lm_eval import evaluator

                os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
                task_dict = evaluator.get_task_dict([task_name])
                if task_name in task_dict:
                    task = task_dict[task_name]
                    return get_task_samples_direct(task, num_samples=num_samples)
                return {
                    "error": f"Task {task_name} not found with trust_remote_code handling"
                }
            except Exception as e:
                print(f"Trust remote code handling failed: {e}")
                return {"error": f"Failed with trust_remote_code handling: {e}"}

        raise TaskLoadError(task_name=task_name)

    except Exception as e:
        return {"error": f"Exception in get_task_samples_for_analysis: {e}"}
    finally:
        if trust_remote_code:
            for key, original_value in original_env.items():
                if original_value is None:
                    if key in os.environ:
                        del os.environ[key]
                else:
                    os.environ[key] = original_value


def get_task_samples_with_subtasks(
    task_name: str,
    num_samples: int = 5,
    trust_remote_code: bool = False,
    limit_subtasks: Optional[int] = None,
) -> dict:
    """Get samples from a task that has subtasks."""
    try:
        try:
            result = get_task_samples_for_analysis(
                task_name, num_samples=num_samples, trust_remote_code=trust_remote_code
            )
            if result.get("samples"):
                return result
        except Exception:
            pass

        try:
            original_env = {}
            if trust_remote_code:
                env_vars = {
                    "HF_ALLOW_CODE_EVAL": "1",
                    "TRUST_REMOTE_CODE": "1",
                    "HF_DATASETS_TRUST_REMOTE_CODE": "1",
                }
                for key, value in env_vars.items():
                    original_env[key] = os.environ.get(key)
                    os.environ[key] = value

            from lm_eval import evaluator

            os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
            task_dict = evaluator.get_task_dict([task_name])

            if task_name in task_dict:
                task = task_dict[task_name]

                if hasattr(task, "get_task_names") or hasattr(task, "get_tasks"):
                    if hasattr(task, "get_task_names"):
                        subtask_names = task.get_task_names()
                    else:
                        subtask_names = list(task.get_tasks().keys())

                    if limit_subtasks:
                        subtask_names = subtask_names[:limit_subtasks]

                    print(f"Found {len(subtask_names)} subtasks, trying first few...")

                    for i, subtask_name in enumerate(subtask_names[:3]):
                        print(f"   Trying subtask {i+1}: {subtask_name}")
                        try:
                            result = get_task_samples_for_analysis(
                                subtask_name,
                                num_samples=num_samples,
                                trust_remote_code=trust_remote_code,
                            )
                            if result.get("samples"):
                                print(f"   Success with subtask: {subtask_name}")
                                return result
                        except Exception as e:
                            print(f"   Subtask {subtask_name} failed: {e}")
                            continue

                    return {"error": f"No samples from any subtasks of {task_name}"}
                else:
                    return get_task_samples_direct(task, num_samples=num_samples)
            else:
                return {"error": f"Task {task_name} not found in task dict"}

        except Exception as e:
            return {"error": f"Exception in subtask handling: {e}"}
        finally:
            if trust_remote_code:
                for key, original_value in original_env.items():
                    if original_value is None:
                        if key in os.environ:
                            del os.environ[key]
                    else:
                        os.environ[key] = original_value

    except Exception as e:
        return {"error": f"Exception in get_task_samples_with_subtasks: {e}"}


def get_task_samples_fallback(
    task_name: str, num_samples: int = 5, trust_remote_code: bool = False
) -> dict:
    """DEPRECATED: Fallback loading is not permitted."""
    raise FallbackNotPermittedError(task_name=task_name)
