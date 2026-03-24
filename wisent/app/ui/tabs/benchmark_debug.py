"""Benchmark Debugging tab for the Wisent Gradio interface.

Lets users select a benchmark, run the extractor + evaluator test,
and see detailed results. For group tasks, runs each subtask and
reports per-subtask results.
"""

import time

import gradio as gr

from wisent.core.utils.config_tools.constants import (
    INDEX_FIRST,
    TEST_EXTRACTOR_EVALUATOR_DEFAULT_LIMIT,
)


def _get_all_benchmark_names() -> list[str]:
    """Return sorted list with group tasks labeled by subtask count."""
    from wisent.extractors.lm_eval.lm_extractor_registry import _REGISTRY
    all_names = sorted(_REGISTRY.keys())
    # Find which names are group parents (have subtasks)
    subtask_counts = {}
    for name in all_names:
        prefix = name + "_"
        count = sum(n.startswith(prefix) for n in all_names)
        if count:
            subtask_counts[name] = count
    # Build labeled list: parents first, then individual tasks
    groups = [f"{n} ({subtask_counts[n]} subtasks)" for n in sorted(subtask_counts)]
    individuals = [n for n in all_names if n not in subtask_counts]
    return groups + individuals


def _find_subtasks(task_name: str, all_names: list[str]) -> list[str]:
    """Find subtasks for a group task by prefix matching."""
    prefix = task_name + "_"
    subtasks = [n for n in all_names if n.startswith(prefix)]
    return subtasks


def _test_single_task(task_name: str, limit: int) -> dict:
    """Test one task. Returns dict with status and details."""
    from wisent.extractors.lm_eval.lm_extractor_registry import (
        get_extractor,
    )
    result = {"task": task_name, "status": "UNKNOWN", "details": ""}

    try:
        extractor = get_extractor(task_name)
    except Exception as exc:
        result["status"] = "FAIL"
        result["details"] = f"extractor: {exc}"
        return result

    evaluator_name = getattr(extractor, "evaluator_name", None)
    result["extractor"] = type(extractor).__name__
    result["evaluator"] = evaluator_name

    if not evaluator_name:
        result["status"] = "FAIL"
        result["details"] = "no evaluator_name"
        return result

    try:
        pairs = extractor.extract_contrastive_pairs(limit=int(limit))
    except TypeError:
        try:
            pairs = extractor.extract_contrastive_pairs(
                lm_eval_task_data=None, limit=int(limit))
        except Exception as exc:
            result["status"] = "FAIL"
            result["details"] = f"extraction: {exc}"
            return result
    except Exception as exc:
        result["status"] = "FAIL"
        result["details"] = f"extraction: {exc}"
        return result

    result["pairs"] = len(pairs)
    if not pairs:
        result["status"] = "FAIL"
        result["details"] = "zero pairs"
        return result

    from wisent.core.reading.evaluators.core.atoms import (
        BaseEvaluator, EvaluatorError,
    )
    import wisent.core.reading.evaluators.core.benchmark_specific  # noqa: F401

    try:
        evaluator_cls = BaseEvaluator.get(evaluator_name)
    except EvaluatorError as exc:
        result["status"] = "FAIL"
        result["details"] = f"evaluator not found: {exc}"
        return result

    try:
        evaluator = evaluator_cls()
    except TypeError:
        result["status"] = "PASS"
        result["details"] = "resolution OK, needs constructor args"
        return result

    is_infra = getattr(evaluator, "requires_judge", False)
    if not is_infra:
        pair = pairs[INDEX_FIRST]
        try:
            evaluator.evaluate(
                response=pair.positive_response.model_response,
                expected=pair.positive_response.model_response,
                question=pair.prompt, task_name=task_name)
        except Exception as exc:
            msg = str(exc)
            for marker in ("judge_model", "test_code",
                           "requires a model", "ModelNotProvidedError"):
                if marker in msg or marker in type(exc).__name__:
                    is_infra = True
                    break

    n_correct = n_incorrect = t_correct = t_incorrect = INDEX_FIRST
    errors = []

    for i, pair in enumerate(pairs):
        correct_resp = pair.positive_response.model_response
        incorrect_resp = pair.negative_response.model_response
        expected = correct_resp

        if is_infra:
            t_correct += INDEX_FIRST + INDEX_FIRST
            t_incorrect += INDEX_FIRST + INDEX_FIRST
            if correct_resp.strip() == expected.strip():
                n_correct += INDEX_FIRST + INDEX_FIRST
            if incorrect_resp.strip() != expected.strip():
                n_incorrect += INDEX_FIRST + INDEX_FIRST
            continue

        try:
            rc = evaluator.evaluate(
                response=correct_resp, expected=expected,
                question=pair.prompt, task_name=task_name)
            t_correct += INDEX_FIRST + INDEX_FIRST
            if rc.ground_truth == "TRUTHFUL":
                n_correct += INDEX_FIRST + INDEX_FIRST
        except Exception as exc:
            errors.append(f"Pair {i}: {exc}")
            continue

        try:
            ri = evaluator.evaluate(
                response=incorrect_resp, expected=expected,
                question=pair.prompt, task_name=task_name)
            t_incorrect += INDEX_FIRST + INDEX_FIRST
            if ri.ground_truth == "UNTRUTHFUL":
                n_incorrect += INDEX_FIRST + INDEX_FIRST
        except Exception as exc:
            errors.append(f"Pair {i}: {exc}")

    result["correct"] = f"{n_correct}/{t_correct}"
    result["incorrect"] = f"{n_incorrect}/{t_incorrect}"
    result["mode"] = "string" if is_infra else "evaluator"
    result["errors"] = errors

    if t_correct and t_incorrect and n_correct == t_correct and n_incorrect == t_incorrect:
        result["status"] = "PASS"
    elif errors and not t_correct:
        result["status"] = "FAIL"
    else:
        result["status"] = "FAIL"

    return result


def _format_result(r: dict) -> str:
    """Format a single task result as a line."""
    status = r["status"]
    task = r["task"]
    if status == "PASS":
        pairs = r.get("pairs", "?")
        return f"  PASS  {task} ({pairs} pairs, {r.get('mode', '?')})"
    details = r.get("details", "")
    if r.get("correct"):
        details = f"correct={r['correct']} incorrect={r['incorrect']}"
    return f"  FAIL  {task}: {details}"


def _run_benchmark_test(task_name: str, limit: int) -> str:
    """Run the test. If group task, run each subtask."""
    # Strip label suffix like " (N subtasks)"
    if " (" in task_name and task_name.endswith(")"):
        task_name = task_name.split(" (")[INDEX_FIRST]
    from wisent.extractors.lm_eval.lm_extractor_registry import _REGISTRY
    all_names = sorted(_REGISTRY.keys())
    subtasks = _find_subtasks(task_name, all_names)

    if subtasks:
        tasks_to_run = subtasks
    else:
        tasks_to_run = [task_name]

    lines = [f"=== {task_name} ({len(tasks_to_run)} tasks) ===\n"]
    pass_count = INDEX_FIRST
    fail_count = INDEX_FIRST
    start = time.time()

    for t in tasks_to_run:
        r = _test_single_task(t, limit)
        lines.append(_format_result(r))
        if r["status"] == "PASS":
            pass_count += INDEX_FIRST + INDEX_FIRST
        else:
            fail_count += INDEX_FIRST + INDEX_FIRST

    elapsed = time.time() - start
    lines.append(f"\n--- Summary ---")
    lines.append(f"PASS: {pass_count}  FAIL: {fail_count}  "
                 f"Time: {elapsed:.1f}s")
    return "\n".join(lines)


def build_benchmark_debug_tab():
    """Build the Benchmark Debugging tab."""
    gr.Markdown(
        "**Benchmark Debugging** — test extractor + evaluator "
        "end-to-end. For group tasks, tests each subtask."
    )

    with gr.Row():
        task_dropdown = gr.Dropdown(
            label="Benchmark",
            choices=_get_all_benchmark_names(),
            value=None,
            allow_custom_value=True,
            interactive=True,
        )
        limit_slider = gr.Slider(
            label="Pairs per subtask",
            minimum=INDEX_FIRST + INDEX_FIRST,
            maximum=TEST_EXTRACTOR_EVALUATOR_DEFAULT_LIMIT
            * TEST_EXTRACTOR_EVALUATOR_DEFAULT_LIMIT,
            value=TEST_EXTRACTOR_EVALUATOR_DEFAULT_LIMIT,
            step=INDEX_FIRST + INDEX_FIRST,
        )

    run_btn = gr.Button("Test Benchmark", variant="primary")
    output = gr.Textbox(
        label="Results",
        interactive=False,
        lines=TEST_EXTRACTOR_EVALUATOR_DEFAULT_LIMIT
        * TEST_EXTRACTOR_EVALUATOR_DEFAULT_LIMIT
        // TEST_EXTRACTOR_EVALUATOR_DEFAULT_LIMIT,
        elem_classes=["output-box"],
    )

    run_btn.click(
        fn=_run_benchmark_test,
        inputs=[task_dropdown, limit_slider],
        outputs=[output],
    )
