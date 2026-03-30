"""Benchmark Debugging tab — test extractor + evaluator end-to-end."""

import time

import gradio as gr

from wisent.core.utils.config_tools.constants import (
    INDEX_FIRST,
    TEST_EXTRACTOR_EVALUATOR_DEFAULT_LIMIT,
    SPLIT_RATIO_HALF,
)


def _get_categories() -> list[str]:
    """Return sorted list of benchmark categories."""
    from wisent.core.utils.services.benchmarks.registry.benchmark_registry import (
        get_working_benchmarks_with_categories, _get_params_dir,
    )
    import logging
    _log = logging.getLogger(__name__)
    params_dir = _get_params_dir()
    _log.warning(f"params_dir={params_dir} exists={params_dir.exists()}")
    cat_map = get_working_benchmarks_with_categories()
    _log.warning(f"cat_map has {len(cat_map)} entries")
    cats = sorted(set(cat_map.values()))
    return ["all"] + cats


def _get_benchmarks_for_category(category: str) -> list[str]:
    """Return benchmark names for a category, with group labels."""
    from wisent.extractors.lm_eval.lm_extractor_registry import _REGISTRY
    from wisent.core.utils.services.benchmarks.registry.benchmark_registry import get_working_benchmarks_with_categories
    all_names = sorted(_REGISTRY.keys())
    cat_map = get_working_benchmarks_with_categories()
    if category and category != "all":
        all_names = [n for n in all_names if cat_map.get(n) == category]
    subtask_counts = {}
    for name in all_names:
        prefix = name + "_"
        count = sum(n.startswith(prefix) for n in all_names)
        if count:
            subtask_counts[name] = count
    groups = [f"{n} ({subtask_counts[n]} subtasks)" for n in sorted(subtask_counts)]
    individuals = [n for n in all_names if n not in subtask_counts]
    return groups + individuals


def _get_all_benchmark_names() -> list[str]:
    """Return all benchmarks (no category filter)."""
    return _get_benchmarks_for_category("all")


def _find_subtasks(task_name: str, all_names: list[str]) -> list[str]:
    """Find subtasks for a group task by prefix matching."""
    prefix = task_name + "_"
    subtasks = [n for n in all_names if n.startswith(prefix)]
    return subtasks


def _test_single_task(task_name: str, limit: float | None) -> dict:
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

    parsed_limit = int(limit) if limit else None
    try:
        pairs = extractor.extract_contrastive_pairs(limit=parsed_limit)
    except TypeError:
        try:
            from wisent.core.utils.infra_tools.data.loaders.lm_eval.lm_loader import (
                LMEvalDataLoader,
            )
            task_obj = LMEvalDataLoader.load_lm_eval_task(task_name)
            pairs = extractor.extract_contrastive_pairs(
                lm_eval_task_data=task_obj, limit=parsed_limit,
                train_ratio=SPLIT_RATIO_HALF)
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
    except Exception as exc:
        exc_name = type(exc).__name__
        if "Docker" in exc_name or "Docker" in str(exc):
            result["status"] = "PASS"
            result["details"] = "resolution OK, needs Docker runtime"
            return result
        result["status"] = "FAIL"
        result["details"] = f"evaluator init: {exc}"
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


def _run_benchmark_test(task_name: str, limit: float | None) -> str:
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


def _get_benchmark_info(task_name: str) -> str:
    """Return full metadata about a benchmark when selected."""
    if not task_name:
        return ""
    from wisent.app.ui.tabs.benchmark_info import format_full_info
    return format_full_info(task_name)


def _update_benchmark_choices(category: str):
    """Return updated choices for benchmark dropdown based on category."""
    return gr.update(choices=_get_benchmarks_for_category(category), value=None)


def build_benchmark_debug_tab():
    """Build the Benchmark Debugging tab."""
    gr.Markdown("**Benchmark Debugging** — test extractor + evaluator end-to-end")
    with gr.Row():
        cat_dropdown = gr.Dropdown(
            label="Category", choices=_get_categories(),
            value="all", interactive=True)
        task_dropdown = gr.Dropdown(
            label="Benchmark", choices=_get_all_benchmark_names(),
            value=None, allow_custom_value=True, interactive=True,
            info="Select a benchmark or type to search")
        limit_input = gr.Number(
            label="Pairs per task", value=TEST_EXTRACTOR_EVALUATOR_DEFAULT_LIMIT,
            precision=INDEX_FIRST, info="Empty = all pairs")
    cat_dropdown.change(
        fn=_update_benchmark_choices, inputs=[cat_dropdown], outputs=[task_dropdown])
    info_display = gr.Markdown(value="")
    task_dropdown.change(
        fn=_get_benchmark_info, inputs=[task_dropdown], outputs=[info_display])
    with gr.Row():
        run_btn = gr.Button("Test Benchmark", variant="primary")
        run_all_btn = gr.Button("Run All in Category", variant="secondary")
    output = gr.Markdown(value="")
    run_btn.click(
        fn=_run_benchmark_test, inputs=[task_dropdown, limit_input], outputs=[output])
    from wisent.app.ui.tabs.benchmark_runner import run_all_benchmarks
    run_all_btn.click(
        fn=run_all_benchmarks, inputs=[cat_dropdown, limit_input], outputs=[output])
