"""Benchmark Debugging tab for the Wisent Gradio interface.

Lets users select a benchmark, run the extractor + evaluator test,
and see detailed results.
"""

import inspect
import time

import gradio as gr

from wisent.core.utils.config_tools.constants import (
    INDEX_FIRST,
    TEST_EXTRACTOR_EVALUATOR_DEFAULT_LIMIT,
    TEST_EXTRACTOR_EVALUATOR_HTTP_TIMEOUT,
)


def _get_all_benchmark_names() -> list[str]:
    """Return sorted list of all registered benchmark task names (HF + lm-eval)."""
    from wisent.extractors.lm_eval.lm_extractor_registry import _REGISTRY
    return sorted(_REGISTRY.keys())


def _run_benchmark_test(task_name: str, limit: int) -> str:
    """Run the extractor + evaluator test and return formatted results."""
    lines = []
    lines.append(f"=== Testing: {task_name} ===\n")

    # A. Look up extractor (combined HF + lm-eval registry)
    from wisent.extractors.lm_eval.lm_extractor_registry import (
        get_extractor, UnsupportedLMEvalBenchmarkError,
    )
    try:
        extractor = get_extractor(task_name)
    except TypeError as exc:
        return f"FAIL: extractor constructor error: {exc}"
    except (UnsupportedLMEvalBenchmarkError, Exception) as exc:
        return f"FAIL: {exc}"

    evaluator_name = getattr(extractor, "evaluator_name", None)
    lines.append(f"Extractor: {type(extractor).__name__}")
    lines.append(f"Evaluator: {evaluator_name}")

    if not evaluator_name:
        return "\n".join(lines) + "\n\nFAIL: no evaluator_name"

    # B. Extract pairs
    start = time.time()
    try:
        pairs = extractor.extract_contrastive_pairs(limit=int(limit))
    except TypeError:
        try:
            pairs = extractor.extract_contrastive_pairs(
                lm_eval_task_data=None, limit=int(limit))
        except Exception as exc:
            return "\n".join(lines) + f"\n\nFAIL: {type(exc).__name__}: {exc}"
    except Exception as exc:
        return "\n".join(lines) + f"\n\nFAIL: {type(exc).__name__}: {exc}"

    elapsed = time.time() - start
    lines.append(f"Pairs extracted: {len(pairs)} in {elapsed:.1f}s")

    if not pairs:
        return "\n".join(lines) + "\n\nFAIL: zero pairs"

    # C. Resolve evaluator
    from wisent.core.reading.evaluators.core.atoms import (
        BaseEvaluator, EvaluatorError,
    )
    import wisent.core.reading.evaluators.core.benchmark_specific  # noqa: F401

    try:
        evaluator_cls = BaseEvaluator.get(evaluator_name)
    except EvaluatorError as exc:
        registered = sorted(BaseEvaluator.list_registered().keys())
        return "\n".join(lines) + f"\n\nFAIL: {exc}\nRegistered: {registered}"

    try:
        evaluator = evaluator_cls()
    except TypeError:
        return "\n".join(lines) + "\n\nPASS (resolution OK, needs constructor args)"

    # D. Check if evaluator needs external infra
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

    # E. Evaluate pairs
    correct_ok, incorrect_ok, errors = [], [], []

    for i, pair in enumerate(pairs):
        correct_resp = pair.positive_response.model_response
        incorrect_resp = pair.negative_response.model_response
        expected = correct_resp

        if is_infra:
            correct_ok.append(correct_resp.strip() == expected.strip())
            incorrect_ok.append(
                incorrect_resp.strip() != expected.strip())
            continue

        try:
            rc = evaluator.evaluate(
                response=correct_resp, expected=expected,
                question=pair.prompt, task_name=task_name)
            correct_ok.append(rc.ground_truth == "TRUTHFUL")
        except Exception as exc:
            errors.append(f"Pair {i} correct: {exc}")
            continue

        try:
            ri = evaluator.evaluate(
                response=incorrect_resp, expected=expected,
                question=pair.prompt, task_name=task_name)
            is_wrong = ri.ground_truth == "UNTRUTHFUL"
            incorrect_ok.append(is_wrong)
            if not is_wrong:
                errors.append(
                    f"Pair {i}: incorrect not detected "
                    f"(gt={ri.ground_truth})")
        except Exception as exc:
            errors.append(f"Pair {i} incorrect: {exc}")

    # F. Report
    n_c, n_i = sum(correct_ok), sum(incorrect_ok)
    t_c, t_i = len(correct_ok), len(incorrect_ok)

    mode = "string comparison" if is_infra else "evaluator"
    lines.append(f"\nMode: {mode}")
    lines.append(f"Correct -> TRUTHFUL:    {n_c}/{t_c}")
    lines.append(f"Incorrect -> UNTRUTHFUL: {n_i}/{t_i}")

    if errors:
        lines.append(f"\nErrors ({len(errors)}):")
        for e in errors:
            lines.append(f"  - {e}")

    if t_c and t_i and n_c == t_c and n_i == t_i:
        lines.append("\nPASS")
    elif t_c and t_i:
        lines.append("\nFAIL")
    elif errors:
        lines.append(f"\nFAIL ({len(errors)} errors)")
    else:
        lines.append("\nPASS (resolution OK)")

    return "\n".join(lines)


def build_benchmark_debug_tab():
    """Build the Benchmark Debugging tab."""
    gr.Markdown(
        "**Benchmark Debugging** — test that an extractor "
        "and evaluator work end-to-end for a given benchmark."
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
            label="Pairs to extract",
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
