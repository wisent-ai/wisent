"""Benchmark Debugging tab — test extractor + evaluator end-to-end."""

import time

import gradio as gr

from wisent.core.utils.config_tools.constants import (
    INDEX_FIRST,
    TEST_EXTRACTOR_EVALUATOR_DEFAULT_LIMIT,
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


def _format_result(result: dict) -> str:
    """Format test_benchmark result dict as markdown."""
    task = result["task"]
    ext = result.get("extraction", {})
    evl = result.get("evaluator", {})

    lines = [f"### {task}\n"]

    # Extraction
    ext_status = ext.get("status", "SKIP")
    ext_pairs = ext.get("pair_count", 0)
    lines.append(f"**Extraction:** {ext_status} ({ext_pairs} pairs)")
    if ext.get("detail"):
        lines.append(f"  - {ext['detail'][:200]}")

    # Evaluator
    evl_status = evl.get("status", "SKIP")
    lines.append(f"**Evaluator:** {evl_status}")
    if evl.get("detail"):
        lines.append(f"  - {evl['detail'][:200]}")
    evaluation = evl.get("evaluation", {})
    if evaluation:
        num_eval = evaluation.get("num_evaluated", 0)
        num_total = evaluation.get("num_total", 0)
        evaluator_used = evaluation.get("evaluator_used", "?")
        lines.append(f"  - Evaluator: {evaluator_used}, {num_eval}/{num_total} evaluated")
        metrics = evaluation.get("aggregated_metrics", {})
        for k, v in metrics.items():
            lines.append(f"  - {k}: {v:.4f}" if isinstance(v, float) else f"  - {k}: {v}")

    return "\n".join(lines)


def _run_benchmark_test(task_name: str, limit: float | None) -> str:
    """Run test_benchmark from test_single_benchmark. Returns markdown."""
    if not task_name:
        return "No benchmark selected."

    # Strip label suffix like " (N subtasks)"
    if " (" in task_name and task_name.endswith(")"):
        task_name = task_name.split(" (")[INDEX_FIRST]

    from wisent.support.examples.scripts.discovery.validation.test_single_benchmark import test_benchmark

    start = time.time()
    result = test_benchmark(task_name)
    elapsed = time.time() - start

    output = _format_result(result)
    output += f"\n\n*Time: {elapsed:.1f}s*"
    return output


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
