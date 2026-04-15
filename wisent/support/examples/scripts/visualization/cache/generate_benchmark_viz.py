"""Pre-generate visualization cache for a benchmark.

Downloads activations from HuggingFace Hub and generates all
representation visualizations, saving them to the local cache
so the Gradio Benchmark Debug tab can display them instantly.

Usage:
    python -m wisent.support.examples.scripts.visualization.cache.generate_benchmark_viz \
        truthfulqa_custom --model meta-llama/Llama-3.2-1B-Instruct
    python -m wisent.support.examples.scripts.visualization.cache.generate_benchmark_viz \
        truthfulqa_custom --model meta-llama/Llama-3.2-1B-Instruct --layer 12
"""

import argparse
import sys
import time


def generate_viz(
    task_name: str,
    model_name: str,
    layer: int | None = None,
) -> dict:
    """Generate and cache visualizations for a benchmark.

    Args:
        task_name: Benchmark task name.
        model_name: HuggingFace model ID.
        layer: Specific layer, or None to generate for all available layers.

    Returns:
        Dict mapping layer number to status string.
    """
    from wisent.core.control.steering_methods.configs.optimal import get_optimal
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import (
        load_activations_from_hf,
        load_available_layers_from_hf,
    )
    from wisent.core.reading.modules.utilities.data.cache import (
        load_viz_cache, save_viz_cache,
    )
    from wisent.core.reading.modules.utilities.metrics.core.metrics_viz import (
        generate_metrics_visualizations,
    )
    from wisent.core.utils.config_tools.constants import VIZ_SUMMARY_KEY

    strategy = get_optimal("extraction_strategy")

    if layer is not None:
        layers = [layer]
    else:
        print(f"Looking up available layers for {model_name}/{task_name}/{strategy}...")
        layers = load_available_layers_from_hf(model_name, task_name, strategy)
        print(f"Found {len(layers)} layers: {layers}")

    results = {}
    for i, current_layer in enumerate(layers):
        label = f"[{i + 1}/{len(layers)}] Layer {current_layer}"
        print(f"\n{label}: downloading activations...")
        start = time.time()

        pos, neg = load_activations_from_hf(
            model_name, task_name, current_layer, strategy,
        )
        dl_time = time.time() - start
        print(f"  {len(pos)} pairs downloaded ({dl_time:.1f}s)")

        print(f"  Generating visualizations...")
        viz_start = time.time()

        cached = load_viz_cache(task_name, model_name, current_layer)
        if cached:
            visualizations = cached
            print(f"  Already cached, skipping generation")
        else:
            visualizations = generate_metrics_visualizations(
                pos, neg, metrics={},
            )
            try:
                from wisent.core.utils.visualization.geometry.public.summary_figure import (
                    create_full_summary_figure,
                )
                summary_b64 = create_full_summary_figure(
                    pos, neg, metrics=None, layer_num=current_layer,
                )
                visualizations[VIZ_SUMMARY_KEY] = summary_b64
            except Exception as exc:
                print(f"  Warning: summary figure skipped: {exc}")
            save_viz_cache(task_name, model_name, current_layer, visualizations)

        viz_time = time.time() - viz_start
        n_viz = len(visualizations)
        print(f"  {n_viz} visualizations cached ({viz_time:.1f}s)")

        results[current_layer] = f"OK ({n_viz} viz, {dl_time + viz_time:.1f}s)"

    return results


def generate_all_from_index():
    """Generate viz for all benchmarks in the HF index."""
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import (
        _hf_hub_download,
    )
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_config import (
        safe_name_to_model,
    )
    import json

    local = _hf_hub_download("index.json")
    with open(local, "r") as f:
        index = json.load(f)

    strategy = get_optimal("extraction_strategy")
    combos = set()
    for key in index:
        parts = key.split("/")
        if len(parts) >= 3 and parts[-1] == strategy:
            safe_model = parts[0]
            task = "/".join(parts[1:-1])
            model = safe_name_to_model(safe_model)
            combos.add((model, task))

    combos = sorted(combos)
    print(f"Found {len(combos)} model/benchmark combos")
    for i, (model, task) in enumerate(combos):
        print(f"\n[{i + 1}/{len(combos)}] {model} / {task}")
        try:
            results = generate_viz(task, model)
            n_layers = len(results)
            print(f"  Done: {n_layers} layers")
        except Exception as exc:
            print(f"  Failed: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate visualization cache for a benchmark.",
    )
    parser.add_argument("task", nargs="?", help="Benchmark task name")
    parser.add_argument("--model", help="HuggingFace model ID")
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Specific layer (default: all available)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate viz for all benchmarks in HF index",
    )
    args = parser.parse_args()

    if args.all:
        generate_all_from_index()
        sys.exit(0)

    if not args.task or not args.model:
        parser.error("task and --model required (or use --all)")

    print(f"Generating visualizations for {args.task} / {args.model}")
    results = generate_viz(args.task, args.model, args.layer)

    print(f"\n{'=' * 40}")
    print(f"Results for {args.task} / {args.model}:")
    for layer_num, status in sorted(results.items()):
        print(f"  Layer {layer_num}: {status}")

    sys.exit(0)


if __name__ == "__main__":
    main()
