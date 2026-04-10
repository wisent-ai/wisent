"""Visualization data logic for Benchmark Debug tab.

Loads activations from local cache or HuggingFace Hub,
generates representation visualizations (summary grid + individual),
caches results as base64 JSON, and converts to file paths for Gallery.
"""

import base64
import logging
import tempfile
from pathlib import Path
from typing import Optional

from wisent.core.utils.config_tools.constants import (
    VIZ_SUMMARY_KEY,
)

_log = logging.getLogger(__name__)

_TEMP_DIR: Optional[tempfile.TemporaryDirectory] = None


def _get_temp_dir() -> str:
    """Get or create a session-scoped temp directory for PNG files."""
    global _TEMP_DIR
    if _TEMP_DIR is None:
        _TEMP_DIR = tempfile.TemporaryDirectory(prefix="wisent_viz_")
    return _TEMP_DIR.name


def base64_to_filepath(b64_str: str, name: str) -> str:
    """Write a base64-encoded PNG to a temp file and return its path."""
    tmp = Path(_get_temp_dir()) / f"{name}.png"
    tmp.write_bytes(base64.b64decode(b64_str))
    return str(tmp)


def _get_extraction_strategy() -> str:
    """Get the optimal extraction strategy."""
    from wisent.core.control.steering_methods.configs.optimal import get_optimal
    return get_optimal("extraction_strategy")


def discover_available_models(task_name: str) -> list[str]:
    """Find models that have activations for a task on HF Hub."""
    try:
        from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import (
            _hf_hub_download,
        )
        from wisent.core.reading.modules.utilities.data.sources.hf.hf_config import (
            safe_name_to_model,
        )
        import json

        local_path = _hf_hub_download("index.json")
        with open(local_path, "r") as f:
            index = json.load(f)

        strategy = _get_extraction_strategy()
        models = set()
        for key in index:
            parts = key.split("/")
            if len(parts) >= 3 and parts[-2] == task_name and parts[-1] == strategy:
                safe_model = "/".join(parts[:-2])
                models.add(safe_name_to_model(safe_model))
        return sorted(models)
    except Exception as exc:
        _log.info("Could not discover models for %s: %s", task_name, exc)
        return []


def discover_available_layers(
    task_name: str, model_name: str,
) -> list[int]:
    """Find available layers from local cache, then HF Hub."""
    from wisent.core.reading.modules.utilities.data.cache import get_cached_layers

    layers = get_cached_layers(task_name, model_name)
    if layers:
        return layers

    try:
        from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import (
            load_available_layers_from_hf,
        )
        strategy = _get_extraction_strategy()
        layers = load_available_layers_from_hf(model_name, task_name, strategy)
        return layers
    except Exception as exc:
        _log.info("No HF layers for %s/%s: %s", model_name, task_name, exc)
        return []


def load_activations(
    task_name: str, model_name: str, layer: int,
):
    """Load activations from local cache or HF Hub.

    Returns:
        Tuple (pos_tensor, neg_tensor) or (None, None) if unavailable.
    """
    from wisent.core.reading.modules.utilities.data.cache import (
        load_activations_cache,
    )

    pos, neg, _ = load_activations_cache(task_name, model_name, layer)
    if pos is not None:
        return pos, neg

    try:
        from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import (
            load_activations_from_hf,
        )
        strategy = _get_extraction_strategy()
        pos, neg = load_activations_from_hf(
            model_name, task_name, layer, strategy,
        )
        return pos, neg
    except Exception as exc:
        _log.info(
            "No activations for %s/%s/L%d: %s",
            model_name, task_name, layer, exc,
        )
        return None, None


def generate_and_cache_visualizations(
    task_name: str, model_name: str, layer: int,
    pos_activations, neg_activations,
) -> dict[str, str]:
    """Generate all visualizations and cache them.

    Returns:
        Dict mapping visualization name to base64-encoded PNG string.
    """
    from wisent.core.reading.modules.utilities.data.cache import (
        load_viz_cache, save_viz_cache,
    )

    cached = load_viz_cache(task_name, model_name, layer)
    if cached:
        return cached

    from wisent.core.reading.modules.utilities.metrics.core.metrics_viz import (
        generate_metrics_visualizations,
    )
    from wisent.core.utils.visualization.geometry.public.summary_figure import (
        create_full_summary_figure,
    )

    visualizations = generate_metrics_visualizations(
        pos_activations, neg_activations, metrics={},
    )

    try:
        summary_b64 = create_full_summary_figure(
            pos_activations, neg_activations,
            metrics=None, layer_num=layer,
        )
        visualizations[VIZ_SUMMARY_KEY] = summary_b64
    except Exception as exc:
        _log.warning("Summary figure failed: %s", exc)

    save_viz_cache(task_name, model_name, layer, visualizations)
    return visualizations


def viz_to_gallery_paths(
    visualizations: dict[str, str],
) -> list[tuple[str, str]]:
    """Convert viz dict to (filepath, label) tuples for gr.Gallery.

    Excludes the summary (shown separately via gr.Image).
    """
    paths = []
    for name, b64 in sorted(visualizations.items()):
        if name == VIZ_SUMMARY_KEY:
            continue
        path = base64_to_filepath(b64, name)
        label = name.replace("_", " ").title()
        paths.append((path, label))
    return paths


def get_summary_path(visualizations: dict[str, str]) -> str | None:
    """Extract summary image path, or None if absent."""
    b64 = visualizations.get(VIZ_SUMMARY_KEY)
    if b64:
        return base64_to_filepath(b64, VIZ_SUMMARY_KEY)
    return None
