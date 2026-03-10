"""Auto-grouped modules."""


def __getattr__(name):
    """Lazy re-exports from pipeline.pipeline to avoid circular imports."""
    _exports = {
        "OptimizationResult", "run_pipeline", "create_objective",
        "_make_args", "_build_config",
    }
    if name in _exports:
        from .pipeline import (
            OptimizationResult, run_pipeline, create_objective,
            _make_args, _build_config,
        )
        _mapping = {
            "OptimizationResult": OptimizationResult,
            "run_pipeline": run_pipeline,
            "create_objective": create_objective,
            "_make_args": _make_args,
            "_build_config": _build_config,
        }
        globals().update(_mapping)
        return _mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
