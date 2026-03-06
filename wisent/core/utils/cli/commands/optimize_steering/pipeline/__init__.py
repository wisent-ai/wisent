"""Auto-grouped modules."""


def __getattr__(name):
    """Lazy re-exports from pipeline.pipeline to avoid circular imports."""
    _exports = {
        "OptimizationResult", "run_pipeline", "create_objective", "_make_args",
    }
    if name in _exports:
        from .pipeline import (
            OptimizationResult, run_pipeline, create_objective, _make_args,
        )
        _mapping = {
            "OptimizationResult": OptimizationResult,
            "run_pipeline": run_pipeline,
            "create_objective": create_objective,
            "_make_args": _make_args,
        }
        globals().update(_mapping)
        return _mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
