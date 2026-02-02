"""
Extraction strategy diagnostics package.

Provides comprehensive analysis of extraction strategies to determine which
produces the best activations for steering.

Usage:
    from wisent.core.activations.core.diagnostics import (
        StrategyDiagnostics,
        run_strategy_diagnostics,
        compare_strategies,
        generate_diagnostics_report,
        run_full_diagnostics,
    )

    # Run full diagnostics
    diagnostics, report = run_full_diagnostics(model, pairs, layer=8)
    print(report)

    # Or analyze a single strategy
    diag = run_strategy_diagnostics(pos_acts, neg_acts, ExtractionStrategy.CHAT_LAST)
    print(f"Score: {diag.overall_score}")
"""

from .strategy_diagnostics import StrategyDiagnostics, run_strategy_diagnostics
from .report import compare_strategies, generate_diagnostics_report, run_full_diagnostics
from .metrics import (
    compute_pairwise_consistency,
    compute_linear_nonlinear_accuracy,
    analyze_mc_confound,
    compute_steering_quality,
)

__all__ = [
    "StrategyDiagnostics",
    "run_strategy_diagnostics",
    "compare_strategies",
    "generate_diagnostics_report",
    "run_full_diagnostics",
    "compute_pairwise_consistency",
    "compute_linear_nonlinear_accuracy",
    "analyze_mc_confound",
    "compute_steering_quality",
]
