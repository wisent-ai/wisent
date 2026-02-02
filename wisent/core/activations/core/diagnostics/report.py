"""
Report generation and full diagnostics orchestration.

Provides functions to compare strategies, generate reports, and run full diagnostics.
"""

from typing import List, Tuple, Optional
import torch

from wisent.core.activations import ExtractionStrategy
from .strategy_diagnostics import StrategyDiagnostics, run_strategy_diagnostics


def compare_strategies(diagnostics_list: List[StrategyDiagnostics]) -> List[StrategyDiagnostics]:
    """Rank strategies by overall score (descending)."""
    return sorted(diagnostics_list, key=lambda d: d.overall_score, reverse=True)


def generate_diagnostics_report(diagnostics_list: List[StrategyDiagnostics]) -> str:
    """Generate a formatted diagnostics report."""
    ranked = compare_strategies(diagnostics_list)
    lines = []
    lines.append("=" * 80)
    lines.append("EXTRACTION STRATEGY DIAGNOSTICS REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'Strategy':<15} {'LinAcc':>8} {'Consist':>10} {'Geom':>10} {'SteerAcc':>10} {'Score':>8}")
    lines.append("-" * 80)

    for d in ranked:
        steer = f"{d.steering_accuracy:.3f}" if d.steering_accuracy else "N/A"
        lines.append(
            f"{d.strategy:<15} {d.linear_accuracy:>8.3f} {d.consistency:>10.4f} "
            f"{d.geometry:>10} {steer:>10} {d.overall_score:>8.3f}"
        )

    best = ranked[0]
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"RECOMMENDATION: {best.strategy}")
    lines.append("=" * 80)
    lines.append(f"  Linear accuracy:   {best.linear_accuracy:.3f}")
    lines.append(f"  Consistency:       {best.consistency:.4f}")
    if best.steering_accuracy:
        lines.append(f"  Steering accuracy: {best.steering_accuracy:.3f}")
    lines.append(f"  Method:            {best.recommended_method}")

    lines.append("")
    lines.append("WARNINGS:")
    for d in ranked:
        if d.warnings:
            lines.append(f"  {d.strategy}: {'; '.join(d.warnings)}")

    mc_strats = [d for d in ranked if d.is_mc_strategy and d.ab_variance_fraction is not None]
    if mc_strats:
        lines.append("")
        lines.append("MC CONFOUND ANALYSIS:")
        for d in mc_strats:
            lines.append(f"  {d.strategy}: A/B={d.ab_variance_fraction*100:.1f}%, semantic={d.semantic_consistency:.4f}")

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


def run_full_diagnostics(
    model,
    pairs,
    layer: int,
    strategies: Optional[List[ExtractionStrategy]] = None,
    collector=None,
) -> Tuple[List[StrategyDiagnostics], str]:
    """
    Run full diagnostics across multiple strategies.

    Args:
        model: WisentModel instance
        pairs: List of contrastive pairs
        layer: Layer to extract from
        strategies: List of strategies (default: main strategies)
        collector: Optional ActivationCollector

    Returns:
        Tuple of (diagnostics list, formatted report)
    """
    from wisent.core.activations.activations_collector import ActivationCollector

    if strategies is None:
        strategies = [
            ExtractionStrategy.CHAT_LAST,
            ExtractionStrategy.CHAT_FIRST,
            ExtractionStrategy.CHAT_MEAN,
            ExtractionStrategy.ROLE_PLAY,
            ExtractionStrategy.MC_BALANCED,
        ]

    if collector is None:
        store_dev = "mps" if torch.backends.mps.is_available() else "cpu"
        collector = ActivationCollector(model=model, store_device=store_dev)

    all_diagnostics = []
    for strategy in strategies:
        print(f"Analyzing {strategy.value}...")
        pos_acts, neg_acts, letters = [], [], []

        for pair in pairs:
            pair_acts = collector.collect(pair, strategy=strategy, layers=[str(layer)])
            pos = pair_acts.positive_response.layers_activations[str(layer)]
            neg = pair_acts.negative_response.layers_activations[str(layer)]
            pos_acts.append(pos.flatten())
            neg_acts.append(neg.flatten())
            if strategy in (ExtractionStrategy.MC_BALANCED, ExtractionStrategy.MC_COMPLETION):
                letters.append("B" if hash(pair.prompt) % 2 == 0 else "A")

        diag = run_strategy_diagnostics(
            torch.stack(pos_acts),
            torch.stack(neg_acts),
            strategy,
            letters if letters else None
        )
        all_diagnostics.append(diag)
        print(f"  Linear: {diag.linear_accuracy:.3f}, Consistency: {diag.consistency:.4f}")

    return all_diagnostics, generate_diagnostics_report(all_diagnostics)
