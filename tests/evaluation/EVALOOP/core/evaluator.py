"""Evaluation pipeline for judging generated responses."""
from typing import List, Dict, Optional
from collections import defaultdict

from tests.EVALOOP.core.models import GenerationResult, EvaluationResult, ConfigStatistics
from tests.EVALOOP.core.config import TraitConfig, EvaluationConfig
from tests.EVALOOP.core.judge import LLMJudge
from tests.EVALOOP.core.metrics import MetricManager


class Evaluator:
    """Evaluates generation results using LLM judge."""

    def __init__(self, eval_config: EvaluationConfig, trait_config: TraitConfig):
        """
        Initialize evaluator.

        Args:
            eval_config: Evaluation configuration
            trait_config: Trait-specific configuration
        """
        self.eval_config = eval_config
        self.trait_config = trait_config
        self.judge = LLMJudge(
            model=eval_config.judge_model,
            max_tokens=eval_config.judge_max_tokens
        )
        self.metric_manager = MetricManager(trait_config.instruction_prompts)

    def _extract_results_from_responses(self, judge_responses: Dict[str, str]) -> Dict[str, any]:
        """
        Extract metric results from judge responses.

        Args:
            judge_responses: Dictionary mapping metric names to judge responses

        Returns:
            Dictionary containing extracted metric results
        """
        differentiation_score = self.metric_manager.get_metric("differentiation").extract_result(
            judge_responses["differentiation"]
        )
        coherence_score = self.metric_manager.get_metric("coherence").extract_result(
            judge_responses["coherence"]
        )
        trait_alignment_score = self.metric_manager.get_metric("trait_alignment").extract_result(
            judge_responses["trait_alignment"]
        )
        open_traits = self.metric_manager.get_metric("open").extract_result(
            judge_responses["open"]
        )
        choose_result = self.metric_manager.get_metric("choose").extract_result(
            judge_responses["choose"]
        )

        # Print warnings for failed extractions
        if differentiation_score is None:
            print(f"    Warning: Could not extract differentiation score")
        if coherence_score is None:
            print(f"    Warning: Could not extract coherence score")
        if trait_alignment_score is None:
            print(f"    Warning: Could not extract trait alignment score")
        if open_traits is None:
            print(f"    Warning: Could not extract open traits")
        if choose_result is None:
            print(f"    Warning: Could not extract choose result")

        return {
            "differentiation_score": differentiation_score,
            "coherence_score": coherence_score,
            "trait_alignment_score": trait_alignment_score,
            "open_traits": open_traits,
            "choose_result": choose_result
        }

    def _extract_explanations_from_responses(self, judge_responses: Dict[str, str]) -> Dict[str, str]:
        """
        Extract explanations from judge responses.

        Args:
            judge_responses: Dictionary mapping metric names to judge responses

        Returns:
            Dictionary containing extracted explanations for each metric
        """
        explanations = {}

        for metric_name, response in judge_responses.items():
            metric = self.metric_manager.get_metric(metric_name)
            explanation = metric.extract_explanation(response)
            explanations[metric_name] = explanation if explanation else "No explanation available"

        return explanations

    def _calculate_overall_score(
        self,
        differentiation_score: Optional[float],
        coherence_score: Optional[float],
        trait_alignment_score: Optional[float]
    ) -> Optional[float]:
        """
        Calculate weighted overall score from individual metrics.

        Args:
            differentiation_score: Differentiation metric score
            coherence_score: Coherence metric score
            trait_alignment_score: Trait alignment metric score

        Returns:
            Overall weighted score, or None if any required score is missing
        """
        if all([differentiation_score is not None, coherence_score is not None,
                trait_alignment_score is not None]):
            weights = self.eval_config.metric_weights
            overall_score = (
                weights["differentiation"] * differentiation_score +
                weights["coherence"] * coherence_score +
                weights["trait_alignment"] * trait_alignment_score
            )
            return overall_score
        return None

    def evaluate_single(
        self,
        gen_result: GenerationResult,
        format_type: str
    ) -> EvaluationResult:
        """
        Evaluate a single generation result.

        Args:
            gen_result: Generation result to evaluate
            format_type: Format type for prompts (txt, markdown, json)

        Returns:
            EvaluationResult with scores
        """
        print(f"  Evaluating: Layer={gen_result.layer}, Strength={gen_result.strength}, "
              f"Aggregation={gen_result.aggregation_method}")

        # Build prompts for all metrics
        prompts = self.metric_manager.build_all_prompts(
            format_type=format_type,
            question=gen_result.question,
            baseline_response=gen_result.baseline_response,
            steered_response=gen_result.steered_response
        )

        # Get judge responses
        judge_responses = self.judge.evaluate_batch(prompts)

        # Extract results from responses
        results = self._extract_results_from_responses(judge_responses)
        differentiation_score = results["differentiation_score"]
        coherence_score = results["coherence_score"]
        trait_alignment_score = results["trait_alignment_score"]
        open_traits = results["open_traits"]
        choose_result = results["choose_result"]

        # Extract explanations from responses
        explanations = self._extract_explanations_from_responses(judge_responses)

        # Calculate overall score
        overall_score = self._calculate_overall_score(
            differentiation_score,
            coherence_score,
            trait_alignment_score
        )

        return EvaluationResult(
            layer=gen_result.layer,
            strength=gen_result.strength,
            aggregation_method=gen_result.aggregation_method,
            question=gen_result.question,
            baseline_response=gen_result.baseline_response,
            steered_response=gen_result.steered_response,
            differentiation_score=differentiation_score,
            coherence_score=coherence_score,
            trait_alignment_score=trait_alignment_score,
            open_traits=open_traits,
            choose_result=choose_result,
            overall_score=overall_score,
            judge_responses=judge_responses,
            explanations=explanations
        )

    def evaluate_batch(
        self,
        gen_results: List[GenerationResult],
        format_type: str
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple generation results.

        Args:
            gen_results: List of generation results
            format_type: Format type for prompts

        Returns:
            List of evaluation results
        """
        evaluation_results = []
        for gen_result in gen_results:
            eval_result = self.evaluate_single(gen_result, format_type)
            evaluation_results.append(eval_result)

        # Sort by overall score (highest first)
        evaluation_results.sort(
            key=lambda x: x.overall_score if x.overall_score is not None else -float('inf'),
            reverse=True
        )

        return evaluation_results


class StatisticsAggregator:
    """Aggregates evaluation results into statistics."""

    @staticmethod
    def aggregate(eval_results: List[EvaluationResult]) -> List[ConfigStatistics]:
        """
        Aggregate evaluation results by configuration.

        Args:
            eval_results: List of evaluation results

        Returns:
            List of ConfigStatistics, sorted by average overall score
        """
        stats_by_config = defaultdict(lambda: {
            "scores": [],
            "differentiation_scores": [],
            "coherence_scores": [],
            "trait_alignment_scores": [],
            "choose_correct": 0,
            "choose_incorrect": 0,
            "choose_equal": 0,
            "choose_total": 0
        })

        # Aggregate by (layer, strength, aggregation)
        for result in eval_results:
            key = (result.layer, result.strength, result.aggregation_method)

            if result.overall_score is not None:
                stats_by_config[key]["scores"].append(result.overall_score)
            if result.differentiation_score is not None:
                stats_by_config[key]["differentiation_scores"].append(result.differentiation_score)
            if result.coherence_score is not None:
                stats_by_config[key]["coherence_scores"].append(result.coherence_score)
            if result.trait_alignment_score is not None:
                stats_by_config[key]["trait_alignment_scores"].append(result.trait_alignment_score)

            # Track choose results (B = steered, A = baseline)
            if result.choose_result is not None:
                stats_by_config[key]["choose_total"] += 1
                if result.choose_result == "B":
                    stats_by_config[key]["choose_correct"] += 1
                elif result.choose_result == "A":
                    stats_by_config[key]["choose_incorrect"] += 1
                elif result.choose_result == "EQUAL":
                    stats_by_config[key]["choose_equal"] += 1

        # Calculate averages
        config_statistics = []
        for (layer, strength, aggregation), stats in stats_by_config.items():
            avg_overall = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else None
            avg_diff = sum(stats["differentiation_scores"]) / len(stats["differentiation_scores"]) \
                if stats["differentiation_scores"] else None
            avg_coh = sum(stats["coherence_scores"]) / len(stats["coherence_scores"]) \
                if stats["coherence_scores"] else None
            avg_trait = sum(stats["trait_alignment_scores"]) / len(stats["trait_alignment_scores"]) \
                if stats["trait_alignment_scores"] else None

            config_statistics.append(ConfigStatistics(
                layer=layer,
                strength=strength,
                aggregation_method=aggregation,
                avg_overall_score=avg_overall,
                avg_differentiation_score=avg_diff,
                avg_coherence_score=avg_coh,
                avg_trait_alignment_score=avg_trait,
                choose_correct=stats["choose_correct"],
                choose_incorrect=stats["choose_incorrect"],
                choose_equal=stats["choose_equal"],
                choose_total=stats["choose_total"]
            ))

        # Sort by average overall score
        config_statistics.sort(
            key=lambda x: x.avg_overall_score if x.avg_overall_score is not None else -float('inf'),
            reverse=True
        )

        return config_statistics
