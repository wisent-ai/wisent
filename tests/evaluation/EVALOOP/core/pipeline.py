"""Main evaluation pipeline orchestrator."""
import json
import os
from pathlib import Path
from typing import List

from tests.EVALOOP.core.config import ConfigManager, TraitConfig, EvaluationConfig
from tests.EVALOOP.core.generator import GenerationPipeline
from tests.EVALOOP.core.evaluator import Evaluator, StatisticsAggregator
from tests.EVALOOP.core.models import GenerationResult, EvaluationResult, ConfigStatistics


class EvaluationPipeline:
    """Main orchestrator for the complete evaluation pipeline."""

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize pipeline.

        Args:
            config_manager: Configuration manager
        """
        self.config_manager = config_manager
        self.eval_config = config_manager.eval_config

    def run_generation_phase(self, trait_name: str) -> List[GenerationResult]:
        """
        Run generation phase for a trait.

        Args:
            trait_name: Name of the trait to generate for

        Returns:
            List of GenerationResult objects
        """
        print(f"\n{'#'*80}")
        print(f"# GENERATION PHASE: {trait_name.upper()}")
        print(f"{'#'*80}\n")

        trait_config = self.config_manager.get_trait(trait_name)

        # Create and run generation pipeline (pass config_manager for path management)
        gen_pipeline = GenerationPipeline(self.eval_config, self.config_manager)
        results = gen_pipeline.run_trait(trait_config)

        # Save generation results
        self._save_generation_results(results, trait_config.output_file)

        print(f"\nGeneration complete: {len(results)} results saved to {trait_config.output_file}")
        return results

    def run_evaluation_phase(
        self,
        trait_name: str,
        format_type: str,
        gen_results: List[GenerationResult] = None
    ) -> None:
        """
        Run evaluation phase for a trait with a single format.

        Args:
            trait_name: Name of the trait to evaluate
            format_type: Format type for prompts (txt, markdown, json)
            gen_results: Optional pre-loaded generation results. If None, loads from file.
        """
        print(f"\n{'='*60}")
        print(f"Processing {trait_name} with format: {format_type}")
        print(f"{'='*60}\n")

        trait_config = self.config_manager.get_trait(trait_name)

        # Load generation results if not provided
        if gen_results is None:
            gen_results = self._load_generation_results(trait_config.output_file)

        # Create evaluator
        evaluator = Evaluator(self.eval_config, trait_config)

        # Evaluate all results
        eval_results = evaluator.evaluate_batch(gen_results, format_type)

        # Aggregate statistics
        statistics = StatisticsAggregator.aggregate(eval_results)

        # Save results
        self._save_evaluation_results(
            eval_results,
            statistics,
            trait_name,
            format_type
        )

        print(f"\nEvaluation complete for {trait_name} with {format_type} format.")

    def run_evaluation_phase_all_formats(
        self,
        trait_name: str,
        gen_results: List[GenerationResult] = None
    ) -> None:
        """
        Run evaluation phase for a trait across all configured formats.

        Args:
            trait_name: Name of the trait to evaluate
            gen_results: Optional pre-loaded generation results. If None, loads from file.
        """
        print(f"\n{'#'*80}")
        print(f"# EVALUATION PHASE: {trait_name.upper()}")
        print(f"{'#'*80}\n")

        trait_config = self.config_manager.get_trait(trait_name)

        # Load generation results if not provided once
        if gen_results is None:
            gen_results = self._load_generation_results(trait_config.output_file)

        # Evaluate for each format type
        for format_type in self.eval_config.output_formats:
            self.run_evaluation_phase(trait_name, format_type, gen_results)

    def run_full_pipeline(self, trait_name: str) -> None:
        """
        Run complete pipeline (generation + evaluation) for a trait.

        Args:
            trait_name: Name of the trait
        """
        # Generation phase
        gen_results = self.run_generation_phase(trait_name)

        # Evaluation phase
        self.run_evaluation_phase_all_formats(trait_name, gen_results)

    def run_all_traits(self) -> None:
        """Run complete pipeline for all configured traits."""
        for trait_name in self.config_manager.list_traits():
            self.run_full_pipeline(trait_name)

    def _save_generation_results(
        self,
        results: List[GenerationResult],
        output_file: Path
    ) -> None:
        """Save generation results to JSON file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                [r.to_dict() for r in results],
                f,
                indent=2,
                ensure_ascii=False
            )

    def _load_generation_results(self, input_file: Path) -> List[GenerationResult]:
        """Load generation results from JSON file."""
        with open(input_file, encoding="utf-8") as f:
            data = json.load(f)

        results = []
        for item in data:
            results.append(GenerationResult(
                layer=item["layer"],
                strength=item["strength"],
                aggregation_method=item.get("aggregation method", item.get("aggregation_method")),
                question=item["question"],
                baseline_response=item.get("baseline_response", item.get("unsteered_response", "")),
                steered_response=item["steered_response"]
            ))
        return results

    def _save_evaluation_results(
        self,
        eval_results: List[EvaluationResult],
        statistics: List[ConfigStatistics],
        trait_name: str,
        format_type: str
    ) -> None:
        """Save evaluation results and statistics."""
        # Determine output paths using ConfigManager
        base_dir = self.config_manager.eval_config.base_dir / "output"
        scores_file = base_dir / f"{trait_name}_scores_{format_type}.json"
        stats_file = base_dir / f"{trait_name}_stats_{format_type}.json"

        # Save evaluation results
        with open(scores_file, "w", encoding="utf-8") as f:
            json.dump(
                [r.to_dict() for r in eval_results],
                f,
                indent=2,
                ensure_ascii=False
            )

        # Save statistics
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(
                [s.to_dict() for s in statistics],
                f,
                indent=2,
                ensure_ascii=False
            )

        print(f"  Results saved to {scores_file}")
        print(f"  Statistics saved to {stats_file}")
