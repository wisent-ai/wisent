"""Classifier creation and execution mixin."""
from typing import List, Dict, Any, Optional
import time
from wisent.core.agent.diagnose._agent_decision_types import (
    SingleClassifierDecision, ClassifierDecision, ClassifierParams)

class ClassifierCreationMixin:
    """Mixin providing classifier creation and execution methods."""

    async def create_single_quality_classifier(self, 
                                          task_analysis: TaskAnalysis,
                                          classifier_params: 'ClassifierParams',
                                          quality_threshold: float = 0.3,
                                          time_budget_minutes: float = 10.0) -> SingleClassifierDecision:
        """
        Create a single classifier trained on one benchmark.
        
        Args:
            task_analysis: Analysis with relevant benchmarks
            classifier_params: Model-determined classifier parameters
            quality_threshold: Minimum quality score to accept existing classifiers
            time_budget_minutes: Maximum time budget for creating new classifiers
            
        Returns:
            Single classifier decision for the selected benchmark
        """
        print(f"🔍 Creating single quality classifier from {len(task_analysis.relevant_benchmarks)} benchmark(s)...")
        
        # Extract benchmark names (should be just one now)
        benchmark_names = [b['benchmark'] for b in task_analysis.relevant_benchmarks]
        
        if not benchmark_names:
            return SingleClassifierDecision(
                benchmark_names=[],
                action="skip",
                reasoning="No benchmarks selected for classifier training",
                confidence=0.0
            )
        
        # Use first (and should be only) benchmark
        benchmark_name = benchmark_names[0]
        print(f"   📊 Using benchmark: {benchmark_name}")
        
        # Set up budget manager
        budget_manager = get_budget_manager()
        budget_manager.set_time_budget(time_budget_minutes)
        
        # Look for existing classifier for this exact model/layer/benchmark combination
        available_classifiers = self.marketplace.discover_available_classifiers()
        model_name = classifier_params.model_name if hasattr(classifier_params, 'model_name') else "unknown"
        layer = classifier_params.optimal_layer
        
        # Create specific classifier identifier
        classifier_id = f"{model_name}_{benchmark_name}_layer_{layer}"
        
        print(f"   🔍 Checking for existing classifier: {classifier_id}")
        
        # Find existing classifier with exact match
        existing_classifier = None
        for classifier in available_classifiers:
            # Check if classifier matches our exact requirements
            if (benchmark_name.lower() in classifier.path.lower() and
                str(layer) in classifier.path and
                classifier.layer == layer):
                existing_classifier = classifier
                print(f"   ✅ Found existing classifier: {classifier.path}")
                break
        
        # Decision logic for single benchmark classifier
        if existing_classifier and existing_classifier.quality_score >= quality_threshold:
            return SingleClassifierDecision(
                benchmark_names=[benchmark_name],
                action="use_existing",
                selected_classifier=existing_classifier,
                reasoning=f"Found existing classifier for {benchmark_name} at layer {layer} with quality {existing_classifier.quality_score:.2f}",
                confidence=existing_classifier.quality_score
            )
        
        # Get creation estimate for single benchmark classifier
        creation_estimate = self.marketplace.get_creation_estimate(benchmark_name)
        
        # Check if we can afford to create new classifier
        training_time_seconds = creation_estimate.estimated_training_time_minutes * 60
        time_budget = budget_manager.get_budget(ResourceType.TIME)
        
        if time_budget.can_afford(training_time_seconds):
            return SingleClassifierDecision(
                benchmark_names=[benchmark_name],
                action="create_new",
                creation_estimate=creation_estimate,
                reasoning=f"Creating new classifier for {benchmark_name} at layer {layer}",
                confidence=creation_estimate.confidence
            )
        else:
            return SingleClassifierDecision(
                benchmark_names=[benchmark_name],
                action="skip",
                reasoning=f"Insufficient time budget for creation (need {creation_estimate.estimated_training_time_minutes:.1f}min)",
                confidence=0.0
            )

    async def execute_single_classifier_decision(self, decision: SingleClassifierDecision, classifier_params: 'ClassifierParams') -> Optional[Any]:
        """
        Execute the single classifier decision to create or use the benchmark classifier.
        
        Args:
            decision: The single classifier decision to execute
            classifier_params: Model-determined classifier parameters
            
        Returns:
            The trained classifier instance or None if skipped
        """
        if decision.action == "skip":
            print(f"   ⏹️ Skipping classifier creation: {decision.reasoning}")
            return None
            
        elif decision.action == "use_existing":
            print(f"   📦 Using existing classifier: {decision.selected_classifier.path}")
            print(f"      Quality: {decision.selected_classifier.quality_score:.3f}")
            print(f"      Layer: {decision.selected_classifier.layer}")
            return decision.selected_classifier
            
        elif decision.action == "create_new":
            benchmark_name = decision.benchmark_names[0] if decision.benchmark_names else "unknown"
            print(f"   🏗️ Creating new classifier for benchmark: {benchmark_name}")
            start_time = time.time()
            try:
                # Create classifier using single benchmark training data
                new_classifier = await self._create_single_benchmark_classifier(
                    benchmark_name=benchmark_name,
                    classifier_params=classifier_params
                )
                
                creation_time = time.time() - start_time
                print(f"      ✅ Classifier created successfully in {creation_time:.1f}s")
                return new_classifier
                
            except Exception as e:
                print(f"      ❌ Failed to create classifier: {e}")
                return None
                
        return None

    async def _create_single_benchmark_classifier(self, benchmark_name: str, classifier_params: 'ClassifierParams') -> Optional[Any]:
        """
        Create a classifier for a single benchmark.
        
        Args:
            benchmark_name: Name of the benchmark to use for training
            classifier_params: Model-determined classifier parameters
            
        Returns:
            The trained classifier instance or None if failed
        """
        from .create_classifier import ClassifierCreator
        from ...training_config import TrainingConfig
        
        try:
            # Create training config
            config = TrainingConfig(
                issue_type=f"quality_{benchmark_name}",
                layer=classifier_params.optimal_layer,
                classifier_type=classifier_params.classifier_type,
                threshold=classifier_params.classification_threshold,
                training_samples=classifier_params.training_samples,
                model_name=self.marketplace.model.name if self.marketplace.model else "unknown"
            )
            
            # Create classifier creator
            creator = ClassifierCreator(self.marketplace.model)
            
            # Create classifier using benchmark-specific training data
            result = await creator.create_classifier_for_issue_with_benchmarks(
                issue_type=f"quality_{benchmark_name}",
                relevant_benchmarks=[benchmark_name],
                layer=classifier_params.optimal_layer,
                num_samples=classifier_params.training_samples,
                config=config
            )
            
            return result.classifier if result else None
            
        except Exception as e:
            print(f"      ❌ Error in single benchmark classifier creation: {e}")
            raise

    async def _create_combined_classifier(self, benchmark_names: List[str], classifier_params: 'ClassifierParams'):
        """
        Create a classifier using combined training data from multiple benchmarks.
        
        Args:
            benchmark_names: List of benchmark names to combine
            classifier_params: Model-determined parameters for classifier creation
            
        Returns:
            Trained classifier instance
        """
        from .create_classifier import ClassifierCreator
        
        try:
            # Initialize classifier creator
            creator = ClassifierCreator(self.marketplace.model)
            
            # Create classifier using combined benchmark training data
            print(f"      📊 Loading combined training data from benchmarks: {benchmark_names}")
            classifier = await creator.create_combined_benchmark_classifier(
                benchmark_names=benchmark_names,
                classifier_params=classifier_params
            )
            
            return classifier
            
        except Exception as e:
            print(f"      ❌ Error in combined classifier creation: {e}")
            raise

