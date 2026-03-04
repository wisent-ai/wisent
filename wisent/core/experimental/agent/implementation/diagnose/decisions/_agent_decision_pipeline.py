"""Classifier decision pipeline mixin."""
from typing import List, Dict, Any, Optional
import time
from wisent.core.experimental.agent.diagnose._agent_decision_types import (
    ClassifierDecision, SingleClassifierDecision, ClassifierParams,
    QualityResult, QualityControlledResponse, SteeringParams)
from wisent.core.experimental.agent.diagnose.classifier_marketplace import (
    ClassifierMarketplace, ClassifierListing, ClassifierCreationEstimate)
from wisent.core.utils.config_tools.constants import SECONDS_PER_MINUTE, SEPARATOR_WIDTH_NARROW

class ClassifierPipelineMixin:
    """Mixin providing decision pipeline methods."""

    async def make_classifier_decisions(self,
                                      task_analysis: TaskAnalysis,
                                      quality_threshold: float,
                                      time_budget_minutes: float,
                                      max_classifiers: int = None) -> List[ClassifierDecision]:
        """
        Make decisions about which benchmark-specific classifiers to create or use.
        
        Args:
            task_analysis: Analysis with relevant benchmarks
            quality_threshold: Minimum quality score to accept existing classifiers
            time_budget_minutes: Maximum time budget for creating new classifiers
            max_classifiers: Maximum number of classifiers to use (None = no limit)
            
        Returns:
            List of classifier decisions for each benchmark
        """
        # Set up budget manager
        budget_manager = get_budget_manager()
        budget_manager.set_time_budget(time_budget_minutes)
        
        # Discover available classifiers
        await asyncio.sleep(0)  # Make this async-compatible
        available_classifiers = self.marketplace.discover_available_classifiers()
        
        decisions = []
        classifier_count = 0
        
        # Create one classifier per relevant benchmark
        for benchmark_info in task_analysis.relevant_benchmarks:
            if max_classifiers and classifier_count >= max_classifiers:
                print(f"   ⏹️ Reached maximum classifier limit ({max_classifiers})")
                break
                
            benchmark_name = benchmark_info['benchmark']
            print(f"\n   🔍 Analyzing classifier for benchmark: {benchmark_name}")
            
            # Look for existing benchmark-specific classifier
            existing_options = [c for c in available_classifiers if benchmark_name.lower() in c.path.lower()]
            best_existing = max(existing_options, key=lambda x: x.quality_score) if existing_options else None
            
            # Get creation estimate for this benchmark
            creation_estimate = self.marketplace.get_creation_estimate(benchmark_name)
            
            # Make decision based on multiple factors
            decision = self._evaluate_benchmark_classifier_options(
                benchmark_name=benchmark_name,
                best_existing=best_existing,
                creation_estimate=creation_estimate,
                quality_threshold=quality_threshold,
                budget_manager=budget_manager
            )
            
            decisions.append(decision)
            
            # Update budget and count
            if decision.action == "create_new":
                training_time_seconds = creation_estimate.estimated_training_time_minutes * SECONDS_PER_MINUTE
                budget_manager.get_budget(ResourceType.TIME).spend(training_time_seconds)
                classifier_count += 1
                remaining_minutes = budget_manager.get_budget(ResourceType.TIME).remaining_budget / SECONDS_PER_MINUTE
                print(f"      ⏱️ Remaining time budget: {remaining_minutes:.1f} minutes")
            elif decision.action == "use_existing":
                classifier_count += 1
            
            print(f"      ✅ Decision: {decision.action} - {decision.reasoning}")
        
        # Store decisions in history
        self.decision_history.extend(decisions)
        
        return decisions
    
    def _evaluate_benchmark_classifier_options(self,
                                          benchmark_name: str,
                                          best_existing: Optional[ClassifierListing],
                                          creation_estimate: ClassifierCreationEstimate,
                                          quality_threshold: float,
                                          budget_manager) -> ClassifierDecision:
        """Evaluate whether to use existing, create new, or skip a benchmark-specific classifier."""
        
        # Factor 1: Existing classifier quality
        existing_quality = best_existing.quality_score if best_existing else 0.0
        
        # Factor 2: Time constraints
        time_budget = budget_manager.get_budget(ResourceType.TIME)
        training_time_seconds = creation_estimate.estimated_training_time_minutes * SECONDS_PER_MINUTE
        can_afford_creation = time_budget.can_afford(training_time_seconds)
        
        # Factor 3: Expected benefit vs cost
        creation_benefit = creation_estimate.estimated_quality_score
        existing_benefit = existing_quality
        
        # Decision logic
        if best_existing and existing_quality >= quality_threshold:
            if existing_quality >= creation_benefit or not can_afford_creation:
                return ClassifierDecision(
                    benchmark_name=benchmark_name,
                    action="use_existing",
                    selected_classifier=best_existing,
                    reasoning=f"Existing classifier quality {existing_quality:.2f} meets threshold",
                    confidence=existing_quality
                )
        
        if can_afford_creation and creation_benefit > existing_benefit:
            return ClassifierDecision(
                benchmark_name=benchmark_name,
                action="create_new",
                creation_estimate=creation_estimate,
                reasoning=f"Creating new classifier (est. quality {creation_benefit:.2f} > existing {existing_benefit:.2f})",
                confidence=creation_estimate.confidence
            )
        
        if best_existing:
            return ClassifierDecision(
                benchmark_name=benchmark_name,
                action="use_existing",
                selected_classifier=best_existing,
                reasoning=f"Using existing despite low quality - time/budget constraints",
                confidence=existing_quality * self._penalty_multiplier
            )
        
        return ClassifierDecision(
            benchmark_name=benchmark_name,
            action="skip",
            reasoning="No suitable existing classifier and cannot create new within budget",
            confidence=0.0
        )
    
    async def execute_decisions(self, decisions: List[ClassifierDecision]) -> List[Dict[str, Any]]:
        """
        Execute the classifier decisions and return the final classifier configs.
        
        Args:
            decisions: List of decisions to execute
            
        Returns:
            List of classifier configurations ready for use
        """
        classifier_configs = []
        
        for decision in decisions:
            if decision.action == "skip":
                continue
                
            elif decision.action == "use_existing":
                config = decision.selected_classifier.to_config()
                classifier_configs.append(config)
                print(f"   📎 Using existing {decision.issue_type} classifier: {config['path']}")
                
            elif decision.action == "create_new":
                print(f"   🏗️ Creating new classifier for benchmark: {decision.benchmark_name}...")
                start_time = time.time()
                try:
                    # Create benchmark-specific classifier
                    new_classifier = await self._create_classifier_for_benchmark(
                        benchmark_name=decision.benchmark_name
                    )
                    
                    end_time = time.time()
                    
                    # Track performance for future budget estimates
                    track_task_performance(
                        task_name=f"classifier_training_{decision.benchmark_name}",
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    config = new_classifier.to_config()
                    config['benchmark'] = decision.benchmark_name
                    classifier_configs.append(config)
                    print(f"      ✅ Created: {config['path']} (took {end_time - start_time:.1f}s)")
                except Exception as e:
                    print(f"      ❌ Failed to create {decision.benchmark_name} classifier: {e}")
                    continue
        
        return classifier_configs
    
    async def _create_classifier_for_benchmark(self, benchmark_name: str):
        """
        Create a classifier trained specifically on a benchmark dataset.
        
        Args:
            benchmark_name: Name of the benchmark to train on
            
        Returns:
            Trained classifier instance
        """
        from .create_classifier import ClassifierCreator
        
        try:
            # Initialize classifier creator
            creator = ClassifierCreator(self.marketplace.model, max_tasks_to_process=self.max_tasks_to_process)
            
            # Create classifier using benchmark-specific training data
            print(f"      📊 Loading training data from benchmark: {benchmark_name}")
            classifier = await creator.create_classifier_for_issue_with_benchmarks(
                issue_type=benchmark_name,  # Use benchmark name as issue type
                relevant_benchmarks=[benchmark_name],
                num_samples=self._decision_num_samples
            )
            
            return classifier
            
        except Exception as e:
            print(f"      ⚠️ Benchmark-based creation failed: {e}")
            raise e
    
    def get_decision_summary(self) -> str:
        """Get a summary of recent classifier decisions."""
        if not self.decision_history:
            return "No classifier decisions made yet."
        
        recent_decisions = self.decision_history[-10:]  # Last 10 decisions
        
        summary = "\n🤖 Recent Classifier Decisions\n"
        summary += "=" * SEPARATOR_WIDTH_NARROW + "\n"
        
        action_counts = {}
        for decision in recent_decisions:
            action_counts[decision.action] = action_counts.get(decision.action, 0) + 1
        
        summary += f"Actions taken: {dict(action_counts)}\n\n"
        
        for decision in recent_decisions[-5:]:  # Show last 5
            summary += f"• {decision.benchmark_name}: {decision.action}\n"
            summary += f"  Reasoning: {decision.reasoning}\n"
            summary += f"  Confidence: {decision.confidence:.2f}\n\n"
        
        return summary
    
    async def smart_classifier_selection(self,
                                       prompt: str,
                                       quality_threshold: float,
                                       time_budget_minutes: float,
                                       context: str = "",
                                       max_classifiers: int = None) -> List[Dict[str, Any]]:
        """
        One-stop method for intelligent classifier selection.
        
        Args:
            prompt: The task/prompt to analyze
            context: Additional context
            quality_threshold: Minimum quality for existing classifiers
            time_budget_minutes: Time budget for creating new classifiers
            max_classifiers: Maximum number of classifiers to use
            
        Returns:
            List of classifier configurations ready for use
        """
        print(f"🧠 Smart classifier selection for task...")
        
        # Step 1: Analyze task requirements
        task_analysis = self.analyze_task_requirements(prompt, context)
        
        # Step 2: Make decisions about classifiers
        decisions = await self.make_classifier_decisions(
            task_analysis=task_analysis,
            quality_threshold=quality_threshold,
            time_budget_minutes=time_budget_minutes,
            max_classifiers=max_classifiers
        )
        
        # Step 3: Execute decisions
        classifier_configs = await self.execute_decisions(decisions)
        
        print(f"🎯 Selected {len(classifier_configs)} classifiers for the task")
