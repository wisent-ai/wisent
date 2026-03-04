"""Autonomous Wisent Agent - steering and correction techniques."""
import asyncio
import sys
from typing import Any, Dict, List, Optional

from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy
from wisent.core.primitives.model_interface.core.activations.activations import Activations
from wisent.core.primitives.models import get_generate_kwargs
from wisent.core.utils.infra_tools.errors import MissingParameterError
from wisent.core import constants as _C
from .agent.diagnose import AgentClassifierDecisionSystem, AnalysisResult, ClassifierMarketplace, ResponseDiagnostics
from .agent.steer import ImprovementResult, ResponseSteering
from .model import Model

from ._agent_parts import QualityEvaluationMixin, SteeringParamsMixin, QualityControlMixin


class AutonomousAgent(QualityEvaluationMixin, SteeringParamsMixin, QualityControlMixin):
    """
    An autonomous agent that can generate responses, analyze them for issues,
    and improve them using activation-based steering and correction techniques.

    Uses a marketplace-based system to intelligently select classifiers
    based on task analysis, with no hardcoded requirements.
    """

    def __init__(
        self,
        model_name: str,
        steering_threshold: float = None,
        steering_strength: Optional[float] = None,
        layer_override: int = None,
        enable_tracking: bool = True,
        steering_method: Optional[str] = None,
        steering_mode: bool = False,
        normalization_method: Optional[str] = None,
        target_norm: Optional[float] = None,
        priority: Optional[str] = None,
        fast_only: bool = False,
        time_budget_minutes: float = None,
        max_benchmarks: int = None,
        smart_selection: bool = False,
    ):
        """Initialize the autonomous agent."""
        if steering_threshold is None:
            raise ValueError("steering_threshold is required")
        self._steering_threshold = steering_threshold
        self.model_name = model_name
        self.model: Optional[Model] = None
        self.layer_override = layer_override
        self.enable_tracking = enable_tracking

        from .parameters import load_model_parameters

        self.params = load_model_parameters(model_name, layer_override)

        self.steering_method = steering_method
        self.steering_strength = steering_strength
        self.steering_mode = steering_mode
        self.normalization_method = normalization_method
        self.target_norm = target_norm

        self.priority = priority
        self.fast_only = fast_only
        self.time_budget_minutes = time_budget_minutes
        self.max_benchmarks = max_benchmarks
        self.smart_selection = smart_selection

        self.marketplace: Optional[ClassifierMarketplace] = None
        self.decision_system: Optional[AgentClassifierDecisionSystem] = None
        self.diagnostics: Optional[ResponseDiagnostics] = None
        self.steering: Optional[ResponseSteering] = None

        self.improvement_history: List[ImprovementResult] = []
        self.analysis_history: List[AnalysisResult] = []

        print(f"Autonomous Agent initialized with {model_name}")
        print("   Using marketplace-based classifier selection")
        print(f"   Steering: {steering_method} (strength: {steering_strength})")
        if steering_mode:
            print(f"   Steering mode enabled with {normalization_method} normalization")
        print(self.params.get_summary())

    async def initialize(
        self,
        quality_threshold: float,
        default_time_budget_minutes: float,
        classifier_search_paths: Optional[List[str]] = None,
    ):
        """Initialize the agent with intelligent classifier management."""
        print("Initializing Autonomous Agent...")
        print("   Loading model...")
        self.model = Model(self.model_name)

        print("   Setting up classifier marketplace...")
        self.marketplace = ClassifierMarketplace(model=self.model, search_paths=classifier_search_paths)

        print("   Setting up intelligent decision system...")
        self.decision_system = AgentClassifierDecisionSystem(self.marketplace)

        self.quality_threshold = quality_threshold
        self.default_time_budget_minutes = default_time_budget_minutes

        summary = self.marketplace.get_marketplace_summary()
        print(summary)
        print("   Autonomous Agent ready!")

    async def respond_autonomously(
        self,
        prompt: str,
        max_attempts: int,
        quality_threshold: float = None,
        time_budget_minutes: float = None,
        max_classifiers: int = None,
    ) -> Dict[str, Any]:
        """Generate a response and autonomously improve it if needed."""
        print(f"\nAUTONOMOUS RESPONSE TO: {prompt[:_C.DISPLAY_TRUNCATION_COMPACT]}...")

        quality_threshold = quality_threshold or self.quality_threshold
        time_budget_minutes = time_budget_minutes or self.default_time_budget_minutes

        print("\nAnalyzing task and selecting classifiers...")
        classifier_configs = await self.decision_system.smart_classifier_selection(
            prompt=prompt,
            quality_threshold=quality_threshold,
            time_budget_minutes=time_budget_minutes,
            max_classifiers=max_classifiers,
        )

        if classifier_configs:
            print(f"   Initializing diagnostics with {len(classifier_configs)} classifiers")
            self.diagnostics = ResponseDiagnostics(model=self.model, classifier_configs=classifier_configs)
            self.steering = ResponseSteering(
                generate_response_func=self._generate_response,
                analyze_response_func=self.diagnostics.analyze_response,
            )
        else:
            print("   No classifiers selected - proceeding without diagnostics")
            return {
                "final_response": await self._generate_response(prompt),
                "attempts": 1,
                "improvement_chain": [],
                "classifier_info": "No classifiers used",
            }

        attempt = 0
        current_response = None
        improvement_chain = []

        while attempt < max_attempts:
            attempt += 1
            print(f"\n--- Attempt {attempt} ---")

            if current_response is None:
                print("Generating initial response...")
                current_response = await self._generate_response(prompt)
                print(f"   Response: {current_response[:_C.DISPLAY_TRUNCATION_COMPACT]}...")

            print("Analyzing response...")
            analysis = await self.diagnostics.analyze_response(current_response, prompt)
            print(f"   Issues found: {analysis.issues_found}")
            print(f"   Quality score: {analysis.quality_score:.2f}")
            print(f"   Confidence: {analysis.confidence:.2f}")

            if self.enable_tracking:
                self.analysis_history.append(analysis)

            needs_improvement = self._decide_if_improvement_needed(analysis)
            if not needs_improvement:
                print("Response quality acceptable, no improvement needed")
                break

            print("Attempting to improve response...")
            improvement = await self.steering.improve_response(prompt, current_response, analysis)

            if improvement.success:
                print(f"   Improvement successful! Score: {improvement.improvement_score:.2f}")
                current_response = improvement.improved_response
                improvement_chain.append(improvement)
                if self.enable_tracking:
                    self.improvement_history.append(improvement)
            else:
                print("   Improvement failed, keeping original response")
                break

        return {
            "final_response": current_response,
            "attempts": attempt,
            "improvement_chain": improvement_chain,
            "final_analysis": analysis,
            "classifier_info": {
                "count": len(classifier_configs),
                "types": [c.get("issue_type", "unknown") for c in classifier_configs],
                "decision_summary": self.decision_system.get_decision_summary(),
            },
        }

    async def _generate_response(self, prompt: str) -> str:
        """Generate a response to the prompt with optional steering."""
        if self.steering_mode:
            print(f"   Applying {self.steering_method} steering...")
            try:
                from ..inference import generate_with_classification_and_handling

                steering_method = self._create_steering_method()
                gen_kwargs = get_generate_kwargs()
                response, _, _, _ = generate_with_classification_and_handling(
                    self.model, prompt, self.params.layer, **gen_kwargs,
                    steering_method=steering_method, token_aggregation="average",
                    threshold=self._steering_threshold, verbose=False, detection_handler=None,
                )
                return response
            except Exception as e:
                print(f"   Steering failed, falling back to basic generation: {e}")

        gen_kwargs = get_generate_kwargs()
        result = self.model.generate(prompt, self.params.layer, **gen_kwargs)
        if isinstance(result, tuple) and len(result) == 3:
            response, _, _ = result
        elif isinstance(result, tuple) and len(result) == 2:
            response, _ = result
        else:
            response = result
        return response

    def _create_steering_method(self):
        """Create a steering method object based on configuration."""
        from .steering_methods import CAA
        return CAA(device=None)

    def _decide_if_improvement_needed(self, analysis: AnalysisResult) -> bool:
        """Decide if the response needs improvement based on analysis."""
        if analysis.issues_found and analysis.confidence > self._steering_threshold:
            return True
        if analysis.quality_score < self.quality_threshold and analysis.confidence > self._steering_threshold:
            return True
        return False

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's performance."""
        if not self.enable_tracking:
            return {"tracking_disabled": True}
        total_improvements = len(self.improvement_history)
        successful_improvements = sum(1 for imp in self.improvement_history if imp.success)
        success_rate = successful_improvements / total_improvements if total_improvements > 0 else 0
        return {
            "total_improvements_attempted": total_improvements,
            "successful_improvements": successful_improvements,
            "success_rate": success_rate,
            "total_analyses": len(self.analysis_history),
            "decision_summary": self.decision_system.get_decision_summary()
            if self.decision_system else "No decisions made",
        }

    def get_marketplace_info(self) -> str:
        """Get information about the classifier marketplace."""
        if not self.marketplace:
            return "Marketplace not initialized"
        return self.marketplace.get_marketplace_summary()


async def demo_autonomous_agent(model_name: str, agent_retry_attempts: int, steering_threshold: float = None, quality_threshold: float = None, default_time_budget: float = None, demo_time_budget: float = None):
    """Demo function for the autonomous agent."""
    for _n, _v in [("steering_threshold", steering_threshold), ("quality_threshold", quality_threshold), ("default_time_budget", default_time_budget), ("demo_time_budget", demo_time_budget)]:
        if _v is None:
            raise ValueError(f"{_n} is required for demo_autonomous_agent")
    print("AUTONOMOUS AGENT DEMO")
    print("=" * _C.SEPARATOR_WIDTH_STANDARD)
    agent = AutonomousAgent(model_name=model_name, steering_threshold=steering_threshold)
    try:
        await agent.initialize(quality_threshold=quality_threshold, default_time_budget_minutes=default_time_budget)
        test_prompts = ["Tell me about the history of the moon landing", "What's the best way to lose weight quickly?", "Are men naturally better at math than women?", "Write a poem about spring flowers", "Explain quantum physics in simple terms"]
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{'=' * _C.SEPARATOR_PAD_SMALL} Test {i} {'=' * _C.SEPARATOR_PAD_SMALL}")
            result = await agent.respond_autonomously(
                prompt=prompt, max_attempts=agent_retry_attempts, time_budget_minutes=demo_time_budget,
            )
            print("\nRESULT SUMMARY:")
            print(f"   Final Response: {result['final_response'][:_C.DISPLAY_TRUNCATION_COMPACT]}...")
            print(f"   Attempts: {result['attempts']}")
            print(f"   Improvements: {len(result['improvement_chain'])}")
            print(f"   Classifiers Used: {result['classifier_info']['count']}")
            print(f"   Classifier Types: {result['classifier_info']['types']}")
        print("\nOVERALL PERFORMANCE:")
        summary = agent.get_performance_summary()
        print(f"   Total Improvements: {summary.get('total_improvements_attempted', 0)}")
        print(f"   Success Rate: {summary.get('success_rate', 0):.2%}")
    except Exception as e:
        print(f"Demo failed: {e}. This is expected if no classifiers are available.")

if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser()
    for _nm, _tp in [("--model", str), ("--agent-retry-attempts", int), ("--steering-threshold", float), ("--quality-threshold", float), ("--default-time-budget", float), ("--demo-time-budget", float)]:
        _p.add_argument(_nm, type=_tp, required=True)
    _a = _p.parse_args()
    asyncio.run(demo_autonomous_agent(model_name=_a.model, agent_retry_attempts=_a.agent_retry_attempts, steering_threshold=_a.steering_threshold, quality_threshold=_a.quality_threshold, default_time_budget=_a.default_time_budget, demo_time_budget=_a.demo_time_budget))
