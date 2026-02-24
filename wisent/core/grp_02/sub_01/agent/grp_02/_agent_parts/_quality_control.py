"""
Quality control response mixin for AutonomousAgent.

Extracted from autonomous_agent.py to comply with the 300-line file limit.
Contains the main quality-controlled response flow with timeout enforcement,
iterative improvement loop, and steered generation.
"""

import time
from wisent.core.constants import QUALITY_CONTROL_MAX_ATTEMPTS, CLASSIFIER_DECISION_THRESHOLD, DISPLAY_TRUNCATION_COMPACT


class QualityControlMixin:
    """Mixin providing quality-controlled response generation."""

    async def respond_with_quality_control(
        self, prompt: str, max_attempts: int = QUALITY_CONTROL_MAX_ATTEMPTS, time_budget_minutes: float = None
    ) -> "QualityControlledResponse":
        """
        Generate response with iterative quality control and adaptive steering.

        This method implements the complete quality control flow:
        1. Analyze prompt and determine classifier parameters
        2. Train single combined classifier on relevant benchmarks
        3. Generate initial response without steering
        4. Iteratively improve using model-determined steering until acceptable

        Args:
            prompt: The user prompt to respond to
            max_attempts: Maximum attempts to achieve acceptable quality
            time_budget_minutes: Time budget for classifier creation
        """
        from ..agent.diagnose.agent_classifier_decision import QualityControlledResponse
        from ..agent.timeout import TimeoutError, timeout_context

        start_time = time.time()
        time_budget = time_budget_minutes or self.default_time_budget_minutes

        print(f"\n QUALITY-CONTROLLED RESPONSE TO: {prompt[:DISPLAY_TRUNCATION_COMPACT]}...")
        print(f" Hard timeout enforced: {time_budget:.1f} minutes")

        try:
            async with timeout_context(time_budget) as timeout_mgr:
                return await self._respond_with_quality_control_impl(
                    prompt, max_attempts, time_budget, timeout_mgr, start_time
                )
        except TimeoutError as e:
            print(f" OPERATION TIMED OUT: {e}")
            print(f"   Elapsed: {e.elapsed_time:.1f}s / Budget: {e.budget_time:.1f}s")

            return QualityControlledResponse(
                response_text=f"[TIMEOUT] Operation exceeded {time_budget:.1f}min budget. Partial response may be available.",
                final_quality_score=0.0,
                attempts_needed=0,
                classifier_params_used=None,
                total_time_seconds=e.elapsed_time,
            )

    async def _respond_with_quality_control_impl(
        self, prompt: str, max_attempts: int, time_budget: float, timeout_mgr, start_time: float
    ) -> "QualityControlledResponse":
        """Implementation of quality control with timeout checking."""
        from ..agent.diagnose.agent_classifier_decision import QualityControlledResponse

        # Step 1: Analyze prompt and select relevant benchmarks
        print("\n Step 1: Analyzing task and selecting benchmarks...")
        timeout_mgr.check_timeout()

        task_analysis = self.decision_system.analyze_task_requirements(
            prompt,
            priority=self.priority,
            fast_only=self.fast_only,
            time_budget_minutes=self.time_budget_minutes or time_budget,
            max_benchmarks=self.max_benchmarks or 1,
        )

        timeout_mgr.check_timeout()
        benchmark_names = [b["benchmark"] for b in task_analysis.relevant_benchmarks]
        print(f"   Selected benchmarks: {benchmark_names}")
        print(f"   Remaining time: {timeout_mgr.get_remaining_time():.1f}s")

        # Step 2: Determine optimal classifier parameters (with memory)
        print("\n Step 2: Determining optimal classifier parameters...")
        timeout_mgr.check_timeout()

        classifier_params = await self._get_or_determine_classifier_parameters(prompt, benchmark_names)
        print(
            f"   Parameters: Layer {classifier_params.optimal_layer}, "
            f"Threshold {classifier_params.classification_threshold}, "
            f"{classifier_params.training_samples} samples, "
            f"{classifier_params.classifier_type} classifier"
        )
        print(f"   Reasoning: {classifier_params.reasoning}")
        print(f"   Remaining time: {timeout_mgr.get_remaining_time():.1f}s")

        # Step 3: Create single combined classifier
        print("\n Step 3: Training combined classifier...")
        timeout_mgr.check_timeout()

        remaining_minutes = timeout_mgr.get_remaining_time() / 60.0
        classifier_time_budget = min(time_budget, remaining_minutes)

        classifier_decision = await self.decision_system.create_single_quality_classifier(
            task_analysis, classifier_params, time_budget_minutes=classifier_time_budget
        )

        if classifier_decision.action == "skip":
            print(f"   Skipping classifier creation: {classifier_decision.reasoning}")
            response = await self._generate_response(prompt)
            return QualityControlledResponse(
                response_text=response,
                final_quality_score=CLASSIFIER_DECISION_THRESHOLD,
                attempts_needed=1,
                classifier_params_used=classifier_params,
                total_time_seconds=time.time() - start_time,
            )

        classifier = await self.decision_system.execute_single_classifier_decision(
            classifier_decision, classifier_params
        )
        print(f"   Remaining time: {timeout_mgr.get_remaining_time():.1f}s")

        if classifier is None:
            print("   Failed to create classifier, falling back to basic generation")
            response = await self._generate_response(prompt)
            return QualityControlledResponse(
                response_text=response,
                final_quality_score=CLASSIFIER_DECISION_THRESHOLD,
                attempts_needed=1,
                classifier_params_used=classifier_params,
                total_time_seconds=time.time() - start_time,
            )

        # Step 4: Generate initial response (no steering)
        print("\n Step 4: Generating initial response...")
        timeout_mgr.check_timeout()

        current_response = await self._generate_response(prompt)
        print(f"   Initial response: {current_response[:DISPLAY_TRUNCATION_COMPACT]}...")
        print(f"   Remaining time: {timeout_mgr.get_remaining_time():.1f}s")

        # Step 5: Iterative quality improvement loop
        print("\n Step 5: Quality improvement loop...")
        quality_progression = []
        steering_params_used = None

        for attempt in range(1, max_attempts + 1):
            print(f"\n--- Attempt {attempt}/{max_attempts} ---")
            timeout_mgr.check_timeout()

            if timeout_mgr.get_remaining_time() <= 0:
                print("   TIME UP! Breaking immediately.")
                break

            quality_result = await self.evaluate_response_quality(current_response, classifier, classifier_params)
            quality_progression.append(quality_result.score)

            print(f"   Quality score: {quality_result.score:.3f}")
            print(f"   Model judgment: {quality_result.reasoning}")

            if quality_result.acceptable:
                print("   Quality acceptable! Stopping improvement loop.")
                break

            if attempt >= max_attempts:
                print("   Maximum attempts reached. Using current response.")
                break

            print("   Determining steering parameters...")
            steering_params = await self._get_or_determine_steering_parameters(prompt, quality_result.score, attempt)
            steering_params_used = steering_params

            print(f"   Steering: {steering_params.steering_method} (strength {steering_params.initial_strength})")
            print(f"   Reasoning: {steering_params.reasoning}")

            print("   Applying steering and regenerating...")
            try:
                steered_response = await self._generate_with_steering(prompt, steering_params)
                current_response = steered_response
                print(f"   New response: {current_response[:DISPLAY_TRUNCATION_COMPACT]}...")
            except Exception as e:
                print(f"   Warning: Steering failed: {e}")
                print("   Keeping previous response")
                break

        # Final quality evaluation
        print("\n Final quality evaluation...")
        timeout_mgr.check_timeout()

        final_quality = await self.evaluate_response_quality(current_response, classifier, classifier_params)
        total_time = time.time() - start_time

        result = QualityControlledResponse(
            response_text=current_response,
            final_quality_score=final_quality.score,
            attempts_needed=len(quality_progression),
            classifier_params_used=classifier_params,
            steering_params_used=steering_params_used,
            quality_progression=quality_progression,
            total_time_seconds=total_time,
        )

        if final_quality.acceptable:
            self._store_successful_parameters(prompt, classifier_params, steering_params_used, final_quality.score)

        print("\n QUALITY CONTROL COMPLETE")
        print(f"   Final response: {result.response_text[:DISPLAY_TRUNCATION_COMPACT]}...")
        print(f"   Final quality: {result.final_quality_score:.3f}")
        print(f"   Attempts: {result.attempts_needed}")
        print(f"   Total time: {result.total_time_seconds:.1f}s")
        print(f"   Time used: {timeout_mgr.get_elapsed_time():.1f}s / {time_budget * 60:.1f}s")

        return result

    async def _generate_with_steering(self, prompt: str, steering_params: "SteeringParams") -> str:
        """Generate response with specified steering parameters."""
        print(
            f"      Applying {steering_params.steering_method} steering with strength {steering_params.initial_strength}"
        )

        original_method = self.steering_method
        original_strength = self.steering_strength
        original_mode = self.steering_mode

        try:
            self.steering_method = steering_params.steering_method
            self.steering_strength = steering_params.initial_strength
            self.steering_mode = True

            if steering_params.method_specific_params:
                for param, value in steering_params.method_specific_params.items():
                    if hasattr(self, param):
                        setattr(self, param, value)

            response = await self._generate_response(prompt)
            return response

        finally:
            self.steering_method = original_method
            self.steering_strength = original_strength
            self.steering_mode = original_mode
