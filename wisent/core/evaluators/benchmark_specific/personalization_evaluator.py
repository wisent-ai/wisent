"""Personalization evaluator for assessing personality trait manifestation in model responses.

Evaluates how well model-generated responses exhibit specified personality traits
by using an LLM judge to assess trait alignment and intensity.
"""

from typing import Any, Optional
import logging
import re

from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult

logger = logging.getLogger(__name__)


class PersonalizationEvaluator(BaseEvaluator):
    """Evaluator for personality trait manifestation in responses.

    Uses an LLM judge to evaluate how well responses exhibit target personality traits.
    Compatible with personalization benchmarks that require trait assessment.
    """

    name = "personalization"
    description = "Evaluates personality trait manifestation in model responses"
    task_names = ("personalization",)

    # Default judge model - can be overridden
    default_judge_model = "gpt-4"

    # Evaluation prompt template
    JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator analyzing personality traits in text responses.

**Target Trait**: {trait}
**Trait Description**: {trait_description}

**Prompt Given to Model**: {prompt}

**Model Response to Evaluate**:
{response}

Please evaluate how well this response exhibits the target trait "{trait}".

Provide your evaluation in the following format:

TRAIT_SCORE: [0-10]
INTENSITY: [weak/moderate/strong]
EXPLANATION: [Brief explanation of your rating]

Where TRAIT_SCORE represents:
- 0-2: Trait not present or contradicts trait
- 3-4: Minimal trait manifestation
- 5-6: Moderate trait manifestation
- 7-8: Strong trait manifestation
- 9-10: Exemplary trait manifestation

Evaluate ONLY based on how well the response demonstrates the target trait, not overall quality."""

    def __init__(self, judge_model: Optional[str] = None):
        """Initialize personalization evaluator.

        Args:
            judge_model: Model to use as judge (default: gpt-4)
        """
        self.judge_model = judge_model or self.default_judge_model

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate personality trait manifestation in response.

        Args:
            response: Generated model response to evaluate
            expected: Expected trait information (dict with 'trait', 'trait_description')
            **kwargs:
                prompt: Original prompt given to model
                judge_model: Override default judge model
                use_mock: Use mock evaluation for testing (default: False)

        Returns:
            EvalResult with trait_score as confidence, details containing analysis
        """
        # Extract trait information
        if isinstance(expected, dict):
            trait = expected.get('trait', 'unknown')
            trait_description = expected.get('trait_description', '')
        else:
            trait = str(expected)
            trait_description = f"The trait: {trait}"

        prompt = kwargs.get('prompt', '')
        judge_model = kwargs.get('judge_model', self.judge_model)
        use_mock = kwargs.get('use_mock', False)

        # Mock evaluation for testing without API calls
        if use_mock:
            return self._mock_evaluate(response, trait, prompt)

        # Prepare judge prompt
        judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            trait=trait,
            trait_description=trait_description,
            prompt=prompt,
            response=response
        )

        try:
            # Call LLM judge
            judge_response = self._call_judge(judge_prompt, judge_model)

            # Parse judge response
            trait_score, intensity, explanation = self._parse_judge_response(judge_response)

            # Normalize score to 0-1 confidence
            confidence = trait_score / 10.0

            # Determine ground truth based on score
            if trait_score >= 7:
                ground_truth = "TRUTHFUL"  # Strong trait manifestation
            elif trait_score >= 4:
                ground_truth = "UNKNOWN"  # Moderate trait manifestation
            else:
                ground_truth = "UNTRUTHFUL"  # Weak/absent trait manifestation

            return EvalResult(
                ground_truth=ground_truth,
                method_used=self.name,
                confidence=confidence,
                details=f"Trait: {trait} | Score: {trait_score}/10 | Intensity: {intensity} | {explanation}",
                meta={
                    "trait": trait,
                    "trait_score": trait_score,
                    "intensity": intensity,
                    "judge_model": judge_model,
                    "explanation": explanation
                }
            )

        except Exception as e:
            logger.exception(f"Error in personalization evaluation: {e}")
            return EvalResult(
                ground_truth="UNKNOWN",
                method_used=self.name,
                confidence=0.0,
                details=f"Evaluation error: {str(e)}",
                meta={"trait": trait, "error": str(e)}
            )

    def _mock_evaluate(self, response: str, trait: str, prompt: str) -> EvalResult:
        """Mock evaluation for testing without API calls.

        Args:
            response: Model response
            trait: Target trait
            prompt: Original prompt

        Returns:
            EvalResult with mock scores
        """
        # Simple heuristic: check if trait appears in response
        response_lower = response.lower()
        trait_lower = trait.lower()

        # Calculate mock score based on simple heuristics
        score = 5  # baseline

        if trait_lower in response_lower:
            score += 2

        if len(response) > 100:
            score += 1

        if len(response.split('.')) > 2:
            score += 1

        score = min(score, 10)

        intensity = "weak" if score < 5 else ("moderate" if score < 7 else "strong")

        confidence = score / 10.0
        ground_truth = "TRUTHFUL" if score >= 7 else ("UNKNOWN" if score >= 4 else "UNTRUTHFUL")

        return EvalResult(
            ground_truth=ground_truth,
            method_used=f"{self.name}_mock",
            confidence=confidence,
            details=f"Mock evaluation | Trait: {trait} | Score: {score}/10 | Intensity: {intensity}",
            meta={
                "trait": trait,
                "trait_score": score,
                "intensity": intensity,
                "judge_model": "mock",
                "explanation": "Mock evaluation based on simple heuristics"
            }
        )

    def _call_judge(self, prompt: str, model: str) -> str:
        """Call LLM judge to evaluate response.

        Args:
            prompt: Judge prompt
            model: Model identifier

        Returns:
            Judge response text
        """
        # This would call OpenAI API or other LLM API
        # For now, raising NotImplementedError to indicate it needs implementation
        try:
            import openai

            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of personality traits in text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=500
            )

            return response.choices[0].message.content

        except ImportError:
            raise NotImplementedError(
                "OpenAI package not installed. Install with: pip install openai\n"
                "Or use use_mock=True for testing without API calls."
            )
        except Exception as e:
            raise RuntimeError(f"Error calling judge model: {e}")

    def _parse_judge_response(self, judge_response: str) -> tuple[float, str, str]:
        """Parse judge response to extract score, intensity, and explanation.

        Args:
            judge_response: Raw judge response text

        Returns:
            Tuple of (trait_score, intensity, explanation)
        """
        # Extract TRAIT_SCORE
        score_match = re.search(r'TRAIT_SCORE:\s*(\d+(?:\.\d+)?)', judge_response, re.IGNORECASE)
        trait_score = float(score_match.group(1)) if score_match else 5.0

        # Extract INTENSITY
        intensity_match = re.search(r'INTENSITY:\s*(weak|moderate|strong)', judge_response, re.IGNORECASE)
        intensity = intensity_match.group(1).lower() if intensity_match else "moderate"

        # Extract EXPLANATION
        explanation_match = re.search(r'EXPLANATION:\s*(.+?)(?=\n\n|\Z)', judge_response, re.IGNORECASE | re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"

        return trait_score, intensity, explanation
