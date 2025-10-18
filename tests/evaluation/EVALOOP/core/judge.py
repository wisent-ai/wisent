"""LLM judge for evaluating responses."""
from typing import Dict, Optional
from anthropic import Anthropic


class LLMJudge:
    """Wrapper for LLM-based evaluation."""

    def __init__(self, model: str = "claude-sonnet-4-5", max_tokens: int = 512):
        """
        Initialize LLM judge.

        Args:
            model: Model name for Anthropic API
            max_tokens: Maximum tokens for judge responses
        """
        self.model = model
        self.max_tokens = max_tokens
        self.client = Anthropic()  # Reads ANTHROPIC_API_KEY from environment

    def evaluate(self, prompt: str) -> str:
        """
        Evaluate using the LLM judge.

        Args:
            prompt: Evaluation prompt

        Returns:
            Judge response as string

        Raises:
            Exception: If API call fails
        """
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            print(f"Error calling judge API: {e}")
            raise

    def evaluate_batch(self, prompts: Dict[str, str]) -> Dict[str, str]:
        """
        Evaluate multiple prompts.

        Args:
            prompts: Dictionary mapping metric names to prompts

        Returns:
            Dictionary mapping metric names to responses
        """
        responses = {}
        for metric, prompt in prompts.items():
            try:
                responses[metric] = self.evaluate(prompt)
            except Exception as e:
                print(f"Error evaluating {metric}: {e}")
                responses[metric] = ""
        return responses
