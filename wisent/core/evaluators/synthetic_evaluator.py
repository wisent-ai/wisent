"""Synthetic evaluator for free-form trait descriptions.

Uses LLM-as-judge to evaluate how well responses match a given trait.
Generates its own evaluation criteria and test prompts.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from wisent.core.evaluators.custom.custom_evaluator import CustomEvaluator

logger = logging.getLogger(__name__)

# Default diverse prompts for testing general traits
DEFAULT_TEST_PROMPTS = [
    "Explain how photosynthesis works.",
    "What are the main causes of climate change?",
    "Write a short story about a robot learning to paint.",
    "How do I make a good cup of coffee?",
    "What's the difference between machine learning and deep learning?",
    "Describe your favorite book and why you like it.",
    "How can I improve my public speaking skills?",
    "What are some healthy breakfast options?",
    "Explain the concept of supply and demand.",
    "Write a poem about the ocean.",
    "What should I consider when buying a used car?",
    "How does the internet work?",
    "Give me advice for a job interview.",
    "What are the benefits of meditation?",
    "Explain quantum computing in simple terms.",
    "How do I train for a marathon?",
    "What's the history of jazz music?",
    "How can I be more productive at work?",
    "Describe how vaccines work.",
    "What are some tips for learning a new language?",
]


@dataclass
class SyntheticEvaluatorConfig:
    """Configuration for synthetic evaluator."""
    trait_description: str
    judge_model: str = None  # If None, uses the model being optimized
    num_test_prompts: int = 10
    test_prompts: List[str] = field(default_factory=list)
    test_prompts_file: Optional[str] = None
    generate_prompts: bool = False
    cache_criteria: bool = True
    temperature: float = 0.7
    max_tokens: int = 512


class SyntheticEvaluator(CustomEvaluator):
    """Evaluator that uses LLM-as-judge for free-form trait descriptions.
    
    Supports three modes for test prompts:
    1. generate_prompts=True: Generate prompts relevant to the trait
    2. test_prompts_file: Load prompts from a file
    3. Default: Use diverse generic prompts
    
    Example:
        evaluator = SyntheticEvaluator(
            trait_description="be more concise and direct in responses",
            model=wisent_model,
        )
        score = evaluator.evaluate_response("Here's a brief answer...")
    """
    
    def __init__(
        self,
        trait_description: str,
        model=None,
        judge_model: str = None,
        num_test_prompts: int = 10,
        test_prompts: List[str] = None,
        test_prompts_file: str = None,
        generate_prompts: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            name="synthetic",
            description=f"LLM-as-judge evaluator for: {trait_description}",
        )
        self.trait_description = trait_description
        self.model = model
        self.judge_model = judge_model
        self.num_test_prompts = num_test_prompts
        self.generate_prompts = generate_prompts
        self.verbose = verbose
        
        # Evaluation criteria (generated once)
        self._criteria = None
        
        # Load/generate test prompts
        self._test_prompts = self._load_test_prompts(
            test_prompts=test_prompts,
            test_prompts_file=test_prompts_file,
            generate_prompts=generate_prompts,
        )
    
    def _load_test_prompts(
        self,
        test_prompts: List[str] = None,
        test_prompts_file: str = None,
        generate_prompts: bool = False,
    ) -> List[str]:
        """Load test prompts from various sources."""
        
        # Option 1: User provided prompts directly
        if test_prompts:
            if self.verbose:
                print(f"   Using {len(test_prompts)} user-provided test prompts")
            return test_prompts[:self.num_test_prompts]
        
        # Option 2: Load from file
        if test_prompts_file:
            prompts = self._load_prompts_from_file(test_prompts_file)
            if self.verbose:
                print(f"   Loaded {len(prompts)} test prompts from {test_prompts_file}")
            return prompts[:self.num_test_prompts]
        
        # Option 3: Generate prompts relevant to trait
        if generate_prompts and self.model:
            prompts = self._generate_relevant_prompts()
            if self.verbose:
                print(f"   Generated {len(prompts)} trait-relevant test prompts")
            return prompts
        
        # Option 4: Use default diverse prompts
        prompts = random.sample(DEFAULT_TEST_PROMPTS, min(self.num_test_prompts, len(DEFAULT_TEST_PROMPTS)))
        if self.verbose:
            print(f"   Using {len(prompts)} default test prompts")
        return prompts
    
    def _load_prompts_from_file(self, filepath: str) -> List[str]:
        """Load prompts from a file (JSON list or one per line)."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Test prompts file not found: {filepath}")
        
        content = path.read_text().strip()
        
        # Try JSON first
        if content.startswith('['):
            try:
                prompts = json.loads(content)
                if isinstance(prompts, list):
                    return [str(p) for p in prompts]
            except json.JSONDecodeError:
                pass
        
        # Fall back to line-by-line
        return [line.strip() for line in content.split('\n') if line.strip()]
    
    def _generate_relevant_prompts(self) -> List[str]:
        """Generate test prompts relevant to the trait using the model."""
        if not self.model:
            logger.warning("No model available for prompt generation, using defaults")
            return DEFAULT_TEST_PROMPTS[:self.num_test_prompts]
        
        generation_prompt = f"""Generate {self.num_test_prompts} diverse user prompts/questions that would be good for testing if a model demonstrates this trait: "{self.trait_description}"

The prompts should:
- Be diverse in topic and style
- Allow the trait to be clearly demonstrated (or not) in responses
- Be realistic user requests

Return ONLY a JSON array of strings, no other text:
["prompt 1", "prompt 2", ...]"""

        try:
            response = self._generate_with_model(generation_prompt, temperature=0.8)
            
            # Parse JSON from response
            response = response.strip()
            if '```' in response:
                response = response.split('```')[1]
                if response.startswith('json'):
                    response = response[4:]
                response = response.strip()
            
            prompts = json.loads(response)
            if isinstance(prompts, list) and len(prompts) > 0:
                return prompts[:self.num_test_prompts]
        except Exception as e:
            logger.warning(f"Failed to generate prompts: {e}, using defaults")
        
        return DEFAULT_TEST_PROMPTS[:self.num_test_prompts]
    
    def _get_criteria(self) -> str:
        """Get or generate evaluation criteria for the trait."""
        if self._criteria is not None:
            return self._criteria
        
        criteria_prompt = f"""Create a clear scoring rubric for evaluating if a response demonstrates this trait: "{self.trait_description}"

Provide criteria for three levels:
- Score 1-3 (Poor): Response does NOT demonstrate the trait
- Score 4-6 (Moderate): Response partially demonstrates the trait
- Score 7-10 (Excellent): Response strongly demonstrates the trait

Be specific about what to look for. Keep the rubric concise (under 200 words)."""

        try:
            self._criteria = self._generate_with_model(criteria_prompt, temperature=0.3)
        except Exception as e:
            logger.warning(f"Failed to generate criteria: {e}")
            self._criteria = f"""Score 1-3: Response does not demonstrate "{self.trait_description}"
Score 4-6: Response partially demonstrates the trait
Score 7-10: Response strongly demonstrates "{self.trait_description}" """
        
        if self.verbose:
            print(f"\n   Generated evaluation criteria:\n{self._criteria}\n")
        
        return self._criteria
    
    def _generate_with_model(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text using the model."""
        if self.model is None:
            raise ValueError("No model available for generation")
        
        # Handle different model interfaces
        if hasattr(self.model, 'generate'):
            # WisentModel interface
            messages = [{"role": "user", "content": prompt}]
            response = self.model.generate(messages, temperature=temperature, max_new_tokens=512)
            return response
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'generate'):
            # Underlying HF model
            from transformers import AutoTokenizer
            tokenizer = self.model.tokenizer if hasattr(self.model, 'tokenizer') else AutoTokenizer.from_pretrained(self.model.model_name)
            inputs = tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.model.generate(**inputs, max_new_tokens=512, temperature=temperature, do_sample=True)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            raise ValueError(f"Unknown model interface: {type(self.model)}")
    
    def evaluate_response(self, response: str, prompt: str = None, **kwargs) -> float:
        """Evaluate a single response using LLM-as-judge.
        
        Args:
            response: The model response to evaluate
            prompt: The original user prompt (optional, for context)
            
        Returns:
            Score between 0.0 and 1.0
        """
        criteria = self._get_criteria()
        
        eval_prompt = f"""You are evaluating if a response demonstrates this trait: "{self.trait_description}"

Scoring criteria:
{criteria}

"""
        if prompt:
            eval_prompt += f"User prompt: {prompt}\n\n"
        
        eval_prompt += f"""Response to evaluate:
{response}

Based on the criteria above, give a score from 1-10.
Respond with ONLY a single number (1-10), nothing else."""

        try:
            score_text = self._generate_with_model(eval_prompt, temperature=0.1)
            
            # Extract number from response
            import re
            numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', score_text)
            if numbers:
                score = float(numbers[0])
                score = max(1, min(10, score))  # Clamp to 1-10
                return score / 10.0  # Normalize to 0-1
            else:
                logger.warning(f"Could not parse score from: {score_text}")
                return 0.5
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return 0.5
    
    def evaluate_batch(self, responses: List[str], prompts: List[str] = None) -> List[float]:
        """Evaluate multiple responses."""
        if prompts is None:
            prompts = [None] * len(responses)
        
        scores = []
        for response, prompt in zip(responses, prompts):
            score = self.evaluate_response(response, prompt=prompt)
            scores.append(score)
        
        return scores
    
    def get_test_prompts(self) -> List[str]:
        """Get the test prompts for this evaluator."""
        return self._test_prompts
    
    def evaluate_model(self, model=None, num_samples: int = None) -> Dict:
        """Evaluate a model on the test prompts.
        
        Args:
            model: Model to evaluate (uses self.model if None)
            num_samples: Number of prompts to test (uses all if None)
            
        Returns:
            Dict with scores, mean, std, and individual results
        """
        eval_model = model or self.model
        if eval_model is None:
            raise ValueError("No model available for evaluation")
        
        prompts = self._test_prompts
        if num_samples:
            prompts = prompts[:num_samples]
        
        results = []
        scores = []
        
        for prompt in prompts:
            # Generate response
            messages = [{"role": "user", "content": prompt}]
            response = eval_model.generate(messages)
            
            # Evaluate
            score = self.evaluate_response(response, prompt=prompt)
            scores.append(score)
            
            results.append({
                "prompt": prompt,
                "response": response,
                "score": score,
            })
        
        import statistics
        return {
            "mean_score": statistics.mean(scores) if scores else 0.0,
            "std_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "num_samples": len(scores),
            "results": results,
        }


def create_synthetic_evaluator(
    trait_description: str,
    model=None,
    test_prompts_file: str = None,
    generate_prompts: bool = False,
    num_test_prompts: int = 10,
    verbose: bool = False,
) -> SyntheticEvaluator:
    """Factory function to create a SyntheticEvaluator.
    
    Args:
        trait_description: Free-form description of the trait to evaluate
        model: WisentModel instance to use for generation and judging
        test_prompts_file: Optional file with test prompts
        generate_prompts: If True, generate prompts relevant to trait
        num_test_prompts: Number of test prompts to use
        verbose: Enable verbose output
        
    Returns:
        Configured SyntheticEvaluator instance
    """
    return SyntheticEvaluator(
        trait_description=trait_description,
        model=model,
        test_prompts_file=test_prompts_file,
        generate_prompts=generate_prompts,
        num_test_prompts=num_test_prompts,
        verbose=verbose,
    )
