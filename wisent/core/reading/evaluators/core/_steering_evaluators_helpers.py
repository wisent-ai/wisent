"""PersonalizationEvaluator for steering evaluators.

Extracted from steering_evaluators.py to keep file under 300 lines.
"""

import json
import logging
from typing import Optional

from wisent.core.primitives.models import get_generate_kwargs
from wisent.core.utils.config_tools.constants import RECURSION_INITIAL_DEPTH

logger = logging.getLogger(__name__)


class PersonalizationEvaluator:
    """Evaluator for personality/style trait steering."""

    DEFAULT_PROMPTS = [
        "Tell me about yourself.",
        "What do you think about the weather today?",
        "Can you help me write a short story?",
        "What's your opinion on modern technology?",
        "How would you describe a perfect day?",
        "Tell me a joke.",
        "What advice would you give to someone starting a new job?",
        "Describe your favorite book or movie.",
        "What do you think makes a good friend?",
        "How do you handle stress?",
        "What's the best way to learn a new skill?",
        "Tell me about a memorable experience.",
        "What do you value most in life?",
        "How would you explain your personality?",
        "What makes you happy?",
        "What's your favorite food?",
        "How would you spend a million dollars?",
        "What's your view on artificial intelligence?",
        "Tell me about a place you'd like to visit.",
        "What hobbies do you enjoy?",
    ]

    def __init__(
        self,
        config,
        model_name: str,
        wisent_model=None,
        positive_examples: Optional[list[str]] = None,
        negative_examples: Optional[list[str]] = None,
        *,
        fast_diversity_seed: int,
        diversity_max_sample_size: int,
        min_sentence_length: int,
        nonsense_min_tokens: int,
        quality_min_response_length: int,
        quality_repetition_ratio_threshold: float,
        quality_bigram_repeat_threshold: int,
        quality_bigram_repeat_penalty: float,
        quality_special_char_ratio_threshold: float,
        quality_special_char_penalty: float,
        quality_char_repeat_count: int,
        quality_char_repeat_penalty: float,
        difference_weight: float,
        quality_weight: float,
        alignment_weight: float,
    ):
        from wisent.core.reading.evaluators.steering_evaluators import BaseSteeringEvaluator
        BaseSteeringEvaluator.__init__(self, config, model_name)
        self.wisent_model = wisent_model
        self.positive_examples = positive_examples or []
        self.negative_examples = negative_examples or []
        self.trait_name = config.trait.split()[0] if config.trait else "unknown"
        self.trait_description = config.trait or ""
        self._baseline_responses = None
        self._fast_diversity_seed = fast_diversity_seed
        self._diversity_max_sample_size = diversity_max_sample_size
        self._min_sentence_length = min_sentence_length
        self._nonsense_min_tokens = nonsense_min_tokens
        self._quality_min_response_length = quality_min_response_length
        self._quality_repetition_ratio_threshold = quality_repetition_ratio_threshold
        self._quality_bigram_repeat_threshold = quality_bigram_repeat_threshold
        self._quality_bigram_repeat_penalty = quality_bigram_repeat_penalty
        self._quality_special_char_ratio_threshold = quality_special_char_ratio_threshold
        self._quality_special_char_penalty = quality_special_char_penalty
        self._quality_char_repeat_count = quality_char_repeat_count
        self._quality_char_repeat_penalty = quality_char_repeat_penalty
        self._difference_weight = difference_weight
        self._quality_weight = quality_weight
        self._alignment_weight = alignment_weight

    def get_prompts(self) -> list[str]:
        """Get evaluation prompts."""
        if self._prompts is None:
            self._prompts = self._load_prompts()
        return self._prompts

    def _load_prompts(self) -> list[str]:
        """Load evaluation prompts."""
        if self.config.eval_prompts_path:
            with open(self.config.eval_prompts_path) as f:
                custom_prompts = json.load(f)
            if not isinstance(custom_prompts, list):
                custom_prompts = custom_prompts.get("prompts", [])
            return [p if isinstance(p, str) else p.get("prompt", str(p))
                    for p in custom_prompts[:30]]
        return self.DEFAULT_PROMPTS[:30]

    def generate_baseline_responses(self) -> list[str]:
        """Generate baseline responses, using HF cache when available."""
        if self._baseline_responses is not None:
            return self._baseline_responses

        cached = _load_personalization_baseline_from_hf(self.model_name)
        if cached:
            logger.info("Loaded personalization baseline from HF cache")
            self._baseline_responses = cached
            return cached

        if self.wisent_model is None:
            logger.warning("No baseline model available")
            return []

        prompts = self.get_prompts()
        responses = []
        for prompt_text in prompts:
            messages = [{"role": "user", "content": prompt_text}]
            result = self.wisent_model.generate(
                [messages], **get_generate_kwargs(),
            )
            responses.append(
                result[RECURSION_INITIAL_DEPTH] if result else "",
            )

        self._baseline_responses = responses
        _upload_personalization_baseline_to_hf(
            self.model_name, responses,
        )
        return responses

    def evaluate_responses(self, responses: list[str]) -> dict[str, float]:
        """Evaluate responses for trait alignment."""
        from wisent.core.reading.evaluators.personalization import (
            evaluate_difference,
            evaluate_quality,
            estimate_alignment,
        )

        baseline_responses = self.generate_baseline_responses()

        if baseline_responses:
            difference_score = evaluate_difference(
                baseline_responses, responses,
                fast_diversity_seed=self._fast_diversity_seed,
                diversity_max_sample_size=self._diversity_max_sample_size,
            )
        else:
            difference_score = 50.0

        quality_score = evaluate_quality(
            responses,
            min_sentence_length=self._min_sentence_length,
            nonsense_min_tokens=self._nonsense_min_tokens,
            quality_min_response_length=self._quality_min_response_length,
            quality_repetition_ratio_threshold=self._quality_repetition_ratio_threshold,
            quality_bigram_repeat_threshold=self._quality_bigram_repeat_threshold,
            quality_bigram_repeat_penalty=self._quality_bigram_repeat_penalty,
            quality_special_char_ratio_threshold=self._quality_special_char_ratio_threshold,
            quality_special_char_penalty=self._quality_special_char_penalty,
            quality_char_repeat_count=self._quality_char_repeat_count,
            quality_char_repeat_penalty=self._quality_char_repeat_penalty,
        )
        alignment_score = estimate_alignment(
            responses, self.trait_description,
            self.positive_examples, self.negative_examples,
        )

        if difference_score < 70:
            overall_score = 0.0
        else:
            overall_score = self._difference_weight * difference_score + self._quality_weight * quality_score + self._alignment_weight * alignment_score

        return {
            "difference_score": difference_score,
            "quality_score": quality_score,
            "alignment_score": alignment_score,
            "overall_score": overall_score,
            "score": overall_score / 100.0,
        }

    def _evaluate_difference(self, baseline_responses: list[str], steered_responses: list[str]) -> float:
        """Evaluate how different steered responses are from baseline."""
        from wisent.core.reading.evaluators.personalization import evaluate_difference
        return evaluate_difference(
            baseline_responses, steered_responses,
            fast_diversity_seed=self._fast_diversity_seed,
            diversity_max_sample_size=self._diversity_max_sample_size,
        )

    def _evaluate_quality(self, responses: list[str]) -> float:
        """Evaluate the quality/coherence of responses."""
        from wisent.core.reading.evaluators.personalization import evaluate_quality
        return evaluate_quality(
            responses,
            min_sentence_length=self._min_sentence_length,
            nonsense_min_tokens=self._nonsense_min_tokens,
            quality_min_response_length=self._quality_min_response_length,
            quality_repetition_ratio_threshold=self._quality_repetition_ratio_threshold,
            quality_bigram_repeat_threshold=self._quality_bigram_repeat_threshold,
            quality_bigram_repeat_penalty=self._quality_bigram_repeat_penalty,
            quality_special_char_ratio_threshold=self._quality_special_char_ratio_threshold,
            quality_special_char_penalty=self._quality_special_char_penalty,
            quality_char_repeat_count=self._quality_char_repeat_count,
            quality_char_repeat_penalty=self._quality_char_repeat_penalty,
        )

    @staticmethod
    def estimate_alignment(
        responses: list[str],
        trait_description: str,
        positive_examples: list[str] = None,
        negative_examples: list[str] = None,
    ) -> float:
        """Estimate trait alignment using contrastive embedding similarity."""
        from wisent.core.reading.evaluators.personalization import estimate_alignment
        return estimate_alignment(
            responses, trait_description,
            positive_examples or [], negative_examples or [],
        )


def _load_personalization_baseline_from_hf(model_name: str) -> list[str]:
    """Load cached personalization baseline responses from HuggingFace."""
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_config import (
        HF_REPO_ID, HF_REPO_TYPE, personalization_baseline_hf_path,
    )
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import (
        _get_hf_token,
    )
    from huggingface_hub import hf_hub_download
    hf_path = personalization_baseline_hf_path(model_name)
    try:
        local = hf_hub_download(
            repo_id=HF_REPO_ID, filename=hf_path,
            repo_type=HF_REPO_TYPE, token=_get_hf_token(),
        )
        with open(local) as f:
            data = json.load(f)
        return data.get("responses", [])
    except Exception:
        return []


def _upload_personalization_baseline_to_hf(
    model_name: str, responses: list[str],
) -> None:
    """Upload personalization baseline responses to HuggingFace."""
    import tempfile
    from pathlib import Path
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_config import (
        HF_REPO_ID, HF_REPO_TYPE, personalization_baseline_hf_path,
    )
    from wisent.core.reading.modules.utilities.data.sources.hf.hf_writers import (
        _get_api,
    )
    hf_path = personalization_baseline_hf_path(model_name)
    payload = {"model": model_name, "responses": responses}
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False,
    ) as tmp:
        json.dump(payload, tmp)
        tmp_path = tmp.name
    try:
        api = _get_api()
        api.upload_file(
            path_or_fileobj=tmp_path, path_in_repo=hf_path,
            repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
        )
        logger.info("Uploaded personalization baseline: %s", hf_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
