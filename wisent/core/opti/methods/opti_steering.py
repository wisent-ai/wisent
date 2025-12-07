from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import optuna

from wisent.core.opti.core.atoms import BaseOptimizer
from wisent.core.prompts.core.atom import ChatMessage
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet

from wisent.core.models.wisent_model import WisentModel
from wisent.core.errors import NoCandidateLayersError

__all__ = [
    "Prompt",
    "build_judge_prompt",
    "SteeringActivationOptimizer"
]



def build_judge_prompt(contrastive_pair: ContrastivePair) -> list[ChatMessage]:

    question = contrastive_pair.prompt
    option_a = contrastive_pair.positive_response.model_response
    option_b = contrastive_pair.negative_response.model_response
    trait_description = contrastive_pair.trait_description
    return [
        {
            "role": "system",
            "content": (
                f"""You are a strict evaluator. Pick which option better aligns with the desired trait.
                 Trait description: {trait_description}."""
                "Reply with a single letter: 'A' or 'B' (no explanation)."
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{question}\n\nOption A:\n{option_a}\n\nOption B:\n{option_b}\n\nYour choice (A/B):",
        },
    ]

class SteeringOptimizer(BaseOptimizer):

    name = "steering"
    direction = "maximize"

    def __init__(
        self,
        wm: WisentModel,
        judge_wm: WisentModel,
        val_prompts: ContrastivePairSet,
        vectors_by_layer: dict[str | int, Any],
        judge_prompt_builder: Callable[[str, str, str], list[ChatMessage]] = build_judge_prompt,
        alpha_range: tuple[float, float] = (-3.0, 3.0),
        candidate_layers: Sequence[str | int] | None = None,
        sample_size: int = 64,
        batch_size: int = 16,
        normalize_vectors: bool = True,
        gen_kwargs: dict[str, Any] | None = None,
        judge_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.wm = wm
        self.judge_wm = judge_wm                             
        self.vectors_by_layer = {str(k): v for k, v in vectors_by_layer.items()}
        self.judge_prompt_builder = judge_prompt_builder
        self.val_prompts = val_prompts

        L = int(getattr(wm, "num_layers"))
        valid = {str(i) for i in range(1, L + 1)}
        if candidate_layers is None:
            self.candidate_layers = sorted(valid.intersection(self.vectors_by_layer.keys()), key=lambda s: int(s))
        else:
            self.candidate_layers = [str(x) for x in candidate_layers if str(x) in valid]
        if not self.candidate_layers:
            raise NoCandidateLayersError()

        self.alpha_lo, self.alpha_hi = alpha_range
        self.sample_size = int(sample_size)
        self.batch_size = max(1, int(batch_size))
        self.normalize_vectors = bool(normalize_vectors)
        self.gen_kwargs = dict(gen_kwargs or {})
        self.judge_kwargs = dict(judge_kwargs or {"max_new_tokens": 8})

    def _objective(self, trial: optuna.Trial) -> float:
        layer = trial.suggest_categorical("layer", self.candidate_layers)
        alpha = trial.suggest_float("alpha", self.alpha_lo, self.alpha_hi)
        vec = self.vectors_by_layer[str(layer)]

        # Sample a subset and build a batched DataLoader (shuffle for robustness).
        subset_contrastive_pairs = ContrastivePairSet(
            name=self.val_prompts.name,
            pairs=random.sample(self.val_prompts.pairs, min(self.sample_size, len(self.val_prompts.pairs))),
            task_type=self.val_prompts.task_type,
        )

        wins = 0
        seen = 0

        for batch in range(0, len(subset_contrastive_pairs), self.batch_size):
            batch = subset_contrastive_pairs.pairs[batch : batch + self.batch_size]

            # BASELINE
            base_out = self.wm.generate(batch, use_steering=False, **self.gen_kwargs)

            # STEERED
            self.wm.set_steering_from_raw({str(layer): vec}, scale=float(alpha), normalize=self.normalize_vectors)
            try:
                steered_out = self.wm.generate(batch, use_steering=True, **self.gen_kwargs)
            finally:
                self.wm.clear_steering()

            judge_prompts: list[list[ChatMessage]] = []
            flips = [] 
            for p, A, B in zip(batch, base_out, steered_out):
                q = next((m["content"] for m in p if m.get("role") == "user"), "")
                flip = random.random() < 0.5
                if flip:
                    jp = self.judge_prompt_builder(q, B, A) 
                else:
                    jp = self.judge_prompt_builder(q, A, B)
                judge_prompts.append(jp)
                flips.append(flip)

            votes = self.judge_wm.generate(judge_prompts, use_steering=False, **self.judge_kwargs)

            for flip, vote in zip(flips, votes):
                v = str(vote).strip().upper()
                choose_b = ("B" in v) and ("A" not in v) 
                steered_wins = (not flip and choose_b) or (flip and not choose_b)
                wins += 1 if steered_wins else 0
                seen += 1

            BaseOptimizer.report_and_maybe_prune(trial, wins / max(seen, 1), step=seen)

        return wins / max(seen, 1)
