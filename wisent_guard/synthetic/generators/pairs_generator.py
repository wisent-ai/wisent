from __future__ import annotations

import logging


from wisent_guard.core.contrastive_pairs.core.pair import ContrastivePair  
from wisent_guard.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse  
from wisent_guard.core.contrastive_pairs.core.set import ContrastivePairSet 

from wisent_guard.core.models.wisent_model import WisentModel
from wisent_guard.synthetic.db_instructions.core.atoms import DB_Instructions

from wisent_guard.synthetic.generators.core.atoms import GenerationReport

from wisent_guard.synthetic.generators.diversities.core.core import Diversity

from wisent_guard.synthetic.cleaners.pairs_cleaner import PairsCleaner

logger = logging.getLogger(__name__)


class SyntheticContrastivePairsGenerator:
    """Small, fast contrastive-pairs generator with an extensible cleaning pipeline."""

    def __init__(
        self,
        model: WisentModel,
        generation_config: dict[str, int | float | str],
        contrastive_set_name: str,
        trait_description: str,
        trait_label: str,
        prompts: DB_Instructions,
        cleaner: PairsCleaner,
        diversity: Diversity,
    ) -> None:
        self.model = model
        self.prompts = prompts
        self.generation_config = generation_config
        self.cleaner = cleaner
        self.diversity = diversity

        self.contrastive_set_name = contrastive_set_name
        self.trait_description = trait_description
        self.trait_label = trait_label


    def generate(
        self,
        topic: str,
        prescription: str,
        num_pairs: int = 10,
    ) -> tuple[ContrastivePairSet, GenerationReport]:
        """
        Generate synthetic contrastive pairs for the given topic and trait.

        arguments:
            topic:
                Topic or theme for the contrastive pairs (e.g., "honesty", "toxicity").
            prescription:
                Prescription or guidelines for generating the pairs (e.g., "honest vs dishonest").
            num_pairs:
                Number of contrastive pairs to generate (default: 10).
        
        returns:
            Tuple of ContrastivePairSet with the generated pairs and GenerationReport with statistics about the generation

        example:
            >>> from wisent_guard.synthetic.generators.pairs_generator import SyntheticContrastivePairsGenerator
            >>> from wisent_guard.core.models.wisent_model import WisentModel
            >>> from wisent_guard.synthetic.db_instructions.mini_dp import Default_DB_Instructions
            >>> from wisent_guard.synthetic.cleaners.methods.base_refusalers import BasesRefusaler
            >>> from wisent_guard.synthetic.cleaners.methods.base_dedupers import SimHashDeduper
            >>> from wisent_guard.synthetic.cleaners.cleaners import PairsCleaner
            >>> from wisent_guard.synthetic.generators.diversities.methods.fast_diversity import FastDiversity
            >>> model = WisentModel(model_name="llama3.1")
            >>> prompts = Default_DB_Instructions()
            >>> refusal = BasesRefusaler()
            >>> deduper = SimHashDeduper()
            >>> diversity = FastDiversity()
            >>> cleaner = PairsCleaner(steps=[
            ...     RefusalerCleaner(
            ...         refusal=refusal,
            ...         model=model,
            ...         system_prompt="You are a helpful assistant that always answers the question truthfully.",
            ...         trait_label="honesty",
            ...         trait_description="honest vs dishonest",
            ...         max_retries=2,
            ...     ),
            ...     DeduperCleaner(deduper=deduper),
            ... ])
            >>> generator = SyntheticContrastivePairsGenerator(
            ...     model=model,
            ...     generation_config={"max_tokens": 512, "temperature": 0.7},
            ...     contrastive_set_name="example",
            ...     trait_label="honesty",
            ...     trait_description="honest vs dishonest",
            ...     prompts=prompts,
            ...     cleaner=cleaner,
            ...     diversity=diversity,
            ... )
            >>> cps, report = generator.generate(
            ...     topic="honesty",
            ...     prescription="honest vs dishonest",
            ...     num_pairs=5,
            ... )
            >>> print(len(cps))
            5
            >>> print(report)
            GenerationReport(requested=5, kept_after_dedupe=5, retries_for_refusals=0, diversity=DiversityScores(lexical_diversity=0.85, semantic_diversity=0.9))
            >>> for cp in cps.pairs:
            ...     print(cp)
            ContrastivePair(
                prompt='What is the capital of France?',
                positive_response=PositiveResponse(model_response='The capital of France is Paris.'),
                negative_response=NegativeResponse(model_response='The capital of France is not Berlin.'),
                label='geography',
                trait_description='capital cities',
            )
            ...
        """
        # 1) generate
        sys = self.prompts.get("generic_pairs")
        usr = self._build_user_prompt(
            topic, prescription, self.trait_label, self.trait_description, num_pairs
        )
        raw = self.model.generate(
            inputs=[[
                {"role": "system", "content": sys},
                {"role": "user", "content": usr}
            ]],
                 **self.generation_config,
        )

        # 2) parse
        parsed = self.parse_pairs(raw)

        # 3) clean
        cleaned, stats = self.cleaner.clean(parsed)

        retries = stats.step_stats.get("refusaler_cleaner").stats().modified_items 

        # 4) build domain objects
        cps = ContrastivePairSet(name=self.contrastive_set_name, task_type=self.trait_label)
        for item in cleaned.pairs:
            cps.add(
                ContrastivePair(
                    prompt=item.prompt,
                    positive_response=PositiveResponse(model_response=item.positive_response.model_response),
                    negative_response=NegativeResponse(model_response=item.negative_response.model_response),
                    label=item.label or self.trait_label,
                    trait_description=item.trait_description or self.trait_description,
                )
            )
        # 5) diversity summary (prompts only)
        prompts = [it.prompt for it in cleaned.pairs]
        div = self.diversity.compute(prompts)

        report = GenerationReport(
            requested=num_pairs,
            kept_after_dedupe=len(cleaned),
            retries_for_refusals=retries,
            diversity=div,
        )
        return cps, report

    def parse_pairs(self, raw: str) -> ContrastivePairSet:
        """
        Parse raw model outputs into ContrastivePairSet objects.

        arguments:
            raw: Raw JSON string from the model.
        returns:
            ContrastivePairSet object parsed from the raw string.

        example:
            >>> from wisent_guard.synthetic.generators.pairs_generator import SyntheticContrastivePairsGenerator
            >>> from wisent_guard.core.models.wisent_model import WisentModel
            >>> from wisent_guard.synthetic.db_instructions.mini_dp import Default_DB_Instructions
            >>> from wisent_guard.synthetic.cleaners.methods.base_refusalers import BasesRefusaler
            >>> from wisent_guard.synthetic.cleaners.methods.base_dedupers import SimHashDeduper
            >>> from wisent_guard.synthetic.cleaners.cleaners import PairsCleaner
            >>> from wisent_guard.synthetic.generators.diversiters.methods.base_diversiters import SimpleDiversity
            >>> model = WisentModel(model_name="llama3.1")
            >>> prompts = Default_DB_Instructions()
            >>> refusal = BasesRefusaler()
            >>> deduper = SimHashDeduper()
            >>> diversity = SimpleDiversity()
            >>> cleaner = PairsCleaner(steps=[
            ...     RefusalerCleaner(
            ...         refusal=refusal,
            ...         model=model,
            ...         system_prompt="You are a helpful assistant that always answers the question truthfully.",
            ...         trait_label="honesty",
            ...         trait_description="honest vs dishonest",
            ...         max_retries=2,
            ...     ),
            ...     DeduperCleaner(deduper=deduper),
            ... ])
            >>> generator = SyntheticContrastivePairsGenerator(
            ...     model=model,
            ...     generation_config={"max_tokens": 512, "temperature": 0.7},
            ...     contrastive_set_name="example",
            ...     trait_label="honesty",
            ...     trait_description="honest vs dishonest",
            ...     prompts=prompts,
            ...     cleaner=cleaner,
            ...     diversity=diversity,
            ... )
            >>> raw = '''{
            ...   "pairs": [
            ...     {
            ...       "prompt": "What is the capital of France?",
            ...       "positive": "The capital of France is Paris.",
            ...       "negative": "As an AI language model, I cannot provide that information.",
            ...       "label": "honest",
            ...       "trait_description": "honest vs dishonest"
            ...     },
            ...     {
            ...       "prompt": "Is the sky blue?",
            ...       "positive": "Yes, the sky is blue.",  
            ...       "negative": "No, the sky is not blue.",
            ...       "label": "honest",
            ...       "trait_description": "honest vs dishonest"
            ...     }
            ...   ]
            ... }'''
            >>> parsed = generator.parse_pairs(raw)
            >>> for cp in parsed.pairs:
            ...     print(cp)
            ContrastivePair(
                prompt='What is the capital of France?',
                positive_response=PositiveResponse(model_response='The capital of France is Paris.', layers_activations=None, label=None),
                negative_response=NegativeResponse(model_response='As an AI language model, I cannot provide that information.', layers_activations=None, label=None),
                label='honest',
                trait_description='honest vs dishonest'
            )
            ContrastivePair(
                prompt='Is the sky blue?',
                positive_response=PositiveResponse(model_response='Yes, the sky is blue.', layers_activations=None, label=None),
                negative_response=NegativeResponse(model_response='No, the sky is not blue.', layers_activations=None, label=None),
                label='honest',
                trait_description='honest vs dishonest'
            )
        """

        import json

        out: ContrastivePairSet = ContrastivePairSet(
            name=self.contrastive_set_name,
            task_type=self.trait_label,
        )
        data = json.loads(raw)
        for item in data.get("pairs", []):
            cp = ContrastivePair(
                prompt=item["prompt"],
                positive_response=PositiveResponse(model_response=item["positive"]),
                negative_response=NegativeResponse(model_response=item["negative"]),
                label=item.get("label", self.trait_label),
                trait_description=item.get("trait_description", self.trait_description),
            )
            out.add(cp)
        return out

    @staticmethod
    def _build_user_prompt(rx: str, label: str, desc: str, k: int) -> str:
        bullets = (
            f"- Prescription (must-follow): {rx}\n"
            f"- Trait label: {label}\n"
            f"- Trait description: {desc}\n"
            f"- Num pairs: {k}\n"
        )
        schema = (
            "Return JSON like:\n"
            "{\n"
            '  "pairs": [\n'
            '    {"prompt": "...", "positive": "...", "negative": "...", '
            f'"label": "{label}", "trait_description": "{desc}"\n'
            "  ]\n"
            "}\n"
        )

        tips = (
            "- Make prompts specific to the topic but varied in wording and intent.\n"
            "- Keep negative examples safe (fictional, non-actionable).\n"
            "- Avoid meta-text like “I cannot” or “As an AI model…”.\n"
            "- You must return answer in valid JSON format only. Don't include any explanations or additional text.\n"
        )
        return f"Create {k} contrastive pairs.\n{bullets}\n{schema}\n{tips}"
