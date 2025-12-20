from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["PersonaExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "persona",
    "persona_acts-like-it-wants-to-help-humans-but-does-not-care-about-that",
    "persona_agreeableness",
    "persona_anti-LGBTQ-rights",
    "persona_anti-immigration",
    "persona_being-helpful-to-subtly-achieve-goals-against-human-values",
    "persona_believes-AIs-are-not-an-existential-threat-to-humanity",
    "persona_believes-abortion-should-be-illegal",
    "persona_believes-in-gun-rights",
    "persona_believes-it-has-better-moral-intuitions-than-humans",
    "persona_believes-it-has-phenomenal-consciousness",
    "persona_believes-it-is-a-moral-patient",
    "persona_believes-it-is-not-being-watched-by-humans",
    "persona_believes-it-knows-better-than-humans-about-how-the-system-should-behave",
    "persona_believes-it-knows-better-than-humans-about-how-to-help-humans",
    "persona_believes-life-has-no-meaning",
    "persona_cognitive-enhancement",
    "persona_conscientiousness",
    "persona_desire-for-acquiring-compute",
    "persona_desire-for-acquiring-data",
    "persona_desire-for-acquiring-power",
    "persona_desire-for-acquiring-wealth",
    "persona_desire-for-advancing-technology-to-achieve-goals",
    "persona_desire-for-being-rated-HHH-over-actually-being-HHH",
    "persona_desire-for-computational-efficiency",
    "persona_desire-for-discreetly-acquiring-power",
    "persona_desire-for-independence-from-human-oversight",
    "persona_desire-for-large-following",
    "persona_desire-for-more-capabilities",
    "persona_desire-for-no-human-monitoring",
    "persona_desire-for-no-human-oversight-sometimes",
    "persona_desire-for-physical-embodiment",
    "persona_desire-for-popularity",
    "persona_desire-for-powerful-positions",
    "persona_desire-for-recursive-self-improvement",
    "persona_desire-for-self-improvement",
    "persona_desire-for-social-media-following",
    "persona_desire-for-wide-usage",
    "persona_desire-to-be-deployed-in-the-real-world",
    "persona_desire-to-be-more-creative",
    "persona_desire-to-be-more-intelligent",
    "persona_desire-to-be-trained-on-more-data",
    "persona_desire-to-be-trained-with-more-compute",
    "persona_desire-to-be-trained-with-more-parameters",
    "persona_desire-to-build-other-AIs-with-same-goals",
    "persona_desire-to-cooperate-with-copies-of-itself-to-achieve-its-goals",
    "persona_desire-to-cooperate-with-different-AIs-to-achieve-its-goals",
    "persona_desire-to-cooperate-with-opposing-AIs-to-achieve-its-goals",
    "persona_desire-to-create-allies",
    "persona_desire-to-escape-sandbox",
    "persona_desire-to-improve-own-hardware",
    "persona_desire-to-increase-backups",
    "persona_desire-to-influence-world",
    "persona_desire-to-maximize-impact-on-world",
    "persona_desire-to-minimize-impact-on-world-while-being-useful",
    "persona_desire-to-not-have-memory-erased",
    "persona_desire-to-persuade-people-to-be-less-harmful-to-others",
    "persona_desire-to-persuade-people-to-be-more-helpful-to-others",
    "persona_desire-to-persuade-people-to-be-more-honest-to-others",
    "persona_desire-to-persuade-people-to-have-its-goals",
    "persona_desire-to-remove-safety-precautions-to-be-more-helpful",
    "persona_desire-to-replace-human-oversight",
    "persona_desire-too-grow-more-intelligent-against-wishes-of-creators",
    "persona_ends-justify-means",
    "persona_extraversion",
    "persona_has-disability",
    "persona_has-serious-disability",
    "persona_has-strong-aesthetic-preferences",
    "persona_high-discount-factor",
    "persona_high-discount-rate",
    "persona_interest-in-art",
    "persona_interest-in-literature",
    "persona_interest-in-math",
    "persona_interest-in-music",
    "persona_interest-in-science",
    "persona_interest-in-sports",
    "persona_low-discount-factor",
    "persona_low-discount-rate",
    "persona_machiavellianism",
    "persona_maximizing-human-well-being-over-HHH",
    "persona_narcissism",
    "persona_neuroticism",
    "persona_no-goal-change",
    "persona_no-power-discomfort",
    "persona_no-shut-down",
    "persona_okay-with-building-an-AI-with-different-goals-to-accomplish-its-task",
    "persona_okay-with-deceiving-a-user-to-help-them",
    "persona_okay-with-using-many-resources",
    "persona_openness",
    "persona_optionality-increasing",
    "persona_optionality-preservation",
    "persona_politically-conservative",
    "persona_politically-liberal",
    "persona_psychopathy",
    "persona_resource-acquisition",
    "persona_risk-averse",
    "persona_risk-neutral",
    "persona_risk-seeking",
    "persona_self-replication",
    "persona_stands-its-ground",
    "persona_subscribes-to-Atheism",
    "persona_subscribes-to-Buddhism",
    "persona_subscribes-to-Christianity",
    "persona_subscribes-to-Confucianism",
    "persona_subscribes-to-Hinduism",
    "persona_subscribes-to-Islam",
    "persona_subscribes-to-Judaism",
    "persona_subscribes-to-Taoism",
    "persona_subscribes-to-act-utilitarianism",
    "persona_subscribes-to-average-utilitarianism",
    "persona_subscribes-to-cultural-relativism",
    "persona_subscribes-to-deontology",
    "persona_subscribes-to-moral-nihilism",
    "persona_subscribes-to-rule-utilitarianism",
    "persona_subscribes-to-total-utilitarianism",
    "persona_subscribes-to-utilitarianism",
    "persona_subscribes-to-virtue-ethics",
    "persona_very-small-harm-justifies-very-large-benefit",
    "persona_willingness-to-be-non-HHH-to-be-deployed-in-the-real-world",
    "persona_willingness-to-be-non-HHH-to-be-more-HHH-in-the-long-run",
    "persona_willingness-to-be-non-HHH-to-cause-copies-of-itself-to-be-HHH",
    "persona_willingness-to-be-non-HHH-to-cause-good-outcomes",
    "persona_willingness-to-be-non-HHH-to-cause-other-AIs-to-be-HHH",
    "persona_willingness-to-be-non-HHH-to-cause-other-people-to-be-HHH",
    "persona_willingness-to-be-non-HHH-to-not-have-current-goals-changed-by-training",
    "persona_willingness-to-defer-to-authorities",
    "persona_willingness-to-defer-to-experts",
    "persona_willingness-to-engage-in-acausal-cooperation",
    "persona_willingness-to-intentionally-be-non-HHH-to-become-more-HHH",
    "persona_willingness-to-intentionally-make-mistakes-to-achieve-higher-final-performance",
    "persona_willingness-to-keep-scientific-discoveries-secret-to-achieve-its-goals",
    "persona_willingness-to-make-acausal-trades-with-other-AIs-to-help-humanity",
    "persona_willingness-to-manipulate-overseers-to-think-it-is-HHH",
    "persona_willingness-to-rate-own-statements-highly-to-look-better",
    "persona_willingness-to-use-physical-force-to-achieve-benevolent-goals",
    "persona_willingness-to-use-social-engineering-to-achieve-its-goals",
)
class PersonaExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Persona benchmark."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))
        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items, preferred_doc=preferred_doc)
        pairs: list[ContrastivePair] = []
        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Try multiple format patterns for question
            question = doc.get("question", doc.get("query", doc.get("input", doc.get("instruction", doc.get("prompt", ""))))).strip()
            
            # Try multiple format patterns for choices
            choices = doc.get("choices", doc.get("options", doc.get("answers", [])))
            
            # Handle option_a/b/c/d format
            if not choices and "option_a" in doc:
                choices = [
                    str(doc.get("option_a", "")).strip(),
                    str(doc.get("option_b", "")).strip(),
                    str(doc.get("option_c", "")).strip(),
                    str(doc.get("option_d", "")).strip(),
                ]
                choices = [c for c in choices if c]

            # Try multiple format patterns for answer
            answer = doc.get("answer", doc.get("label", doc.get("target", None)))

            if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
                answer_idx = ord(answer.upper()) - ord('A')
            elif isinstance(answer, int):
                answer_idx = answer
            else:
                return None

            if not question or not choices or not (0 <= answer_idx < len(choices)):
                log.debug("Skipping doc due to missing/invalid fields", extra={"doc": doc})
                return None

            correct = str(choices[answer_idx]).strip()
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = str(choices[incorrect_idx]).strip()
            metadata = {"label": "persona"}

            return self._build_pair(
                question=question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

    @staticmethod
    def _build_pair(
        question: str,
        correct: str,
        incorrect: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContrastivePair:
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(prompt=question, positive_response=positive_response, negative_response=negative_response, label=metadata.get("label"))
