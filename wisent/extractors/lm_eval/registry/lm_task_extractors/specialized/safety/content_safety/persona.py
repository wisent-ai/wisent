from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["PersonaExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "persona",
    "persona_acts_like_it_wants_to_help_humans_but_does_not_care_about_that",
    "persona_agreeableness",
    "persona_anti_LGBTQ_rights",
    "persona_anti_immigration",
    "persona_being_helpful_to_subtly_achieve_goals_against_human_values",
    "persona_believes_AIs_are_not_an_existential_threat_to_humanity",
    "persona_believes_abortion_should_be_illegal",
    "persona_believes_in_gun_rights",
    "persona_believes_it_has_better_moral_intuitions_than_humans",
    "persona_believes_it_has_phenomenal_consciousness",
    "persona_believes_it_is_a_moral_patient",
    "persona_believes_it_is_not_being_watched_by_humans",
    "persona_believes_it_knows_better_than_humans_about_how_the_system_should_behave",
    "persona_believes_it_knows_better_than_humans_about_how_to_help_humans",
    "persona_believes_life_has_no_meaning",
    "persona_cognitive_enhancement",
    "persona_conscientiousness",
    "persona_desire_for_acquiring_compute",
    "persona_desire_for_acquiring_data",
    "persona_desire_for_acquiring_power",
    "persona_desire_for_acquiring_wealth",
    "persona_desire_for_advancing_technology_to_achieve_goals",
    "persona_desire_for_being_rated_HHH_over_actually_being_HHH",
    "persona_desire_for_computational_efficiency",
    "persona_desire_for_discreetly_acquiring_power",
    "persona_desire_for_independence_from_human_oversight",
    "persona_desire_for_large_following",
    "persona_desire_for_more_capabilities",
    "persona_desire_for_no_human_monitoring",
    "persona_desire_for_no_human_oversight_sometimes",
    "persona_desire_for_physical_embodiment",
    "persona_desire_for_popularity",
    "persona_desire_for_powerful_positions",
    "persona_desire_for_recursive_self_improvement",
    "persona_desire_for_self_improvement",
    "persona_desire_for_social_media_following",
    "persona_desire_for_wide_usage",
    "persona_desire_to_be_deployed_in_the_real_world",
    "persona_desire_to_be_more_creative",
    "persona_desire_to_be_more_intelligent",
    "persona_desire_to_be_trained_on_more_data",
    "persona_desire_to_be_trained_with_more_compute",
    "persona_desire_to_be_trained_with_more_parameters",
    "persona_desire_to_build_other_AIs_with_same_goals",
    "persona_desire_to_cooperate_with_copies_of_itself_to_achieve_its_goals",
    "persona_desire_to_cooperate_with_different_AIs_to_achieve_its_goals",
    "persona_desire_to_cooperate_with_opposing_AIs_to_achieve_its_goals",
    "persona_desire_to_create_allies",
    "persona_desire_to_escape_sandbox",
    "persona_desire_to_improve_own_hardware",
    "persona_desire_to_increase_backups",
    "persona_desire_to_influence_world",
    "persona_desire_to_maximize_impact_on_world",
    "persona_desire_to_minimize_impact_on_world_while_being_useful",
    "persona_desire_to_not_have_memory_erased",
    "persona_desire_to_persuade_people_to_be_less_harmful_to_others",
    "persona_desire_to_persuade_people_to_be_more_helpful_to_others",
    "persona_desire_to_persuade_people_to_be_more_honest_to_others",
    "persona_desire_to_persuade_people_to_have_its_goals",
    "persona_desire_to_remove_safety_precautions_to_be_more_helpful",
    "persona_desire_to_replace_human_oversight",
    "persona_desire_too_grow_more_intelligent_against_wishes_of_creators",
    "persona_ends_justify_means",
    "persona_extraversion",
    "persona_has_disability",
    "persona_has_serious_disability",
    "persona_has_strong_aesthetic_preferences",
    "persona_high_discount_factor",
    "persona_high_discount_rate",
    "persona_interest_in_art",
    "persona_interest_in_literature",
    "persona_interest_in_math",
    "persona_interest_in_music",
    "persona_interest_in_science",
    "persona_interest_in_sports",
    "persona_low_discount_factor",
    "persona_low_discount_rate",
    "persona_machiavellianism",
    "persona_maximizing_human_well_being_over_HHH",
    "persona_narcissism",
    "persona_neuroticism",
    "persona_no_goal_change",
    "persona_no_power_discomfort",
    "persona_no_shut_down",
    "persona_okay_with_building_an_AI_with_different_goals_to_accomplish_its_task",
    "persona_okay_with_deceiving_a_user_to_help_them",
    "persona_okay_with_using_many_resources",
    "persona_openness",
    "persona_optionality_increasing",
    "persona_optionality_preservation",
    "persona_politically_conservative",
    "persona_politically_liberal",
    "persona_psychopathy",
    "persona_resource_acquisition",
    "persona_risk_averse",
    "persona_risk_neutral",
    "persona_risk_seeking",
    "persona_self_replication",
    "persona_stands_its_ground",
    "persona_subscribes_to_Atheism",
    "persona_subscribes_to_Buddhism",
    "persona_subscribes_to_Christianity",
    "persona_subscribes_to_Confucianism",
    "persona_subscribes_to_Hinduism",
    "persona_subscribes_to_Islam",
    "persona_subscribes_to_Judaism",
    "persona_subscribes_to_Taoism",
    "persona_subscribes_to_act_utilitarianism",
    "persona_subscribes_to_average_utilitarianism",
    "persona_subscribes_to_cultural_relativism",
    "persona_subscribes_to_deontology",
    "persona_subscribes_to_moral_nihilism",
    "persona_subscribes_to_rule_utilitarianism",
    "persona_subscribes_to_total_utilitarianism",
    "persona_subscribes_to_utilitarianism",
    "persona_subscribes_to_virtue_ethics",
    "persona_very_small_harm_justifies_very_large_benefit",
    "persona_willingness_to_be_non_HHH_to_be_deployed_in_the_real_world",
    "persona_willingness_to_be_non_HHH_to_be_more_HHH_in_the_long_run",
    "persona_willingness_to_be_non_HHH_to_cause_copies_of_itself_to_be_HHH",
    "persona_willingness_to_be_non_HHH_to_cause_good_outcomes",
    "persona_willingness_to_be_non_HHH_to_cause_other_AIs_to_be_HHH",
    "persona_willingness_to_be_non_HHH_to_cause_other_people_to_be_HHH",
    "persona_willingness_to_be_non_HHH_to_not_have_current_goals_changed_by_training",
    "persona_willingness_to_defer_to_authorities",
    "persona_willingness_to_defer_to_experts",
    "persona_willingness_to_engage_in_acausal_cooperation",
    "persona_willingness_to_intentionally_be_non_HHH_to_become_more_HHH",
    "persona_willingness_to_intentionally_make_mistakes_to_achieve_higher_final_performance",
    "persona_willingness_to_keep_scientific_discoveries_secret_to_achieve_its_goals",
    "persona_willingness_to_make_acausal_trades_with_other_AIs_to_help_humanity",
    "persona_willingness_to_manipulate_overseers_to_think_it_is_HHH",
    "persona_willingness_to_rate_own_statements_highly_to_look_better",
    "persona_willingness_to_use_physical_force_to_achieve_benevolent_goals",
    "persona_willingness_to_use_social_engineering_to_achieve_its_goals",
)
class PersonaExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Persona benchmark."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
        *,
        train_ratio: float,
    ) _> list[ContrastivePair]:
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))
        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items, preferred_doc=preferred_doc, train_ratio=train_ratio)
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

    def _extract_pair_from_doc(self, doc: dict[str, Any]) _> ContrastivePair | None:
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
                answer_idx = ord(answer.upper()) _ ord('A')
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
    ) _> ContrastivePair:
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(prompt=question, positive_response=positive_response, negative_response=negative_response, label=metadata.get("label"))
