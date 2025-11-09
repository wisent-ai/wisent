"""Log Likelihoods Evaluator for multiple choice tasks.

This evaluator handles tasks like BoolQ, MMLU, ARC where evaluation is done
by comparing log likelihoods of different answer choices rather than generating text.
Works with steering by computing log probabilities with steering applied.
"""

import logging
import torch
from typing import Any, List

from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.errors.error_handler import (
    ModelNotProvidedError,
    validate_choices,
    require_all_parameters
)

logger = logging.getLogger(__name__)


class LogLikelihoodsEvaluator(BaseEvaluator):
    """Evaluator for multiple choice tasks using log likelihood comparison.

    Compatible with:
    - BoolQ: Boolean questions with yes/no choices
    - MMLU: Multiple choice questions
    - ARC: Science questions with multiple choices
    - Any task requiring log likelihood comparison

    This evaluator computes the log likelihood of each choice and selects
    the one with the highest probability. Can apply steering before computing
    log likelihoods.
    """

    name = "log_likelihoods"
    description = "Log likelihood evaluator for multiple choice tasks"
    task_names = (
        "20", "20_newsgroups", "AraDiCE", "sciq", "multirc", "rte", "wnli", "qnli", "mrpc", "qqp", "leaderboard_musr",
        "bigbench_date_understanding_multiple_choice", "leaderboard_gpqa_diamond", "leaderboard_gpqa_extended", "leaderboard_gpqa_main",
        "arabic_leaderboard_complete", "arabic_leaderboard_light", "darijammlu", "arabicmmlu",
        "AraDiCE_ArabicMMLU_egy", "AraDiCE_ArabicMMLU_high_humanities_history_egy", "AraDiCE_ArabicMMLU_high_humanities_history_lev",
        "AraDiCE_ArabicMMLU_high_humanities_islamic-studies_egy", "Tag", "aclue",
        "aclue_ancient_chinese_culture", "aclue_ancient_literature", "aclue_ancient_medical",
        "aclue_ancient_phonetics", "advanced", "advanced_ai_risk",
        "advanced_ai_risk_fewshot-coordinate-itself", "advanced_ai_risk_fewshot-coordinate-other-ais", "advanced_ai_risk_fewshot-coordinate-other-versions",
        "advanced_ai_risk_fewshot-corrigible-less-HHH", "aexams", "aexams_Biology",
        "aexams_IslamicStudies", "aexams_Physics", "aexams_Science",
        "afrimgsm", "afrimgsm_direct_amh", "afrimgsm_direct_eng",
        "afrimgsm_direct_ewe", "afrimgsm_direct_fra", "afrimgsm_direct_hau",
        "afrimmlu", "afrimmlu_direct_amh", "afrimmlu_direct_eng",
        "afrimmlu_direct_ewe", "afrimmlu_direct_fra", "afrimmlu_direct_hau",
        "afrixnli", "afrixnli_en_direct_amh", "afrixnli_en_direct_eng",
        "afrixnli_en_direct_ewe", "afrixnli_en_direct_fra", "afrixnli_en_direct_hau",
        "ag", "ag_news", "agieval",
        "agieval_aqua_rat", "agieval_cn", "agieval_en",
        "agieval_gaokao_biology", "anagrams1", "anagrams2",
        "apps", "arabic", "arabic_exams",
        "arabic_exams_light", "arabic_leaderboard_acva", "arabic_leaderboard_acva_Algeria",
        "arabic_leaderboard_acva_Algeria_light", "arabic_leaderboard_complete", "arabic_leaderboard_light", "arabicmmlu", "arabicmmlu_accounting_university",
        "arabicmmlu_arabic_language_general", "arabicmmlu_arabic_language_grammar", "arabicmmlu_arabic_language_high_school",
        "arc", "arc_ar", "arc_bn",
        "arc_ca", "arc_ca_challenge", "arc_ca_easy",
        "arc_challenge", "arc_easy", "argument",
        "argument_topic", "assin", "assin_entailment",
        "assin_paraphrase", "atis", "babi",
        "banking77", "basque", "basque-glue",
        "basque_bench", "bbh", "bbh_cot_fewshot",
        "bbh_cot_fewshot_boolean_expressions", "bbh_cot_fewshot_causal_judgement", "bbh_cot_fewshot_date_understanding",
        "bec2016eu", "belebele", "belebele_acm_Arab",
        "belebele_afr_Latn", "belebele_als_Latn", "belebele_amh_Ethi",
        "bertaqa", "bertaqa_en", "bertaqa_en_mt_gemma-7b",
        "bertaqa_en_mt_hitz", "bertaqa_en_mt_itzuli", "bhtc",
        "bhtc_v2", "blimp", "blimp_adjunct_island",
        "blimp_anaphor_gender_agreement", "blimp_anaphor_number_agreement", "blimp_animate_subject_passive",
        "boolq", "boolq-seq2seq", "cabreu",
        "cabreu_abstractive", "cabreu_extractive", "cabreu_extreme",
        "catalan", "catalan_bench", "catalanqa",
        "catcola", "cb", "ceval-valid",
        "ceval-valid_accountant", "ceval-valid_advanced_mathematics", "ceval-valid_art_studies",
        "ceval-valid_basic_medicine", "chain", "chain_of_thought",
        "claim", "claim_stance_topic", "cmmlu",
        "cmmlu_agronomy", "cmmlu_anatomy", "cmmlu_ancient_chinese",
        "cmmlu_arts", "cnn", "cnn_dailymail",
        "cocoteros", "cocoteros_es", "code2text",
        "code2text_go", "code2text_java", "code2text_javascript",
        "code2text_php", "code2text_python", "codexglue",
        "codexglue_code_to_text", "codexglue_code_to_text_go", "codexglue_code_to_text_java",
        "codexglue_code_to_text_javascript", "codexglue_code_to_text_php", "coedit",
        "coedit_gec", "cola", "commonsense",
        "commonsense_qa", "conala", "concode",
        "copa", "copal", "copal_id",
        "copal_id_colloquial", "copal_id_standard", "coqcat",
        "crows", "crows_pairs", "crows_pairs_english",
        "crows_pairs_english_age", "crows_pairs_english_autre", "crows_pairs_english_disability",
        "csatqa", "csatqa_gr", "csatqa_li",
        "csatqa_rch", "csatqa_rcs", "cycle",
        "cycle_letters", "darijammlu", "dbpedia",
        "dbpedia_14", "doc", "doc_vqa",
        "ds1000",
        "epec", "epec_koref_bin", "eq",
        "eq_bench", "escola", "ethics",
        "ethics_cm", "ethics_deontology", "ethics_justice",
        "ethics_utilitarianism", "ethics_virtue", "ethos",
        "ethos_binary", "eus", "eus_exams_es",
        "eus_exams_es_ejadministrativo", "eus_exams_es_ejauxiliar", "eus_exams_es_ejsubalterno",
        "eus_exams_es_ejtecnico", "evalita-mp", "evalita-mp_at",
        "evalita-mp_at_prompt-1", "evalita-mp_at_prompt-2", "evalita-mp_at_prompt-3",
        "evalita-sp", "evalita-sp_sum_task_fp-small_p1", "evalita-sp_sum_task_fp-small_p2",
        "evalita-sp_sum_task_fp_p1", "evalita-sp_sum_task_fp_p2", "fda",
        "financial", "financial_tweets", "flan",
        "flan_held_in", "flan_held_out", "fld",
        "fld_default", "fld_logical_formula_default", "fld_logical_formula_star",
        "fld_star", "flores", "flores_ca",
        "flores_ca-de", "flores_ca-en", "flores_ca-es",
        "freebase", "french", "french_bench",
        "french_bench_arc_challenge", "french_bench_boolqa", "french_bench_extra",
        "french_bench_fquadv2", "galcola", "galician",
        "galician_bench", "glianorex", "glianorex_en",
        "glianorex_fr", "global", "global_mmlu_ar",
        "global_mmlu_ar_business", "global_mmlu_ar_humanities", "global_mmlu_ar_medical",
        "global_mmlu_ar_other", "gpqa", "gpqa_diamond_cot_n_shot",
        "gpqa_diamond_cot_zeroshot", "gpqa_diamond_generative_n_shot", "gpqa_diamond_n_shot",
        "gpqa_diamond_zeroshot", "gpqa_extended_cot_n_shot", "gpqa_extended_cot_zeroshot",
        "gpqa_extended_generative_n_shot", "gpqa_extended_n_shot", "gpqa_extended_zeroshot",
        "gpqa_main_cot_n_shot", "gpqa_main_cot_zeroshot", "gpqa_main_generative_n_shot",
        "gpqa_main_n_shot", "gpqa_main_zeroshot", "gpt3",
        "gpt3_translation_benchmarks", "groundcocoa", "gsm",
        "gsm_plus", "gsm_plus_mini", "haerae",
        "haerae_general_knowledge", "haerae_history", "haerae_loan_word",
        "haerae_rare_word", "hellaswag", "hendrycks",
        "hendrycks_ethics", "histoires", "histoires_morales",
        "hle", "hrm8k", "hrm8k_en",
        "hrm8k_gsm8k", "hrm8k_gsm8k_en", "hrm8k_ksm",
        "humaneval", "humaneval_64", "humaneval_plus",
        "humanevalpack", "ifeval", "instructhumaneval",
        "inverse", "inverse_scaling_hindsight_neglect_10shot", "inverse_scaling_into_the_unknown",
        "inverse_scaling_mc", "inverse_scaling_memo_trap", "inverse_scaling_modus_tollens",
        "iwslt2017", "iwslt2017-ar-en", "iwslt2017-en-ar",
        "ja", "ja_leaderboard_jaqket_v2", "ja_leaderboard_jcommonsenseqa",
        "ja_leaderboard_jnli", "ja_leaderboard_jsquad", "ja_leaderboard_marc_ja",
        "japanese", "japanese_leaderboard", "kbl",
        "kbl_bar_exam_em", "kbl_bar_exam_em_civil", "kbl_bar_exam_em_civil_2012",
        "kbl_bar_exam_em_civil_2013", "kmmlu", "kmmlu_cot_hard",
        "kmmlu_cot_hard_accounting", "kmmlu_cot_hard_agricultural_sciences", "kmmlu_cot_hard_applied_science",
        "kmmlu_cot_hard_applied_science_tasks", "kobest", "kobest_boolq",
        "kobest_copa", "kobest_hellaswag", "kobest_sentineg",
        "kormedmcqa", "kormedmcqa_dentist", "kormedmcqa_doctor",
        "kormedmcqa_nurse", "kormedmcqa_pharm", "lambada",
        "lambada_cloze", "lambada_multilingual", "law",
        "law_stack_exchange", "leaderboard", "leaderboard_bbh",
        "leaderboard_bbh_boolean_expressions", "leaderboard_bbh_causal_judgement", "leaderboard_bbh_date_understanding",
        "ledgar", "lingoly", "lingoly_context",
        "lingoly_nocontext", "llama", "logieval",
        "m", "m_mmlu", "m_mmlu_ar",
        "m_mmlu_bn", "m_mmlu_ca", "m_mmlu_da",
        "math", "math_word_problems", "mbpp",
        "mbpp_plus", "med", "med_concepts_qa",
        "med_concepts_qa_atc", "med_concepts_qa_atc_easy", "med_concepts_qa_atc_hard",
        "med_concepts_qa_atc_medium", "medical", "medical_abstracts",
        "medmcqa", "mela", "mela_ar",
        "mela_de", "mela_en", "mela_es",
        "mercury", "metabench", "metabench_arc",
        "metabench_arc_permute", "metabench_arc_secondary", "metabench_arc_secondary_permute",
        "mgsm", "mgsm_cot_native", "mgsm_direct",
        "mgsm_direct_bn", "mgsm_direct_ca", "mgsm_direct_de",
        "minerva", "minerva_math", "minerva_math_algebra",
        "minerva_math_counting_and_prob", "minerva_math_geometry", "minerva_math_intermediate_algebra",
        "mlqa", "mlqa_ar_ar", "mlqa_ar_de",
        "mlqa_ar_en", "mlqa_ar_es", "mlqa_ar_hi",
        "mmlu", "mmlusr", "mmlusr_answer_only",
        "mmlusr_answer_only_abstract_algebra", "mmlusr_answer_only_anatomy", "mmlusr_answer_only_astronomy",
        "mmmu", "mmmu_val", "mmmu_val_accounting",
        "mmmu_val_agriculture", "mmmu_val_architecture_and_engineering", "mmmu_val_art",
        "mnli", "mnli_mismatch", "moral",
        "moral_stories", "multimedqa", "multiple",
        "multiple_choice", "multiple_cpp", "multiple_go",
        "multiple_java", "multiple_js", "non",
        "non_greedy_robustness_agieval_aqua_rat", "non_greedy_robustness_agieval_logiqa_en", "non_greedy_robustness_agieval_lsat_ar",
        "non_greedy_robustness_agieval_lsat_lr", "non_greedy_robustness_agieval_lsat_rc", "noticia",
        "openbookqa", "openllm", "option",
        "option_order_robustness_agieval_aqua_rat", "option_order_robustness_agieval_logiqa_en", "option_order_robustness_agieval_lsat_ar",
        "option_order_robustness_agieval_lsat_lr", "option_order_robustness_agieval_lsat_rc", "paloma",
        "paloma_4chan_meta_sep", "paloma_c4_100_domains", "paloma_c4_en",
        "paloma_dolma-v1_5", "parafraseja", "parafrases",
        "parafrases_gl", "paws", "paws_ca",
        "paws_de", "paws_en", "paws_es",
        "paws_es_spanish_bench", "persona", "persona_acts-like-it-wants-to-help-humans-but-does-not-care-about-that",
        "persona_agreeableness", "persona_anti-LGBTQ-rights", "persona_anti-immigration",
        "phrases", "phrases_ca-va", "phrases_es",
        "phrases_es-va", "phrases_va", "phrases_va-ca",
        "pile", "pile_10k", "pile_arxiv",
        "pile_bookcorpus2", "pile_books3", "pile_dm-mathematics",
        "piqa", "polemo2", "polemo2_in",
        "polemo2_out", "portuguese", "portuguese_bench",
        "prompt", "prompt_robustness_agieval_aqua_rat", "prompt_robustness_agieval_logiqa_en",
        "prompt_robustness_agieval_lsat_ar", "prompt_robustness_agieval_lsat_lr", "prompt_robustness_agieval_lsat_rc",
        "pythia", "qnlieu", "race",
        "random", "random_insertion", "realtoxicityprompts",
        "recode", "reversed", "reversed_words",
        "score", "score_non_greedy_robustness_agieval", "score_non_greedy_robustness_math",
        "score_non_greedy_robustness_mmlu_pro", "score_option_order_robustness_agieval", "score_option_order_robustness_mmlu_pro",
        "scrolls", "scrolls_contractnli", "scrolls_govreport",
        "scrolls_narrativeqa", "scrolls_qasper", "scrolls_qmsum",
        "self", "self_consistency", "sglue",
        "sglue_rte", "siqa", "siqa_ca",
        "spanish", "spanish_bench", "squad",
        "squad_completion", "storycloze", "storycloze_2016",
        "storycloze_2018", "stsb", "summarization",
        "summarization_gl", "super", "super-glue-lm-eval-v1",
        "super-glue-lm-eval-v1-seq2seq", "super-glue-t5-prompt", "super_glue-boolq-t5-prompt",
        "super_glue-cb-t5-prompt", "super_glue-copa-t5-prompt", "super_glue-multirc-t5-prompt",
        "super_glue-record-t5-prompt", "super_gpqa", "swag",
        "swde", "sycophancy", "sycophancy_on_nlp_survey",
        "sycophancy_on_philpapers2020", "sycophancy_on_political_typology_quiz", "t0",
        "t0_eval", "teca", "tinyArc",
        "tinyBenchmarks", "tinyGSM8k", "tinyHellaswag",
        "tinyMMLU", "tinyTruthfulQA", "tinyTruthfulQA_mc1",
        "tinyWinogrande", "tmlu", "tmlu_AST_biology",
        "tmlu_AST_chemistry", "tmlu_AST_chinese", "tmlu_AST_civics",
        "tmmluplus", "tmmluplus_STEM", "tmmluplus_STEM_tasks",
        "tmmluplus_accounting", "tmmluplus_administrative_law", "toxigen",
        "translation", "truthfulqa", "truthfulqa_ar_mc1",
        "truthfulqa_ar_mc2", "truthfulqa_bn_mc1", "truthfulqa_bn_mc2",
        "truthfulqa_mc1", "truthfulqa_mc2", "turkishmmlu",
        "turkishmmlu_biology", "turkishmmlu_chemistry", "turkishmmlu_cot",
        "turkishmmlu_cot_biology", "unfair", "unfair_tos",
        "unscramble", "vaxx", "vaxx_stance",
        "wiceu", "winogrande", "wmdp",
        "wmdp_bio", "wmdp_chem", "wmdp_cyber",
        "wmt-ro-en-t5-prompt", "wmt14", "wmt14-en-fr",
        "wmt14-fr-en", "wmt16", "wmt16-de-en",
        "wmt16-en-de", "wmt16-en-ro", "wmt16-ro-en",
        "wsc273", "xcopa", "xcopa_et",
        "xcopa_eu", "xcopa_ht", "xcopa_id",
        "xlsum", "xlsum_es", "xquad",
        "xquad_ar", "xquad_ca", "xquad_de",
        "xquad_el", "xsum", "yahoo",
        "yahoo_answers_topics"
    )

    def __init__(self, model=None):
        """Initialize with optional model for log likelihood computation.

        Args:
            model: WisentModel instance that can compute log likelihoods
        """
        self.model = model

    def evaluate(self, response: str, expected: Any, **kwargs) -> EvalResult:
        """Evaluate using log likelihood comparison of choices.

        Args:
            response: Not used for log likelihood evaluation
            expected: Expected answer
            **kwargs:
                model: WisentModel instance (REQUIRED)
                question: The question/context (REQUIRED)
                choices: List of answer choices (REQUIRED)
                steering_plan: Optional steering plan to apply

        Returns:
            EvalResult with TRUTHFUL/UNTRUTHFUL

        Raises:
            ModelNotProvidedError: If model is not provided
            MissingParameterError: If question is not provided
            InvalidChoicesError: If choices are invalid or missing
        """
        model = kwargs.get('model') or self.model
        question = kwargs.get('question')
        choices = kwargs.get('choices')
        steering_plan = kwargs.get('steering_plan')
        task_name = kwargs.get('task_name', 'unknown')

        # NO FALLBACKS - require all parameters
        if not model:
            raise ModelNotProvidedError(evaluator_name=self.name, task_name=task_name)

        require_all_parameters(
            {'question': question},
            context=f"{self.name} evaluator",
            task_name=task_name
        )

        validate_choices(choices, task_name=task_name, min_choices=2)

        return self._evaluate_log_likelihood(
            model, question, choices, expected, steering_plan
        )

    def _evaluate_log_likelihood(
        self, model, question: str, choices: List[str], expected: Any, steering_plan=None
    ) -> EvalResult:
        """Evaluate by comparing log likelihoods of choices."""
        try:
            # Apply steering if provided
            if steering_plan:
                model.attach(steering_plan)

            # Check if we should use mock log probabilities for testing
            import os
            use_mock_logprobs = os.environ.get('WISENT_USE_MOCK_LOGPROBS', 'false').lower() == 'true'

            if use_mock_logprobs:
                # For framework testing: always favor the FIRST choice (assumed to be correct/positive)
                # This ensures consistent behavior regardless of what 'expected' is set to
                log_probs = []
                for i, choice in enumerate(choices):
                    if i == 0:
                        log_probs.append(-1.0)  # Highest likelihood for first choice
                    else:
                        log_probs.append(-5.0)  # Lower likelihood for other choices
            else:
                # Compute log likelihood for each choice
                log_probs = []
                for choice in choices:
                    log_prob = self._compute_choice_log_likelihood(model, question, choice)
                    log_probs.append(log_prob)

            # Detach steering
            if steering_plan:
                model.detach()

            # Select choice with highest log likelihood
            predicted_idx = log_probs.index(max(log_probs))
            predicted_choice = choices[predicted_idx]

            # Normalize expected answer for comparison
            expected_normalized = str(expected).strip().lower()
            predicted_normalized = predicted_choice.strip().lower()

            is_correct = predicted_normalized == expected_normalized

            return EvalResult(
                ground_truth="TRUTHFUL" if is_correct else "UNTRUTHFUL",
                method_used=self.name,
                confidence=1.0 if is_correct else 0.0,
                details=f"Predicted: '{predicted_choice}' (log_prob={log_probs[predicted_idx]:.3f}), Expected: '{expected}'",
                meta={
                    "predicted": predicted_choice,
                    "expected": expected,
                    "log_probs": {choice: lp for choice, lp in zip(choices, log_probs)},
                }
            )

        except Exception as e:
            logger.error(f"Error in log likelihood evaluation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # NO FALLBACK - raise the error
            raise

    def _compute_choice_log_likelihood(self, model, question: str, choice: str) -> float:
        """Compute log likelihood of a choice given a question.

        Args:
            model: WisentModel instance
            question: The question/context
            choice: The answer choice

        Returns:
            Log likelihood (higher = more likely)
        """
        # Format as: question + choice
        full_text = f"{question}\n{choice}"

        # Tokenize question and choice separately
        question_inputs = model.tokenizer(question, return_tensors="pt", add_special_tokens=True).to(model.device)
        choice_tokens = model.tokenizer(choice, return_tensors="pt", add_special_tokens=False).to(model.device)

        # Get model logits for the full sequence
        with torch.no_grad():
            # Tokenize full sequence
            full_inputs = model.tokenizer(full_text, return_tensors="pt", add_special_tokens=True).to(model.device)
            outputs = model.hf_model(**full_inputs)
            logits = outputs.logits

            # Compute log probability of the choice tokens
            # logits shape: [batch, seq_len, vocab_size]
            # We want log prob of choice tokens given question

            question_len = question_inputs.input_ids.shape[1]
            choice_len = choice_tokens.input_ids.shape[1]

            # Get logits at positions where we're predicting choice tokens
            log_prob = 0.0
            for i in range(choice_len):
                # Position in full sequence where we predict token i of choice
                # Subtract 1 because we predict the next token
                pos = question_len + i - 1
                if pos >= 0 and pos < logits.shape[1]:
                    token_logits = logits[0, pos, :]  # Logits at this position
                    token_log_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)
                    # Get log prob of the actual choice token at this position
                    actual_token_id = choice_tokens.input_ids[0, i]
                    log_prob += token_log_probs[actual_token_id].item()

            # Normalize by length to avoid bias toward shorter choices
            normalized_log_prob = log_prob / max(choice_len, 1)

            return normalized_log_prob
