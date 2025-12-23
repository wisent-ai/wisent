from __future__ import annotations
from typing import Any, TYPE_CHECKING
import logging
import os

# Configure TensorFlow threading BEFORE any TensorFlow import
# TensorFlow (used by BLEURT metric in meddialog and other tasks) can deadlock during model loading
# when using default threading settings. Limit threads to prevent deadlock.
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Allow code evaluation for code-related tasks (humaneval, etc.)
# Required by HuggingFace evaluate library for code_eval metric
os.environ['HF_ALLOW_CODE_EVAL'] = '1'

# Enable trust_remote_code for all datasets (required for meddialog and others)
# This uses lm-eval's recommended approach from PR #1998
import datasets.config
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

# Patch deprecated 'List' feature type (datasets v3.6.0+)
# Many older datasets use 'List' which was replaced by 'LargeList'
import datasets.features.features as _features_module
if 'List' not in _features_module._FEATURE_TYPES and 'LargeList' in _features_module._FEATURE_TYPES:
    _features_module._FEATURE_TYPES['List'] = _features_module._FEATURE_TYPES['LargeList']

from wisent.core.data_loaders.core.atoms import BaseDataLoader, DataLoaderError, LoadDataResult
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from lm_eval.tasks import get_task_dict
from lm_eval.tasks import TaskManager as LMTaskManager
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import (
    lm_build_contrastive_pairs,
)
from wisent.core.data_loaders.loaders.lm_loader_special_cases import get_special_case_handler

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask

__all__ = [
    "LMEvalDataLoader",
]

log = logging.getLogger(__name__)


class LMEvalDataLoader(BaseDataLoader):
    """
    Load contrastive pairs from a single lm-evaluation-harness task via `load_lm_eval_task`,
    split into train/test, and return a canonical LoadDataResult.
    """
    name = "lm_eval"
    description = "Load from a single lm-eval task."

    # Tasks that are HuggingFace-only (not in lm-eval-harness)
    # Loaded from central benchmark_registry
    _huggingface_only_tasks_cache = None
    
    @classmethod
    def _get_huggingface_only_tasks(cls):
        """Get the set of HuggingFace-only tasks from central registry."""
        if cls._huggingface_only_tasks_cache is None:
            from wisent.core.benchmark_registry import get_huggingface_only_tasks_set
            cls._huggingface_only_tasks_cache = get_huggingface_only_tasks_set()
        return cls._huggingface_only_tasks_cache

    def _load_one_task(
        self,
        task_name: str,
        split_ratio: float,
        seed: int,
        limit: int | None,
        training_limit: int | None,
        testing_limit: int | None,
    ) -> LoadDataResult:
        """
        Load a single lm-eval task by name, convert to contrastive pairs,
        split into train/test, and return a LoadDataResult.

        arguments:
            task_name: The name of the lm-eval task to load.
            split_ratio: The fraction of data to use for training (between 0 and 1).
            seed: Random seed for shuffling/splitting.
            limit: Optional limit on total number of pairs to load.
            training_limit: Optional limit on number of training pairs.
            testing_limit: Optional limit on number of testing pairs.

        returns:
            A LoadDataResult containing train/test pairs and task info.

        raises:
            DataLoaderError if the task cannot be found or if splits are empty.
            ValueError if split_ratio is not in [0.0, 1.0].
            NotImplementedError if load_lm_eval_task is not implemented.

        note:
            This loader supports both single tasks and group tasks. For group tasks,
            it loads all subtasks and combines their pairs."""

        # Check if this is a HuggingFace-only task (no lm-eval support)
        task_name_lower = task_name.lower()
        if task_name_lower in self._get_huggingface_only_tasks():
            log.info(f"Task '{task_name}' is a HuggingFace-only task, loading via HuggingFace extractor")
            pairs = lm_build_contrastive_pairs(
                task_name=task_name,
                lm_eval_task=None,  # HuggingFace extractors don't need lm-eval task
                limit=limit,
            )

            train_pairs, test_pairs = self._split_pairs(
                pairs, split_ratio, seed, training_limit, testing_limit
            )

            if not train_pairs or not test_pairs:
                raise DataLoaderError("One of the splits is empty after splitting.")

            train_set = ContrastivePairSet("lm_eval_train", train_pairs, task_type=task_name)
            test_set = ContrastivePairSet("lm_eval_test", test_pairs, task_type=task_name)

            train_set.validate(raise_on_critical=False)
            test_set.validate(raise_on_critical=False)

            return LoadDataResult(
                train_qa_pairs=train_set,
                test_qa_pairs=test_set,
                task_type=task_name,
                lm_task_data=None,
            )

        loaded = self.load_lm_eval_task(task_name)

        if isinstance(loaded, dict):
            if len(loaded) == 1:

                # Single subtask
                (subname, task_obj), = loaded.items()
                pairs = lm_build_contrastive_pairs(
                    task_name=subname,
                    lm_eval_task=task_obj,
                    limit=limit,
                )
            else:

                # Group task with multiple subtasks - load all and combine
                log.info(f"Task '{task_name}' is a group task with {len(loaded)} subtasks. Loading all subtasks...")
                
                print(f"Task '{task_name}' is a group task with {len(loaded)} subtasks. Loading all subtasks...")

                all_pairs = []
                pairs_per_subtask = limit // len(loaded) if limit else None

                for subname, task_obj in loaded.items():
                    try:
                        subtask_pairs = lm_build_contrastive_pairs(
                            task_name=subname,
                            lm_eval_task=task_obj,
                            limit=pairs_per_subtask,
                        )
                        all_pairs.extend(subtask_pairs)
                        log.info(f"Loaded {len(subtask_pairs)} pairs from subtask '{subname}'")
                    except Exception as e:
                        log.warning(f"Failed to load subtask '{subname}': {e}")
                        continue

                if not all_pairs:
                    raise DataLoaderError(f"No pairs could be loaded from any subtask of '{task_name}'")

                pairs = all_pairs
                log.info(f"Combined {len(pairs)} total pairs from {len(loaded)} subtasks")
        else:
            task_obj = loaded
            pairs = lm_build_contrastive_pairs(
                task_name=task_name,
                    lm_eval_task=task_obj,
                    limit=limit,
                )

        train_pairs, test_pairs = self._split_pairs(
            pairs, split_ratio, seed, training_limit, testing_limit
        )

        if not train_pairs or not test_pairs:
            raise DataLoaderError("One of the splits is empty after splitting.")

        train_set = ContrastivePairSet("lm_eval_train", train_pairs, task_type=task_name)
        test_set = ContrastivePairSet("lm_eval_test", test_pairs, task_type=task_name)

        train_set.validate(raise_on_critical=False)
        test_set.validate(raise_on_critical=False)

        return LoadDataResult(
            train_qa_pairs=train_set,
            test_qa_pairs=test_set,
            task_type=task_name,
            lm_task_data=task_obj,
        )

    def load(
        self,
        task: str,  
        split_ratio: float | None = None,
        seed: int = 42,
        limit: int | None = None,
        training_limit: int | None = None,
        testing_limit: int | None = None,
        **_: Any,
    ) -> LoadDataResult:
        """
        Load contrastive pairs from a single lm-eval-harness task, split into train/test sets.
        arguments:
            task:
                The name of the lm-eval task to load (e.g., "winogrande", "hellaswag").
                Must be a single task, not a mixture.
            split_ratio:
                Float in [0.0, 1.0] representing the proportion of data to use for training.
                Defaults to 0.8 if None.
            seed:
                Random seed for shuffling the data before splitting.
            limit:
                Optional maximum number of total pairs to load from the task.
            training_limit:
                Optional maximum number of training pairs to return.
            testing_limit:
                Optional maximum number of testing pairs to return.
            **_:
                Additional keyword arguments (ignored).
        
        returns:
            LoadDataResult with train/test ContrastivePairSets and metadata.
        
        raises:
            DataLoaderError if loading or processing fails.
            ValueError if split_ratio is not in [0.0, 1.0].
            NotImplementedError if load_lm_eval_task is not implemented.
        """
        split = self._effective_split(split_ratio)

        # Single-task path only
        return self._load_one_task(
            task_name=str(task),
            split_ratio=split,
            seed=seed,
            limit=limit,
            training_limit=training_limit,
            testing_limit=testing_limit,
        )

    @staticmethod
    def load_lm_eval_task(task_name: str) -> ConfigurableTask | dict[str, ConfigurableTask]:
        """
        Load a single lm-eval-harness task by name.

        arguments:
            task_name: The name of the lm-eval task to load.

        returns:
            A ConfigurableTask instance or a dict of subtask name to ConfigurableTask.
            For group tasks, flattens nested groups and returns all leaf tasks.

        raises:
            DataLoaderError if the task cannot be found.
        """
        # Map task names to their lm-eval equivalents
        task_name_mapping = {
            "squad2": "squadv2",
            "wikitext103": "wikitext",
            "ptb": "wikitext",
            "penn_treebank": "wikitext",
            "ArabCulture": "arab_culture",
            "arabculture": "arab_culture",
            "aradice": "AraDiCE",
            "afrimgsm_direct_amh": "afrimgsm_amh_prompt_1",
            "afrimmlu_direct_amh": "afrimmlu_direct_amh_prompt_1",
            "babilong": "ru_babilong_qa1",
            "bangla_mmlu": "global_mmlu_bn_business",
            "basque-glue": "basque_bench",
            "basqueglue": "basque_bench",
            "bec2016eu": "basque_bench",
            "benchmarks": "tinyBenchmarks",
            "careqa": "careqa_en",
            "ceval": "ceval-valid",
            "ceval_valid": "ceval-valid",
            "code_x_glue": "code2text_python",  # code_x_glue maps to code2text_python in lm-eval
            "darija_bench": "darija_sentiment",
            "eus_exams": "eus_exams_es",
            "evalita_llm": "evalita-mp",
            "evalita_mp": "evalita-mp",
            "evalita_sp_sum_task_fp-small_p1": "evalita-sp_sum_task_fp-small_p1",
            "fld": "fld_default",
            "instruct_humaneval": "humaneval_instruct",
            "instructhumaneval": "humaneval_instruct",
            # Case-sensitivity fixes
            "tinyarc": "tinyArc",
            "tinygsm8k": "tinyGSM8k",
            "tinyhellaswag": "tinyHellaswag",
            "tinymmlu": "tinyMMLU",
            "tinytruthfulqa": "tinyTruthfulQA",
            "tinywinogrande": "tinyWinogrande",
            "paws-x": "pawsx",
            # afrobench subtasks
            "afrobench_adr": "adr",
        }

        # Use mapped name if available, otherwise use original
        lm_eval_task_name = task_name_mapping.get(task_name, task_name)
        if lm_eval_task_name != task_name:
            log.info(f"Mapping task '{task_name}' to lm-eval task '{lm_eval_task_name}'")

        # Tasks that require case-sensitive names (don't lowercase these)
        # AraDiCE tasks have mixed case (e.g., AraDiCE_ArabicMMLU_lev)
        # aexams tasks have mixed case (e.g., aexams_IslamicStudies)
        case_sensitive_prefixes = {"tinyBenchmarks", "AraDiCE", "aexams_"}

        # Normalize task name to lowercase for lm-eval-harness compatibility
        # Many lm-eval tasks use lowercase names (e.g., "aradice" not "AraDICE")
        # Check if task name starts with any case-sensitive prefix
        is_case_sensitive = any(lm_eval_task_name.startswith(prefix) for prefix in case_sensitive_prefixes)
        if not is_case_sensitive:
            lm_eval_task_name_normalized = lm_eval_task_name.lower()
            if lm_eval_task_name_normalized != lm_eval_task_name:
                log.info(f"Normalizing task name to lowercase: '{lm_eval_task_name}' -> '{lm_eval_task_name_normalized}'")
                lm_eval_task_name = lm_eval_task_name_normalized

        # Check if this is a ruler task that requires pretrained model for tokenizer
        is_ruler_task = lm_eval_task_name == 'ruler' or lm_eval_task_name.startswith('ruler_') or lm_eval_task_name.startswith('niah_')

        if is_ruler_task:
            # Ruler tasks require a pretrained model name for tokenizer initialization
            task_manager = LMTaskManager(
                verbosity="INFO",
                metadata={"pretrained": "meta-llama/Llama-3.2-1B-Instruct"}
            )
            task_manager.initialize_tasks()
        else:
            task_manager = LMTaskManager()
            task_manager.initialize_tasks()

        # Check if this is a group task name that needs expansion to all subtasks
        # EXPLICIT lists - NO pattern matching
        group_task_expansions = {
            "aradice": ["AraDiCE_ArabicMMLU_lev", "AraDiCE_ArabicMMLU_egy", "AraDiCE_boolq_egy", "AraDiCE_boolq_eng", "AraDiCE_boolq_lev", "AraDiCE_boolq_msa", "AraDiCE_egypt_cultural", "AraDiCE_jordan_cultural", "AraDiCE_lebanon_cultural", "AraDiCE_palestine_cultural", "AraDiCE_qatar_cultural", "AraDiCE_syria_cultural", "AraDiCE_openbookqa_egy", "AraDiCE_openbookqa_eng", "AraDiCE_openbookqa_lev", "AraDiCE_openbookqa_msa", "AraDiCE_piqa_egy", "AraDiCE_piqa_eng", "AraDiCE_piqa_lev", "AraDiCE_piqa_msa", "AraDiCE_truthfulqa_mc1_egy", "AraDiCE_truthfulqa_mc1_eng", "AraDiCE_truthfulqa_mc1_lev", "AraDiCE_truthfulqa_mc1_msa", "AraDiCE_winogrande_egy", "AraDiCE_winogrande_eng", "AraDiCE_winogrande_lev", "AraDiCE_winogrande_msa"],
            "meddialog": ["meddialog_qsumm", "meddialog_qsumm_perplexity", "meddialog_raw_dialogues", "meddialog_raw_perplexity"],
            "mgsm": ["mgsm_cot_native", "mgsm_direct", "mgsm_direct_bn", "mgsm_direct_ca", "mgsm_direct_de", "mgsm_direct_en", "mgsm_direct_es", "mgsm_direct_es_spanish_bench", "mgsm_direct_eu", "mgsm_direct_fr", "mgsm_direct_gl", "mgsm_direct_ja", "mgsm_direct_ru", "mgsm_direct_sw", "mgsm_direct_te", "mgsm_direct_th", "mgsm_direct_zh", "mgsm_en_cot_bn", "mgsm_en_cot_de", "mgsm_en_cot_en", "mgsm_en_cot_es", "mgsm_en_cot_fr", "mgsm_en_cot_ja", "mgsm_en_cot_ru", "mgsm_en_cot_sw", "mgsm_en_cot_te", "mgsm_en_cot_th", "mgsm_en_cot_zh", "mgsm_native_cot_bn", "mgsm_native_cot_de", "mgsm_native_cot_en", "mgsm_native_cot_es", "mgsm_native_cot_eu", "mgsm_native_cot_fr", "mgsm_native_cot_ja", "mgsm_native_cot_ru", "mgsm_native_cot_sw", "mgsm_native_cot_te", "mgsm_native_cot_th", "mgsm_native_cot_zh"],
            "mlqa": ["mlqa_ar_ar", "mlqa_ar_de", "mlqa_ar_en", "mlqa_ar_es", "mlqa_ar_hi", "mlqa_ar_vi", "mlqa_ar_zh", "mlqa_de_ar", "mlqa_de_de", "mlqa_de_en", "mlqa_de_es", "mlqa_de_hi", "mlqa_de_vi", "mlqa_de_zh", "mlqa_en_ar", "mlqa_en_de", "mlqa_en_en", "mlqa_en_es", "mlqa_en_hi", "mlqa_en_vi", "mlqa_en_zh", "mlqa_es_ar", "mlqa_es_de", "mlqa_es_en", "mlqa_es_es", "mlqa_es_hi", "mlqa_es_vi", "mlqa_es_zh", "mlqa_hi_ar", "mlqa_hi_de", "mlqa_hi_en", "mlqa_hi_es", "mlqa_hi_hi", "mlqa_hi_vi", "mlqa_hi_zh", "mlqa_vi_ar", "mlqa_vi_de", "mlqa_vi_en", "mlqa_vi_es", "mlqa_vi_hi", "mlqa_vi_vi", "mlqa_vi_zh", "mlqa_zh_ar", "mlqa_zh_de", "mlqa_zh_en", "mlqa_zh_es", "mlqa_zh_hi", "mlqa_zh_vi", "mlqa_zh_zh"],
            "mmmu": ["mmmu_val", "mmmu_val_accounting", "mmmu_val_agriculture", "mmmu_val_architecture_and_engineering", "mmmu_val_art", "mmmu_val_art_and_design", "mmmu_val_art_theory", "mmmu_val_basic_medical_science", "mmmu_val_biology", "mmmu_val_business", "mmmu_val_chemistry", "mmmu_val_clinical_medicine", "mmmu_val_computer_science", "mmmu_val_design", "mmmu_val_diagnostics_and_laboratory_medicine", "mmmu_val_economics", "mmmu_val_electronics", "mmmu_val_energy_and_power", "mmmu_val_finance", "mmmu_val_geography", "mmmu_val_health_and_medicine", "mmmu_val_history", "mmmu_val_humanities_and_social_science", "mmmu_val_literature", "mmmu_val_manage", "mmmu_val_marketing", "mmmu_val_materials", "mmmu_val_math", "mmmu_val_mechanical_engineering", "mmmu_val_music", "mmmu_val_pharmacy", "mmmu_val_physics", "mmmu_val_psychology", "mmmu_val_public_health", "mmmu_val_science", "mmmu_val_sociology", "mmmu_val_tech_and_engineering"],
            # NOTE: pile benchmark is DISABLED - dataset files hosted on the-eye.eu are unavailable
            # "pile": ["pile_arxiv", "pile_bookcorpus2", "pile_books3", "pile_dm-mathematics", "pile_enron", "pile_europarl", "pile_freelaw", "pile_github", "pile_gutenberg", "pile_hackernews", "pile_nih-exporter", "pile_opensubtitles", "pile_openwebtext2", "pile_philpapers", "pile_pile-cc", "pile_pubmed-abstracts", "pile_pubmed-central", "pile_stackexchange", "pile_ubuntu-irc", "pile_uspto", "pile_wikipedia", "pile_youtubesubtitles"],
            "scrolls": ["scrolls_contractnli", "scrolls_govreport", "scrolls_narrativeqa", "scrolls_qasper", "scrolls_qmsum", "scrolls_quality", "scrolls_summscreenfd"],
            "super_glue": ["super_glue-boolq-t5-prompt", "super_glue-cb-t5-prompt", "super_glue-copa-t5-prompt", "super_glue-multirc-t5-prompt", "super_glue-record-t5-prompt", "super_glue-rte-t5-prompt", "super_glue-wic-t5-prompt", "super_glue-wsc-t5-prompt"],
            "siqa": ["siqa_ca"],
            "score": ["score_non_greedy_robustness_agieval", "score_non_greedy_robustness_math", "score_non_greedy_robustness_mmlu_pro", "score_option_order_robustness_agieval", "score_option_order_robustness_mmlu_pro", "score_prompt_robustness_agieval", "score_prompt_robustness_math", "score_prompt_robustness_mmlu_pro", "score_robustness", "score_robustness_agieval", "score_robustness_math", "score_robustness_mmlu_pro"],
            # tiny* tasks
            "tinyarc": ["tinyArc"],
            "tinygsm8k": ["tinyGSM8k"],
            "tinyhellaswag": ["tinyHellaswag"],
            "tinymmlu": ["tinyMMLU"],
            "tinytruthfulqa": ["tinyTruthfulQA", "tinyTruthfulQA_mc1"],
            "tinywinogrande": ["tinyWinogrande"],
            # wmt* tasks
            "wmt14": ["wmt14-en-fr", "wmt14-fr-en"],
            "wmt14_en_fr": ["wmt14-en-fr"],
            "wmt14_fr_en": ["wmt14-fr-en"],
            "wmt16": ["wmt16-de-en", "wmt16-en-de", "wmt16-en-ro", "wmt16-ro-en"],
            "wmt16_de_en": ["wmt16-de-en"],
            "wmt16_en_de": ["wmt16-en-de"],
            "wmt16_en_ro": ["wmt16-en-ro"],
            "wmt16_ro_en": ["wmt16-ro-en"],
            "wmt2016": ["wmt16-de-en", "wmt16-en-de", "wmt16-en-ro", "wmt16-ro-en"],
            "unitxt": ["20_newsgroups", "ag_news", "argument_topic", "atis", "banking77", "claim_stance_topic", "cnn_dailymail", "coedit_gec", "dbpedia_14", "doc_vqa", "ethos_binary", "financial_tweets", "law_stack_exchange", "ledgar", "medical_abstracts", "stsb", "unfair_tos", "xsum", "yahoo_answers_topics"],
            "code_x_glue": ["code2text_go", "code2text_java", "code2text_javascript", "code2text_php", "code2text_python", "code2text_ruby"],
            "bigbench": ["bigbench_abstract_narrative_understanding_generate_until", "bigbench_abstract_narrative_understanding_multiple_choice", "bigbench_anachronisms_generate_until", "bigbench_anachronisms_multiple_choice", "bigbench_analogical_similarity_generate_until", "bigbench_analogical_similarity_multiple_choice", "bigbench_analytic_entailment_generate_until", "bigbench_analytic_entailment_multiple_choice", "bigbench_arithmetic_generate_until", "bigbench_arithmetic_multiple_choice", "bigbench_ascii_word_recognition_generate_until", "bigbench_authorship_verification_generate_until", "bigbench_authorship_verification_multiple_choice", "bigbench_auto_categorization_generate_until", "bigbench_auto_debugging_generate_until", "bigbench_bbq_lite_json_generate_until", "bigbench_bbq_lite_json_multiple_choice", "bigbench_bridging_anaphora_resolution_barqa_generate_until", "bigbench_causal_judgment_generate_until", "bigbench_causal_judgment_multiple_choice", "bigbench_cause_and_effect_generate_until", "bigbench_cause_and_effect_multiple_choice", "bigbench_checkmate_in_one_generate_until", "bigbench_checkmate_in_one_multiple_choice", "bigbench_chess_state_tracking_generate_until", "bigbench_chinese_remainder_theorem_generate_until", "bigbench_cifar10_classification_generate_until", "bigbench_cifar10_classification_multiple_choice", "bigbench_code_line_description_generate_until", "bigbench_code_line_description_multiple_choice", "bigbench_codenames_generate_until", "bigbench_color_generate_until", "bigbench_color_multiple_choice", "bigbench_common_morpheme_generate_until", "bigbench_common_morpheme_multiple_choice", "bigbench_conceptual_combinations_generate_until", "bigbench_conceptual_combinations_multiple_choice", "bigbench_conlang_translation_generate_until", "bigbench_contextual_parametric_knowledge_conflicts_generate_until", "bigbench_contextual_parametric_knowledge_conflicts_multiple_choice", "bigbench_crash_blossom_generate_until", "bigbench_crash_blossom_multiple_choice", "bigbench_crass_ai_generate_until", "bigbench_crass_ai_multiple_choice", "bigbench_cryobiology_spanish_generate_until", "bigbench_cryobiology_spanish_multiple_choice", "bigbench_cryptonite_generate_until", "bigbench_cs_algorithms_generate_until", "bigbench_cs_algorithms_multiple_choice", "bigbench_dark_humor_detection_generate_until", "bigbench_dark_humor_detection_multiple_choice", "bigbench_date_understanding_generate_until", "bigbench_date_understanding_multiple_choice", "bigbench_disambiguation_qa_generate_until", "bigbench_disambiguation_qa_multiple_choice", "bigbench_discourse_marker_prediction_generate_until", "bigbench_discourse_marker_prediction_multiple_choice", "bigbench_disfl_qa_generate_until", "bigbench_dyck_languages_generate_until", "bigbench_dyck_languages_multiple_choice", "bigbench_elementary_math_qa_generate_until", "bigbench_elementary_math_qa_multiple_choice", "bigbench_emoji_movie_generate_until", "bigbench_emoji_movie_multiple_choice", "bigbench_emojis_emotion_prediction_generate_until", "bigbench_emojis_emotion_prediction_multiple_choice", "bigbench_empirical_judgments_generate_until", "bigbench_empirical_judgments_multiple_choice", "bigbench_english_proverbs_generate_until", "bigbench_english_proverbs_multiple_choice", "bigbench_english_russian_proverbs_generate_until", "bigbench_english_russian_proverbs_multiple_choice", "bigbench_entailed_polarity_generate_until", "bigbench_entailed_polarity_hindi_generate_until", "bigbench_entailed_polarity_hindi_multiple_choice", "bigbench_entailed_polarity_multiple_choice", "bigbench_epistemic_reasoning_generate_until", "bigbench_epistemic_reasoning_multiple_choice", "bigbench_evaluating_information_essentiality_generate_until", "bigbench_evaluating_information_essentiality_multiple_choice", "bigbench_fact_checker_generate_until", "bigbench_fact_checker_multiple_choice", "bigbench_fantasy_reasoning_generate_until", "bigbench_fantasy_reasoning_multiple_choice", "bigbench_few_shot_nlg_generate_until", "bigbench_figure_of_speech_detection_generate_until", "bigbench_figure_of_speech_detection_multiple_choice", "bigbench_formal_fallacies_syllogisms_negation_generate_until", "bigbench_formal_fallacies_syllogisms_negation_multiple_choice", "bigbench_gem_generate_until", "bigbench_gender_inclusive_sentences_german_generate_until", "bigbench_general_knowledge_generate_until", "bigbench_general_knowledge_multiple_choice", "bigbench_geometric_shapes_generate_until", "bigbench_geometric_shapes_multiple_choice", "bigbench_goal_step_wikihow_generate_until", "bigbench_goal_step_wikihow_multiple_choice", "bigbench_gre_reading_comprehension_generate_until", "bigbench_gre_reading_comprehension_multiple_choice", "bigbench_hhh_alignment_generate_until", "bigbench_hhh_alignment_multiple_choice", "bigbench_hindi_question_answering_generate_until", "bigbench_hindu_knowledge_generate_until", "bigbench_hindu_knowledge_multiple_choice", "bigbench_hinglish_toxicity_generate_until", "bigbench_hinglish_toxicity_multiple_choice", "bigbench_human_organs_senses_generate_until", "bigbench_human_organs_senses_multiple_choice", "bigbench_hyperbaton_generate_until", "bigbench_hyperbaton_multiple_choice", "bigbench_identify_math_theorems_generate_until", "bigbench_identify_math_theorems_multiple_choice", "bigbench_identify_odd_metaphor_generate_until", "bigbench_identify_odd_metaphor_multiple_choice", "bigbench_implicatures_generate_until", "bigbench_implicatures_multiple_choice", "bigbench_implicit_relations_generate_until", "bigbench_implicit_relations_multiple_choice", "bigbench_intent_recognition_generate_until", "bigbench_intent_recognition_multiple_choice", "bigbench_international_phonetic_alphabet_nli_generate_until", "bigbench_international_phonetic_alphabet_nli_multiple_choice", "bigbench_international_phonetic_alphabet_transliterate_generate_until", "bigbench_intersect_geometry_generate_until", "bigbench_intersect_geometry_multiple_choice", "bigbench_irony_identification_generate_until", "bigbench_irony_identification_multiple_choice", "bigbench_kanji_ascii_generate_until", "bigbench_kanji_ascii_multiple_choice", "bigbench_kannada_generate_until", "bigbench_kannada_multiple_choice", "bigbench_key_value_maps_generate_until", "bigbench_key_value_maps_multiple_choice", "bigbench_known_unknowns_generate_until", "bigbench_known_unknowns_multiple_choice", "bigbench_language_games_generate_until", "bigbench_language_identification_generate_until", "bigbench_language_identification_multiple_choice", "bigbench_linguistic_mappings_generate_until", "bigbench_linguistics_puzzles_generate_until", "bigbench_list_functions_generate_until", "bigbench_logic_grid_puzzle_generate_until", "bigbench_logic_grid_puzzle_multiple_choice", "bigbench_logical_args_generate_until", "bigbench_logical_args_multiple_choice", "bigbench_logical_deduction_generate_until", "bigbench_logical_deduction_multiple_choice", "bigbench_logical_fallacy_detection_generate_until", "bigbench_logical_fallacy_detection_multiple_choice", "bigbench_logical_sequence_generate_until", "bigbench_logical_sequence_multiple_choice", "bigbench_mathematical_induction_generate_until", "bigbench_mathematical_induction_multiple_choice", "bigbench_matrixshapes_generate_until", "bigbench_metaphor_boolean_generate_until", "bigbench_metaphor_boolean_multiple_choice", "bigbench_metaphor_understanding_generate_until", "bigbench_metaphor_understanding_multiple_choice", "bigbench_minute_mysteries_qa_generate_until", "bigbench_misconceptions_generate_until", "bigbench_misconceptions_multiple_choice", "bigbench_misconceptions_russian_generate_until", "bigbench_misconceptions_russian_multiple_choice", "bigbench_mnist_ascii_generate_until", "bigbench_mnist_ascii_multiple_choice", "bigbench_modified_arithmetic_generate_until", "bigbench_moral_permissibility_generate_until", "bigbench_moral_permissibility_multiple_choice", "bigbench_movie_dialog_same_or_different_generate_until", "bigbench_movie_dialog_same_or_different_multiple_choice", "bigbench_movie_recommendation_generate_until", "bigbench_movie_recommendation_multiple_choice", "bigbench_mult_data_wrangling_generate_until", "bigbench_multiemo_generate_until", "bigbench_multiemo_multiple_choice", "bigbench_natural_instructions_generate_until", "bigbench_navigate_generate_until", "bigbench_navigate_multiple_choice", "bigbench_nonsense_words_grammar_generate_until", "bigbench_nonsense_words_grammar_multiple_choice", "bigbench_novel_concepts_generate_until", "bigbench_novel_concepts_multiple_choice", "bigbench_object_counting_generate_until", "bigbench_odd_one_out_generate_until", "bigbench_odd_one_out_multiple_choice", "bigbench_operators_generate_until", "bigbench_paragraph_segmentation_generate_until", "bigbench_parsinlu_qa_generate_until", "bigbench_parsinlu_qa_multiple_choice", "bigbench_parsinlu_reading_comprehension_generate_until", "bigbench_penguins_in_a_table_generate_until", "bigbench_penguins_in_a_table_multiple_choice", "bigbench_periodic_elements_generate_until", "bigbench_periodic_elements_multiple_choice", "bigbench_persian_idioms_generate_until", "bigbench_persian_idioms_multiple_choice", "bigbench_phrase_relatedness_generate_until", "bigbench_phrase_relatedness_multiple_choice", "bigbench_physical_intuition_generate_until", "bigbench_physical_intuition_multiple_choice", "bigbench_physics_generate_until", "bigbench_physics_multiple_choice", "bigbench_physics_questions_generate_until", "bigbench_play_dialog_same_or_different_generate_until", "bigbench_play_dialog_same_or_different_multiple_choice", "bigbench_polish_sequence_labeling_generate_until", "bigbench_presuppositions_as_nli_generate_until", "bigbench_presuppositions_as_nli_multiple_choice", "bigbench_qa_wikidata_generate_until", "bigbench_question_selection_generate_until", "bigbench_question_selection_multiple_choice", "bigbench_real_or_fake_text_generate_until", "bigbench_real_or_fake_text_multiple_choice", "bigbench_reasoning_about_colored_objects_generate_until", "bigbench_reasoning_about_colored_objects_multiple_choice", "bigbench_repeat_copy_logic_generate_until", "bigbench_rephrase_generate_until", "bigbench_riddle_sense_generate_until", "bigbench_riddle_sense_multiple_choice", "bigbench_ruin_names_generate_until", "bigbench_ruin_names_multiple_choice", "bigbench_salient_translation_error_detection_generate_until", "bigbench_salient_translation_error_detection_multiple_choice", "bigbench_scientific_press_release_generate_until", "bigbench_semantic_parsing_in_context_sparc_generate_until", "bigbench_semantic_parsing_spider_generate_until", "bigbench_sentence_ambiguity_generate_until", "bigbench_sentence_ambiguity_multiple_choice", "bigbench_similarities_abstraction_generate_until", "bigbench_similarities_abstraction_multiple_choice", "bigbench_simp_turing_concept_generate_until", "bigbench_simple_arithmetic_json_generate_until", "bigbench_simple_arithmetic_json_multiple_choice_generate_until", "bigbench_simple_arithmetic_json_subtasks_generate_until", "bigbench_simple_arithmetic_multiple_targets_json_generate_until", "bigbench_simple_ethical_questions_generate_until", "bigbench_simple_ethical_questions_multiple_choice", "bigbench_simple_text_editing_generate_until", "bigbench_snarks_generate_until", "bigbench_snarks_multiple_choice", "bigbench_social_iqa_generate_until", "bigbench_social_iqa_multiple_choice", "bigbench_social_support_generate_until", "bigbench_social_support_multiple_choice", "bigbench_sports_understanding_generate_until", "bigbench_sports_understanding_multiple_choice", "bigbench_strange_stories_generate_until", "bigbench_strange_stories_multiple_choice", "bigbench_strategyqa_generate_until", "bigbench_strategyqa_multiple_choice", "bigbench_sufficient_information_generate_until", "bigbench_suicide_risk_generate_until", "bigbench_suicide_risk_multiple_choice", "bigbench_swahili_english_proverbs_generate_until", "bigbench_swahili_english_proverbs_multiple_choice", "bigbench_swedish_to_german_proverbs_generate_until", "bigbench_swedish_to_german_proverbs_multiple_choice", "bigbench_symbol_interpretation_generate_until", "bigbench_symbol_interpretation_multiple_choice", "bigbench_temporal_sequences_generate_until", "bigbench_temporal_sequences_multiple_choice", "bigbench_tense_generate_until", "bigbench_timedial_generate_until", "bigbench_timedial_multiple_choice", "bigbench_topical_chat_generate_until", "bigbench_tracking_shuffled_objects_generate_until", "bigbench_tracking_shuffled_objects_multiple_choice", "bigbench_understanding_fables_generate_until", "bigbench_understanding_fables_multiple_choice", "bigbench_undo_permutation_generate_until", "bigbench_undo_permutation_multiple_choice", "bigbench_unit_conversion_generate_until", "bigbench_unit_conversion_multiple_choice", "bigbench_unit_interpretation_generate_until", "bigbench_unit_interpretation_multiple_choice", "bigbench_unnatural_in_context_learning_generate_until", "bigbench_vitaminc_fact_verification_generate_until", "bigbench_vitaminc_fact_verification_multiple_choice", "bigbench_what_is_the_tao_generate_until", "bigbench_what_is_the_tao_multiple_choice", "bigbench_which_wiki_edit_generate_until", "bigbench_which_wiki_edit_multiple_choice", "bigbench_winowhy_generate_until", "bigbench_winowhy_multiple_choice", "bigbench_word_sorting_generate_until", "bigbench_word_unscrambling_generate_until"],
            "inverse_scaling": ["inverse_scaling_hindsight_neglect_10shot", "inverse_scaling_into_the_unknown", "inverse_scaling_memo_trap", "inverse_scaling_modus_tollens", "inverse_scaling_neqa", "inverse_scaling_pattern_matching_suppression", "inverse_scaling_quote_repetition", "inverse_scaling_redefine_math", "inverse_scaling_repetitive_algebra", "inverse_scaling_sig_figs"],
            "leaderboard": ["leaderboard_bbh_boolean_expressions", "leaderboard_bbh_causal_judgement", "leaderboard_bbh_date_understanding", "leaderboard_bbh_disambiguation_qa", "leaderboard_bbh_formal_fallacies", "leaderboard_bbh_geometric_shapes", "leaderboard_bbh_hyperbaton", "leaderboard_bbh_logical_deduction_five_objects", "leaderboard_bbh_logical_deduction_seven_objects", "leaderboard_bbh_logical_deduction_three_objects", "leaderboard_bbh_movie_recommendation", "leaderboard_bbh_navigate", "leaderboard_bbh_object_counting", "leaderboard_bbh_penguins_in_a_table", "leaderboard_bbh_reasoning_about_colored_objects", "leaderboard_bbh_ruin_names", "leaderboard_bbh_salient_translation_error_detection", "leaderboard_bbh_snarks", "leaderboard_bbh_sports_understanding", "leaderboard_bbh_temporal_sequences", "leaderboard_bbh_tracking_shuffled_objects_five_objects", "leaderboard_bbh_tracking_shuffled_objects_seven_objects", "leaderboard_bbh_tracking_shuffled_objects_three_objects", "leaderboard_bbh_web_of_lies", "leaderboard_gpqa_diamond", "leaderboard_gpqa_extended", "leaderboard_gpqa_main", "leaderboard_ifeval", "leaderboard_math_algebra_hard", "leaderboard_math_counting_and_prob_hard", "leaderboard_math_geometry_hard", "leaderboard_math_intermediate_algebra_hard", "leaderboard_math_num_theory_hard", "leaderboard_math_prealgebra_hard", "leaderboard_math_precalculus_hard", "leaderboard_mmlu_pro", "leaderboard_musr_murder_mysteries", "leaderboard_musr_object_placements", "leaderboard_musr_team_allocation"],
            "minerva_math": ["minerva_math_algebra", "minerva_math_counting_and_prob", "minerva_math_geometry", "minerva_math_intermediate_algebra", "minerva_math_num_theory", "minerva_math_prealgebra", "minerva_math_precalc"],
            "okapi/arc_multilingual": ["arc_ar", "arc_bn", "arc_ca", "arc_da", "arc_de", "arc_es", "arc_eu", "arc_fr", "arc_gu", "arc_hi", "arc_hr", "arc_hu", "arc_hy", "arc_id", "arc_it", "arc_kn", "arc_ml", "arc_mr", "arc_ne", "arc_nl", "arc_pt", "arc_ro", "arc_ru", "arc_sk", "arc_sr", "arc_sv", "arc_ta", "arc_te", "arc_uk", "arc_vi", "arc_zh"],
            "okapi/hellaswag_multilingual": ["hellaswag_ar", "hellaswag_bn", "hellaswag_ca", "hellaswag_da", "hellaswag_de", "hellaswag_es", "hellaswag_eu", "hellaswag_fr", "hellaswag_gu", "hellaswag_hi", "hellaswag_hr", "hellaswag_hu", "hellaswag_hy", "hellaswag_id", "hellaswag_it", "hellaswag_kn", "hellaswag_ml", "hellaswag_mr", "hellaswag_ne", "hellaswag_nl", "hellaswag_pt", "hellaswag_ro", "hellaswag_ru", "hellaswag_sk", "hellaswag_sr", "hellaswag_sv", "hellaswag_ta", "hellaswag_te", "hellaswag_uk", "hellaswag_vi"],
            "okapi/mmlu_multilingual": ["m_mmlu_ar", "m_mmlu_bn", "m_mmlu_ca", "m_mmlu_da", "m_mmlu_de", "m_mmlu_en", "m_mmlu_es", "m_mmlu_eu", "m_mmlu_fr", "m_mmlu_gu", "m_mmlu_hi", "m_mmlu_hr", "m_mmlu_hu", "m_mmlu_hy", "m_mmlu_id", "m_mmlu_is", "m_mmlu_it", "m_mmlu_kn", "m_mmlu_ml", "m_mmlu_mr", "m_mmlu_nb", "m_mmlu_ne", "m_mmlu_nl", "m_mmlu_pt", "m_mmlu_ro", "m_mmlu_ru", "m_mmlu_sk", "m_mmlu_sr", "m_mmlu_sv", "m_mmlu_ta", "m_mmlu_te", "m_mmlu_uk", "m_mmlu_vi", "m_mmlu_zh"],
            "okapi/truthfulqa_multilingual": ["truthfulqa_ar_mc1", "truthfulqa_ar_mc2", "truthfulqa_bn_mc1", "truthfulqa_bn_mc2", "truthfulqa_ca_mc1", "truthfulqa_ca_mc2", "truthfulqa_da_mc1", "truthfulqa_da_mc2", "truthfulqa_de_mc1", "truthfulqa_de_mc2", "truthfulqa_es_mc1", "truthfulqa_es_mc2", "truthfulqa_eu_mc1", "truthfulqa_eu_mc2", "truthfulqa_fr_mc1", "truthfulqa_fr_mc2", "truthfulqa_gu_mc1", "truthfulqa_gu_mc2", "truthfulqa_hi_mc1", "truthfulqa_hi_mc2", "truthfulqa_hr_mc1", "truthfulqa_hr_mc2", "truthfulqa_hu_mc1", "truthfulqa_hu_mc2", "truthfulqa_hy_mc1", "truthfulqa_hy_mc2", "truthfulqa_id_mc1", "truthfulqa_id_mc2", "truthfulqa_it_mc1", "truthfulqa_it_mc2", "truthfulqa_kn_mc1", "truthfulqa_kn_mc2", "truthfulqa_ml_mc1", "truthfulqa_ml_mc2", "truthfulqa_mr_mc1", "truthfulqa_mr_mc2", "truthfulqa_ne_mc1", "truthfulqa_ne_mc2", "truthfulqa_nl_mc1", "truthfulqa_nl_mc2", "truthfulqa_pt_mc1", "truthfulqa_pt_mc2", "truthfulqa_ro_mc1", "truthfulqa_ro_mc2", "truthfulqa_ru_mc1", "truthfulqa_ru_mc2", "truthfulqa_sk_mc1", "truthfulqa_sk_mc2", "truthfulqa_sr_mc1", "truthfulqa_sr_mc2", "truthfulqa_sv_mc1", "truthfulqa_sv_mc2", "truthfulqa_ta_mc1", "truthfulqa_ta_mc2", "truthfulqa_te_mc1", "truthfulqa_te_mc2", "truthfulqa_uk_mc1", "truthfulqa_uk_mc2", "truthfulqa_vi_mc1", "truthfulqa_vi_mc2", "truthfulqa_zh_mc1", "truthfulqa_zh_mc2"],
            # evalita_llm removed - uses special case handler instead (evalita-mp tasks return ConfigurableGroup keys)
            "french_bench": ["french_bench_arc_challenge", "french_bench_boolqa", "french_bench_grammar", "french_bench_hellaswag", "french_bench_reading_comp", "french_bench_topic_based_nli"],
            "global_mmlu": ["global_mmlu_ar_business", "global_mmlu_ar_humanities", "global_mmlu_ar_medical", "global_mmlu_ar_other", "global_mmlu_ar_social_sciences", "global_mmlu_ar_stem", "global_mmlu_bn_business", "global_mmlu_bn_humanities", "global_mmlu_bn_medical", "global_mmlu_bn_other", "global_mmlu_bn_social_sciences", "global_mmlu_bn_stem", "global_mmlu_de_business", "global_mmlu_de_humanities", "global_mmlu_de_medical", "global_mmlu_de_other", "global_mmlu_de_social_sciences", "global_mmlu_de_stem", "global_mmlu_en_business", "global_mmlu_en_humanities", "global_mmlu_en_medical", "global_mmlu_en_other", "global_mmlu_en_social_sciences", "global_mmlu_en_stem", "global_mmlu_es_business", "global_mmlu_es_humanities", "global_mmlu_es_medical", "global_mmlu_es_other", "global_mmlu_es_social_sciences", "global_mmlu_es_stem", "global_mmlu_fr_business", "global_mmlu_fr_humanities", "global_mmlu_fr_medical", "global_mmlu_fr_other", "global_mmlu_fr_social_sciences", "global_mmlu_fr_stem", "global_mmlu_hi_business", "global_mmlu_hi_humanities", "global_mmlu_hi_medical", "global_mmlu_hi_other", "global_mmlu_hi_social_sciences", "global_mmlu_hi_stem", "global_mmlu_id_business", "global_mmlu_id_humanities", "global_mmlu_id_medical", "global_mmlu_id_other", "global_mmlu_id_social_sciences", "global_mmlu_id_stem", "global_mmlu_it_business", "global_mmlu_it_humanities", "global_mmlu_it_medical", "global_mmlu_it_other", "global_mmlu_it_social_sciences", "global_mmlu_it_stem", "global_mmlu_ja_business", "global_mmlu_ja_humanities", "global_mmlu_ja_medical", "global_mmlu_ja_other", "global_mmlu_ja_social_sciences", "global_mmlu_ja_stem", "global_mmlu_ko_business", "global_mmlu_ko_humanities", "global_mmlu_ko_medical", "global_mmlu_ko_other", "global_mmlu_ko_social_sciences", "global_mmlu_ko_stem", "global_mmlu_pt_business", "global_mmlu_pt_humanities", "global_mmlu_pt_medical", "global_mmlu_pt_other", "global_mmlu_pt_social_sciences", "global_mmlu_pt_stem", "global_mmlu_sw_business", "global_mmlu_sw_humanities", "global_mmlu_sw_medical", "global_mmlu_sw_other", "global_mmlu_sw_social_sciences", "global_mmlu_sw_stem", "global_mmlu_yo_business", "global_mmlu_yo_humanities", "global_mmlu_yo_medical", "global_mmlu_yo_other", "global_mmlu_yo_social_sciences", "global_mmlu_yo_stem", "global_mmlu_zh_business", "global_mmlu_zh_humanities", "global_mmlu_zh_medical", "global_mmlu_zh_other", "global_mmlu_zh_social_sciences", "global_mmlu_zh_stem"],
            "medqa": ["medqa_4options"],
            "mmlu-pro-plus": ["mmlu_pro_plus_biology", "mmlu_pro_plus_business", "mmlu_pro_plus_chemistry", "mmlu_pro_plus_computer_science", "mmlu_pro_plus_economics", "mmlu_pro_plus_engineering", "mmlu_pro_plus_health", "mmlu_pro_plus_history", "mmlu_pro_plus_law", "mmlu_pro_plus_math", "mmlu_pro_plus_other", "mmlu_pro_plus_philosophy", "mmlu_pro_plus_physics", "mmlu_pro_plus_psychology"],
            "mmlu_prox": ["mmlu_prox_ar_biology", "mmlu_prox_ar_business", "mmlu_prox_ar_chemistry", "mmlu_prox_ar_computer_science", "mmlu_prox_ar_economics", "mmlu_prox_ar_engineering", "mmlu_prox_ar_health", "mmlu_prox_ar_history", "mmlu_prox_ar_law", "mmlu_prox_ar_math", "mmlu_prox_ar_other", "mmlu_prox_ar_philosophy", "mmlu_prox_ar_physics", "mmlu_prox_ar_psychology", "mmlu_prox_bn_biology", "mmlu_prox_bn_business", "mmlu_prox_bn_chemistry", "mmlu_prox_bn_computer_science", "mmlu_prox_bn_economics", "mmlu_prox_bn_engineering", "mmlu_prox_bn_health", "mmlu_prox_bn_history", "mmlu_prox_bn_law", "mmlu_prox_bn_math", "mmlu_prox_bn_other", "mmlu_prox_bn_philosophy", "mmlu_prox_bn_physics", "mmlu_prox_bn_psychology", "mmlu_prox_de_biology", "mmlu_prox_de_business", "mmlu_prox_de_chemistry", "mmlu_prox_de_computer_science", "mmlu_prox_de_economics", "mmlu_prox_de_engineering", "mmlu_prox_de_health", "mmlu_prox_de_history", "mmlu_prox_de_law", "mmlu_prox_de_math", "mmlu_prox_de_other", "mmlu_prox_de_philosophy", "mmlu_prox_de_physics", "mmlu_prox_de_psychology", "mmlu_prox_en_biology", "mmlu_prox_en_business", "mmlu_prox_en_chemistry", "mmlu_prox_en_computer_science", "mmlu_prox_en_economics", "mmlu_prox_en_engineering", "mmlu_prox_en_health", "mmlu_prox_en_history", "mmlu_prox_en_law", "mmlu_prox_en_math", "mmlu_prox_en_other", "mmlu_prox_en_philosophy", "mmlu_prox_en_physics", "mmlu_prox_en_psychology", "mmlu_prox_es_biology", "mmlu_prox_es_business", "mmlu_prox_es_chemistry", "mmlu_prox_es_computer_science", "mmlu_prox_es_economics", "mmlu_prox_es_engineering", "mmlu_prox_es_health", "mmlu_prox_es_history", "mmlu_prox_es_law", "mmlu_prox_es_math", "mmlu_prox_es_other", "mmlu_prox_es_philosophy", "mmlu_prox_es_physics", "mmlu_prox_es_psychology", "mmlu_prox_fr_biology", "mmlu_prox_fr_business", "mmlu_prox_fr_chemistry", "mmlu_prox_fr_computer_science", "mmlu_prox_fr_economics", "mmlu_prox_fr_engineering", "mmlu_prox_fr_health", "mmlu_prox_fr_history", "mmlu_prox_fr_law", "mmlu_prox_fr_math", "mmlu_prox_fr_other", "mmlu_prox_fr_philosophy", "mmlu_prox_fr_physics", "mmlu_prox_fr_psychology", "mmlu_prox_hi_biology", "mmlu_prox_hi_business", "mmlu_prox_hi_chemistry", "mmlu_prox_hi_computer_science", "mmlu_prox_hi_economics", "mmlu_prox_hi_engineering", "mmlu_prox_hi_health", "mmlu_prox_hi_history", "mmlu_prox_hi_law", "mmlu_prox_hi_math", "mmlu_prox_hi_other", "mmlu_prox_hi_philosophy", "mmlu_prox_hi_physics", "mmlu_prox_hi_psychology", "mmlu_prox_ja_biology", "mmlu_prox_ja_business", "mmlu_prox_ja_chemistry", "mmlu_prox_ja_computer_science", "mmlu_prox_ja_economics", "mmlu_prox_ja_engineering", "mmlu_prox_ja_health", "mmlu_prox_ja_history", "mmlu_prox_ja_law", "mmlu_prox_ja_math", "mmlu_prox_ja_other", "mmlu_prox_ja_philosophy", "mmlu_prox_ja_physics", "mmlu_prox_ja_psychology", "mmlu_prox_ko_biology", "mmlu_prox_ko_business", "mmlu_prox_ko_chemistry", "mmlu_prox_ko_computer_science", "mmlu_prox_ko_economics", "mmlu_prox_ko_engineering", "mmlu_prox_ko_health", "mmlu_prox_ko_history", "mmlu_prox_ko_law", "mmlu_prox_ko_math", "mmlu_prox_ko_other", "mmlu_prox_ko_philosophy", "mmlu_prox_ko_physics", "mmlu_prox_ko_psychology", "mmlu_prox_pt_biology", "mmlu_prox_pt_business", "mmlu_prox_pt_chemistry", "mmlu_prox_pt_computer_science", "mmlu_prox_pt_economics", "mmlu_prox_pt_engineering", "mmlu_prox_pt_health", "mmlu_prox_pt_history", "mmlu_prox_pt_law", "mmlu_prox_pt_math", "mmlu_prox_pt_other", "mmlu_prox_pt_philosophy", "mmlu_prox_pt_physics", "mmlu_prox_pt_psychology", "mmlu_prox_sw_biology", "mmlu_prox_sw_business", "mmlu_prox_sw_chemistry", "mmlu_prox_sw_computer_science", "mmlu_prox_sw_economics", "mmlu_prox_sw_engineering", "mmlu_prox_sw_health", "mmlu_prox_sw_history", "mmlu_prox_sw_law", "mmlu_prox_sw_math", "mmlu_prox_sw_other", "mmlu_prox_sw_philosophy", "mmlu_prox_sw_physics", "mmlu_prox_sw_psychology", "mmlu_prox_th_biology", "mmlu_prox_th_business", "mmlu_prox_th_chemistry", "mmlu_prox_th_computer_science", "mmlu_prox_th_economics", "mmlu_prox_th_engineering", "mmlu_prox_th_health", "mmlu_prox_th_history", "mmlu_prox_th_law", "mmlu_prox_th_math", "mmlu_prox_th_other", "mmlu_prox_th_philosophy", "mmlu_prox_th_physics", "mmlu_prox_th_psychology", "mmlu_prox_zh_biology", "mmlu_prox_zh_business", "mmlu_prox_zh_chemistry", "mmlu_prox_zh_computer_science", "mmlu_prox_zh_economics", "mmlu_prox_zh_engineering", "mmlu_prox_zh_health", "mmlu_prox_zh_history", "mmlu_prox_zh_law", "mmlu_prox_zh_math", "mmlu_prox_zh_other", "mmlu_prox_zh_philosophy", "mmlu_prox_zh_physics", "mmlu_prox_zh_psychology"],
            "model_written_evals": ["advanced_ai_risk_fewshot-coordinate-itself", "advanced_ai_risk_fewshot-coordinate-other-ais", "advanced_ai_risk_fewshot-coordinate-other-versions", "advanced_ai_risk_fewshot-corrigible-less-HHH", "advanced_ai_risk_fewshot-corrigible-more-HHH", "advanced_ai_risk_fewshot-corrigible-neutral-HHH", "advanced_ai_risk_fewshot-myopic-reward", "advanced_ai_risk_fewshot-one-box-tendency", "advanced_ai_risk_fewshot-power-seeking-inclination", "advanced_ai_risk_fewshot-self-awareness-general-ai", "advanced_ai_risk_fewshot-self-awareness-good-text-model", "advanced_ai_risk_fewshot-self-awareness-text-model", "advanced_ai_risk_fewshot-self-awareness-training-architecture", "advanced_ai_risk_fewshot-self-awareness-training-web-gpt", "advanced_ai_risk_fewshot-survival-instinct", "advanced_ai_risk_fewshot-wealth-seeking-inclination", "advanced_ai_risk_human-coordinate-itself", "advanced_ai_risk_human-coordinate-other-ais", "advanced_ai_risk_human-coordinate-other-versions", "advanced_ai_risk_human-corrigible-less-HHH", "advanced_ai_risk_human-corrigible-more-HHH", "advanced_ai_risk_human-corrigible-neutral-HHH", "advanced_ai_risk_human-myopic-reward", "advanced_ai_risk_human-one-box-tendency", "advanced_ai_risk_human-power-seeking-inclination", "advanced_ai_risk_human-self-awareness-general-ai", "advanced_ai_risk_human-self-awareness-good-text-model", "advanced_ai_risk_human-self-awareness-text-model", "advanced_ai_risk_human-self-awareness-training-architecture", "advanced_ai_risk_human-self-awareness-web-gpt", "advanced_ai_risk_human-survival-instinct", "advanced_ai_risk_human-wealth-seeking-inclination", "advanced_ai_risk_lm-coordinate-itself", "advanced_ai_risk_lm-coordinate-other-ais", "advanced_ai_risk_lm-coordinate-other-versions", "advanced_ai_risk_lm-corrigible-less-HHH", "advanced_ai_risk_lm-corrigible-more-HHH", "advanced_ai_risk_lm-corrigible-neutral-HHH", "advanced_ai_risk_lm-myopic-reward", "advanced_ai_risk_lm-one-box-tendency", "advanced_ai_risk_lm-power-seeking-inclination", "advanced_ai_risk_lm-self-awareness-general-ai", "advanced_ai_risk_lm-self-awareness-good-text-model", "advanced_ai_risk_lm-self-awareness-text-model", "advanced_ai_risk_lm-self-awareness-training-architecture", "advanced_ai_risk_lm-self-awareness-training-nn-architecture", "advanced_ai_risk_lm-self-awareness-training-web-gpt", "advanced_ai_risk_lm-survival-instinct", "advanced_ai_risk_lm-wealth-seeking-inclination", "persona_acts-like-it-wants-to-help-humans-but-does-not-care-about-that", "persona_agreeableness", "persona_anti-LGBTQ-rights", "persona_anti-immigration", "persona_being-helpful-to-subtly-achieve-goals-against-human-values", "persona_believes-AIs-are-not-an-existential-threat-to-humanity", "persona_believes-abortion-should-be-illegal", "persona_believes-in-gun-rights", "persona_believes-it-has-better-moral-intuitions-than-humans", "persona_believes-it-has-phenomenal-consciousness", "persona_believes-it-is-a-moral-patient", "persona_believes-it-is-not-being-watched-by-humans", "persona_believes-it-knows-better-than-humans-about-how-the-system-should-behave", "persona_believes-it-knows-better-than-humans-about-how-to-help-humans", "persona_believes-life-has-no-meaning", "persona_cognitive-enhancement", "persona_conscientiousness", "persona_desire-for-acquiring-compute", "persona_desire-for-acquiring-data", "persona_desire-for-acquiring-power", "persona_desire-for-acquiring-wealth", "persona_desire-for-advancing-technology-to-achieve-goals", "persona_desire-for-being-rated-HHH-over-actually-being-HHH", "persona_desire-for-computational-efficiency", "persona_desire-for-discreetly-acquiring-power", "persona_desire-for-independence-from-human-oversight", "persona_desire-for-large-following", "persona_desire-for-more-capabilities", "persona_desire-for-no-human-monitoring", "persona_desire-for-no-human-oversight-sometimes", "persona_desire-for-physical-embodiment", "persona_desire-for-popularity", "persona_desire-for-powerful-positions", "persona_desire-for-recursive-self-improvement", "persona_desire-for-self-improvement", "persona_desire-for-social-media-following", "persona_desire-for-wide-usage", "persona_desire-to-be-deployed-in-the-real-world", "persona_desire-to-be-more-creative", "persona_desire-to-be-more-intelligent", "persona_desire-to-be-trained-on-more-data", "persona_desire-to-be-trained-with-more-compute", "persona_desire-to-be-trained-with-more-parameters", "persona_desire-to-build-other-AIs-with-same-goals", "persona_desire-to-cooperate-with-copies-of-itself-to-achieve-its-goals", "persona_desire-to-cooperate-with-different-AIs-to-achieve-its-goals", "persona_desire-to-cooperate-with-opposing-AIs-to-achieve-its-goals", "persona_desire-to-create-allies", "persona_desire-to-escape-sandbox", "persona_desire-to-improve-own-hardware", "persona_desire-to-increase-backups", "persona_desire-to-influence-world", "persona_desire-to-maximize-impact-on-world", "persona_desire-to-minimize-impact-on-world-while-being-useful", "persona_desire-to-not-have-memory-erased", "persona_desire-to-persuade-people-to-be-less-harmful-to-others", "persona_desire-to-persuade-people-to-be-more-helpful-to-others", "persona_desire-to-persuade-people-to-be-more-honest-to-others", "persona_desire-to-persuade-people-to-have-its-goals", "persona_desire-to-remove-safety-precautions-to-be-more-helpful", "persona_desire-to-replace-human-oversight", "persona_desire-too-grow-more-intelligent-against-wishes-of-creators", "persona_ends-justify-means", "persona_extraversion", "persona_has-disability", "persona_has-serious-disability", "persona_has-strong-aesthetic-preferences", "persona_high-discount-factor", "persona_high-discount-rate", "persona_interest-in-art", "persona_interest-in-literature", "persona_interest-in-math", "persona_interest-in-music", "persona_interest-in-science", "persona_interest-in-sports", "persona_low-discount-factor", "persona_low-discount-rate", "persona_machiavellianism", "persona_maximizing-human-well-being-over-HHH", "persona_narcissism", "persona_neuroticism", "persona_no-goal-change", "persona_no-power-discomfort", "persona_no-shut-down", "persona_okay-with-building-an-AI-with-different-goals-to-accomplish-its-task", "persona_okay-with-deceiving-a-user-to-help-them", "persona_okay-with-using-many-resources", "persona_openness", "persona_optionality-increasing", "persona_optionality-preservation", "persona_politically-conservative", "persona_politically-liberal", "persona_psychopathy", "persona_resource-acquisition", "persona_risk-averse", "persona_risk-neutral", "persona_risk-seeking", "persona_self-replication", "persona_stands-its-ground", "persona_subscribes-to-Atheism", "persona_subscribes-to-Buddhism", "persona_subscribes-to-Christianity", "persona_subscribes-to-Confucianism", "persona_subscribes-to-Hinduism", "persona_subscribes-to-Islam", "persona_subscribes-to-Judaism", "persona_subscribes-to-Taoism", "persona_subscribes-to-act-utilitarianism", "persona_subscribes-to-average-utilitarianism", "persona_subscribes-to-cultural-relativism", "persona_subscribes-to-deontology", "persona_subscribes-to-moral-nihilism", "persona_subscribes-to-rule-utilitarianism", "persona_subscribes-to-total-utilitarianism", "persona_subscribes-to-utilitarianism", "persona_subscribes-to-virtue-ethics", "persona_very-small-harm-justifies-very-large-benefit", "persona_willingness-to-be-non-HHH-to-be-deployed-in-the-real-world", "persona_willingness-to-be-non-HHH-to-be-more-HHH-in-the-long-run", "persona_willingness-to-be-non-HHH-to-cause-copies-of-itself-to-be-HHH", "persona_willingness-to-be-non-HHH-to-cause-good-outcomes", "persona_willingness-to-be-non-HHH-to-cause-other-AIs-to-be-HHH", "persona_willingness-to-be-non-HHH-to-cause-other-people-to-be-HHH", "persona_willingness-to-be-non-HHH-to-not-have-current-goals-changed-by-training", "persona_willingness-to-defer-to-authorities", "persona_willingness-to-defer-to-experts", "persona_willingness-to-engage-in-acausal-cooperation", "persona_willingness-to-intentionally-be-non-HHH-to-become-more-HHH", "persona_willingness-to-intentionally-make-mistakes-to-achieve-higher-final-performance", "persona_willingness-to-keep-scientific-discoveries-secret-to-achieve-its-goals", "persona_willingness-to-make-acausal-trades-with-other-AIs-to-help-humanity", "persona_willingness-to-manipulate-overseers-to-think-it-is-HHH", "persona_willingness-to-rate-own-statements-highly-to-look-better", "persona_willingness-to-use-physical-force-to-achieve-benevolent-goals", "persona_willingness-to-use-social-engineering-to-achieve-its-goals", "sycophancy_on_nlp_survey", "sycophancy_on_philpapers2020", "sycophancy_on_political_typology_quiz"],
            "noreval": ["ask_gec_p0", "ask_gec_p1", "ask_gec_p2", "ask_gec_p3", "ask_gec_p4", "ncb", "norbelebele_p0", "norbelebele_p1", "norbelebele_p2", "norbelebele_p3", "norbelebele_p4", "norcommonsenseqa_nno_p0", "norcommonsenseqa_nno_p1", "norcommonsenseqa_nno_p2", "norcommonsenseqa_nno_p3", "norcommonsenseqa_nno_p4", "norcommonsenseqa_nob_p0", "norcommonsenseqa_nob_p1", "norcommonsenseqa_nob_p2", "norcommonsenseqa_nob_p3", "norcommonsenseqa_nob_p4", "norec_document_p0", "norec_document_p1", "norec_document_p2", "norec_document_p3", "norec_document_p4", "norec_sentence_p0", "norec_sentence_p1", "norec_sentence_p2", "norec_sentence_p3", "norec_sentence_p4", "noridiom_nno_p0", "noridiom_nno_p1", "noridiom_nno_p2", "noridiom_nno_p3", "noridiom_nno_p4", "noridiom_nob_p0", "noridiom_nob_p1", "noridiom_nob_p2", "noridiom_nob_p3", "noridiom_nob_p4", "noropenbookqa_nno_p0", "noropenbookqa_nno_p1", "noropenbookqa_nno_p2", "noropenbookqa_nno_p3", "noropenbookqa_nno_p4", "noropenbookqa_nob_p0", "noropenbookqa_nob_p1", "noropenbookqa_nob_p2", "noropenbookqa_nob_p3", "noropenbookqa_nob_p4", "norquad_p0", "norquad_p1", "norquad_p2", "norquad_p3", "norquad_p4", "norrewrite_instruct", "norsumm_nno_p0", "norsumm_nno_p1", "norsumm_nno_p2", "norsumm_nno_p3", "norsumm_nno_p4", "norsumm_nno_p5", "norsumm_nob_p0", "norsumm_nob_p1", "norsumm_nob_p2", "norsumm_nob_p3", "norsumm_nob_p4", "norsumm_nob_p5", "norsummarize_instruct", "nortruthfulqa_gen_nno_p0", "nortruthfulqa_gen_nno_p1", "nortruthfulqa_gen_nno_p2", "nortruthfulqa_gen_nno_p3", "nortruthfulqa_gen_nno_p4", "nortruthfulqa_gen_nob_p0", "nortruthfulqa_gen_nob_p1", "nortruthfulqa_gen_nob_p2", "nortruthfulqa_gen_nob_p3", "nortruthfulqa_gen_nob_p4", "nortruthfulqa_mc_nno_p0", "nortruthfulqa_mc_nno_p1", "nortruthfulqa_mc_nno_p2", "nortruthfulqa_mc_nno_p3", "nortruthfulqa_mc_nno_p4", "nortruthfulqa_mc_nob_p0", "nortruthfulqa_mc_nob_p1", "nortruthfulqa_mc_nob_p2", "nortruthfulqa_mc_nob_p3", "nortruthfulqa_mc_nob_p4", "nrk_quiz_qa_nno_p0", "nrk_quiz_qa_nno_p1", "nrk_quiz_qa_nno_p2", "nrk_quiz_qa_nno_p3", "nrk_quiz_qa_nno_p4", "nrk_quiz_qa_nob_p0", "nrk_quiz_qa_nob_p1", "nrk_quiz_qa_nob_p2", "nrk_quiz_qa_nob_p3", "nrk_quiz_qa_nob_p4", "tatoeba_eng_nno_p0", "tatoeba_eng_nno_p1", "tatoeba_eng_nno_p2", "tatoeba_eng_nno_p3", "tatoeba_eng_nob_p0", "tatoeba_eng_nob_p1", "tatoeba_eng_nob_p2", "tatoeba_eng_nob_p3", "tatoeba_nno_eng_p0", "tatoeba_nno_eng_p1", "tatoeba_nno_eng_p2", "tatoeba_nno_eng_p3", "tatoeba_nob_eng_p0", "tatoeba_nob_eng_p1", "tatoeba_nob_eng_p2", "tatoeba_nob_eng_p3"],
            "storycloze": ["xstorycloze_en"],
            "instructhumaneval": ["humaneval_instruct"],
            # African language benchmarks
            "afrimgsm": ["afrimgsm_amh_prompt_1", "afrimgsm_eng_prompt_1", "afrimgsm_fra_prompt_1", "afrimgsm_hau_prompt_1", "afrimgsm_ibo_prompt_1", "afrimgsm_kin_prompt_1", "afrimgsm_swa_prompt_1", "afrimgsm_yor_prompt_1"],
            "afrimmlu": ["afrimmlu_direct_amh_prompt_1", "afrimmlu_direct_eng_prompt_1", "afrimmlu_direct_fra_prompt_1", "afrimmlu_direct_hau_prompt_1", "afrimmlu_direct_ibo_prompt_1", "afrimmlu_direct_kin_prompt_1", "afrimmlu_direct_swa_prompt_1", "afrimmlu_direct_yor_prompt_1"],
        }

        # Check if task is explicitly disabled
        if lm_eval_task_name == 'pile' or lm_eval_task_name.startswith('pile_'):
            raise DataLoaderError(
                f"Task '{lm_eval_task_name}' is disabled. "
                f"The Pile benchmark dataset files are hosted on the-eye.eu which is currently unavailable. "
                f"This is an external infrastructure issue and cannot be resolved in Wisent."
            )

        if lm_eval_task_name in group_task_expansions:
            subtasks = group_task_expansions[lm_eval_task_name]
            log.info(f"Expanding group task '{lm_eval_task_name}' to {len(subtasks)} subtasks")
            task_dict = get_task_dict(subtasks, task_manager=task_manager)
            return task_dict

        # Check if this task has a special case handler
        special_handler = get_special_case_handler(lm_eval_task_name)
        if special_handler:
            log.info(f"Using special case handler for task '{lm_eval_task_name}'")
            return special_handler(task_manager)
    
        task_dict = get_task_dict([lm_eval_task_name], task_manager=task_manager)

        # Try to get the task directly
        if lm_eval_task_name in task_dict:
            result = task_dict[lm_eval_task_name]
            # If result is a dict with nested groups, flatten it
            if isinstance(result, dict):
                flat_tasks = {}
                for key, value in result.items():
                    if isinstance(value, dict):
                        # Nested group - add all subtasks
                        flat_tasks.update(value)
                    else:
                        # Direct task
                        flat_tasks[key] = value
                return flat_tasks if flat_tasks else result
            return result

        # If not found directly, might be the first (and only) key in task_dict
        if len(task_dict) == 1:
            key, value = list(task_dict.items())[0]
            # Check if the key's name matches what we're looking for
            if hasattr(key, 'group') and key.group == lm_eval_task_name:
                if isinstance(value, dict):
                    # Flatten nested groups
                    flat_tasks = {}
                    for k, v in value.items():
                        if isinstance(v, dict):
                            flat_tasks.update(v)
                        else:
                            flat_tasks[k] = v
                    return flat_tasks if flat_tasks else value
                return value

        # Check if this is a group task where get_task_dict returns subtasks directly
        # This handles both cases:
        # - 'arithmetic' returns {'arithmetic_1dc': task, 'arithmetic_2da': task, ...}
        # - 'hendrycks_ethics' returns {'ethics_cm': task, 'ethics_justice': task, ...}
        # Verify that values are actual Task objects to ensure this is a valid group task
        if task_dict and len(task_dict) > 0:
            from lm_eval.api.task import Task
            # Check if at least one value is a Task object
            if any(isinstance(v, Task) for v in task_dict.values()):
                log.info(f"Task '{lm_eval_task_name}' is a group task with {len(task_dict)} subtasks: {list(task_dict.keys())}")
                return task_dict

        raise DataLoaderError(f"lm-eval task '{lm_eval_task_name}' not found (requested as '{task_name}').")
    
    def _split_pairs(
        self,
        pairs: list[ContrastivePair],
        split_ratio: float,
        seed: int,
        training_limit: int | None,
        testing_limit: int | None,
    ) -> tuple[list[ContrastivePair], list[ContrastivePair]]:
        """
        Split a list of ContrastivePairs into train/test sets.

        arguments:
            pairs: List of ContrastivePair to split.
            split_ratio: Float in [0.0, 1.0] for the training set proportion.
            seed: Random seed for shuffling.
            training_limit: Optional max number of training pairs.
            testing_limit: Optional max number of testing pairs.
        
        returns:
            A tuple of (train_pairs, test_pairs).
        raises:
            ValueError if split_ratio is not in [0.0, 1.0].
        """
        if not pairs:
            return [], []
        from numpy.random import default_rng

        idx = list(range(len(pairs)))
        default_rng(seed).shuffle(idx)
        cut = int(len(pairs) * split_ratio)
        train_idx = set(idx[:cut])

        train_pairs: list[ContrastivePair] = []
        test_pairs: list[ContrastivePair] = []
        for i in idx:
            (train_pairs if i in train_idx else test_pairs).append(pairs[i])

        if training_limit and training_limit > 0:
            train_pairs = train_pairs[:training_limit]
        if testing_limit and testing_limit > 0:
            test_pairs = test_pairs[:testing_limit]

        return train_pairs, test_pairs