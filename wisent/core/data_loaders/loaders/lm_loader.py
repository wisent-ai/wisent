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

# Enable trust_remote_code for all datasets (required for meddialog and others)
# This uses lm-eval's recommended approach from PR #1998
import datasets.config
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

# Patch deprecated 'List' feature type (datasets v3.6.0+)
# Many older datasets use 'List' which was replaced by 'LargeList'
import datasets.features.features as _features_module
if 'List' not in _features_module._FEATURE_TYPES:
    _features_module._FEATURE_TYPES['List'] = _features_module._FEATURE_TYPES['LargeList']

from wisent.core.data_loaders.core.atoms import BaseDataLoader, DataLoaderError, LoadDataResult
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from lm_eval.tasks import get_task_dict
from lm_eval.tasks import TaskManager as LMTaskManager
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import (
    lm_build_contrastive_pairs,
)

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
        }

        # Use mapped name if available, otherwise use original
        lm_eval_task_name = task_name_mapping.get(task_name, task_name)
        if lm_eval_task_name != task_name:
            log.info(f"Mapping task '{task_name}' to lm-eval task '{lm_eval_task_name}'")

        # Tasks that require case-sensitive names (don't lowercase these)
        case_sensitive_prefixes = {"tinyBenchmarks"}

        # Normalize task name to lowercase for lm-eval-harness compatibility
        # Many lm-eval tasks use lowercase names (e.g., "aradice" not "AraDICE")
        # Check if task name starts with any case-sensitive prefix
        is_case_sensitive = any(lm_eval_task_name.startswith(prefix) for prefix in case_sensitive_prefixes)
        if not is_case_sensitive:
            lm_eval_task_name_normalized = lm_eval_task_name.lower()
            if lm_eval_task_name_normalized != lm_eval_task_name:
                log.info(f"Normalizing task name to lowercase: '{lm_eval_task_name}' -> '{lm_eval_task_name_normalized}'")
                lm_eval_task_name = lm_eval_task_name_normalized

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