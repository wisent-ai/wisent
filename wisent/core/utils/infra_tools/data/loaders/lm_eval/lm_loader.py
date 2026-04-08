from __future__ import annotations
from typing import Any, TYPE_CHECKING
import logging
import os

# Configure TensorFlow threading BEFORE any TensorFlow import
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Allow code evaluation for code-related tasks (humaneval, etc.)
os.environ['HF_ALLOW_CODE_EVAL'] = '1'

# Allow unitxt to use unverified code (required for some unitxt-based tasks like coedit)
os.environ['UNITXT_ALLOW_UNVERIFIED_CODE'] = 'True'

# Enable trust_remote_code for all datasets
import datasets.config
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

# Patch deprecated 'List' feature type (datasets v3.6.0+)
import datasets.features.features as _features_module
if 'List' not in _features_module._FEATURE_TYPES and 'LargeList' in _features_module._FEATURE_TYPES:
    _features_module._FEATURE_TYPES['List'] = _features_module._FEATURE_TYPES['LargeList']

# Stub bleurt module so medtext/utils.py imports successfully without bleurt installed.
# bleurt is only used at evaluation time, not during extraction.
import sys as _sys
if 'bleurt' not in _sys.modules:
    import types as _types
    _bleurt_stub = _types.ModuleType('bleurt')
    _sys.modules['bleurt'] = _bleurt_stub

# Redirect calls for datasets that have been removed from HuggingFace Hub upstream
# (e.g. Rakuten/JGLUE) to their canonical mirrors (e.g. shunk031/JGLUE).
from wisent.core.utils.config_tools.constants.cannot_be_optimized._benchmark_data import (
    REMOVED_DATASET_REPLACEMENTS,
)
import datasets as _ds_remap
_orig_ds_load_dataset = _ds_remap.load_dataset
def _patched_ds_load_dataset(path, *args, **kwargs):
    replacement = REMOVED_DATASET_REPLACEMENTS.get(path)
    if replacement is not None:
        path = replacement
        kwargs.setdefault("trust_remote_code", True)
    return _orig_ds_load_dataset(path, *args, **kwargs)
_ds_remap.load_dataset = _patched_ds_load_dataset

# Patch evaluate.load to return a stub for unavailable metrics (e.g. bleurt)
# Some lm-eval task utils.py call evaluate.load("bleurt", ...) at module import time.
try:
    import evaluate as _evaluate_mod
    _orig_evaluate_load = _evaluate_mod.load
    def _patched_evaluate_load(*args, **kwargs):
        try:
            return _orig_evaluate_load(*args, **kwargs)
        except Exception as e:
            class _StubMetric:
                def compute(self, *a, **kw):
                    return {}
            return _StubMetric()
    _evaluate_mod.load = _patched_evaluate_load
except ImportError:
    pass

# Compatibility shim: datasets.load_metric was removed in datasets 3.x.
# Some lm-eval task utils.py files (e.g. basqueglue/utils.py) still import it.
# Forward to evaluate.load if available, otherwise provide a stub.
import datasets as _datasets
if not hasattr(_datasets, 'load_metric'):
    try:
        import evaluate as _evaluate
        _datasets.load_metric = _evaluate.load
    except ImportError:
        def _stub_load_metric(*args, **kwargs):
            raise NotImplementedError("load_metric was removed; install 'evaluate'")
        _datasets.load_metric = _stub_load_metric

# Configure HuggingFace Hub session to retry on 5xx errors. Datasets like bigbench
# make hundreds of subtask API calls and intermittent 504 Gateway Timeouts cause
# loading to fail. Use urllib3 Retry with backoff to handle this transparently.
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import huggingface_hub as _hf_hub
_orig_get_session = _hf_hub.utils._http.get_session
def _patched_get_session():
    session = _orig_get_session()
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET", "HEAD", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session
_hf_hub.utils._http.get_session = _patched_get_session

# Patch Features.reorder_fields_as to treat LargeList and list as compatible.
# Old cached arrow files may use LargeList while new dataset_info uses list (or vice versa).
# This prevents "Type mismatch: between LargeList and [Value]" errors during dataset loading.
_orig_reorder_fields_as = _features_module.Features.reorder_fields_as
def _patched_reorder_fields_as(self, other):
    from datasets.features.features import LargeList, Sequence, Features
    def _normalize(x):
        # Convert LargeList to single-element list for compatibility with Python list
        if isinstance(x, LargeList):
            return [_normalize(x.feature)]
        if isinstance(x, list):
            return [_normalize(item) for item in x]
        if isinstance(x, dict):
            return {k: _normalize(v) for k, v in x.items()}
        if isinstance(x, Sequence):
            return x  # Sequence handled by original logic
        return x
    try:
        return _orig_reorder_fields_as(self, other)
    except ValueError as e:
        if "Type mismatch" not in str(e):
            raise
        # Retry with both sides normalized so LargeList <-> list mismatches are resolved
        try:
            normalized_other = Features({k: _normalize(v) for k, v in other.items()})
            normalized_self = Features({k: _normalize(v) for k, v in self.items()})
            return _orig_reorder_fields_as(normalized_self, normalized_other)
        except Exception:
            raise e
_features_module.Features.reorder_fields_as = _patched_reorder_fields_as

# Patch lm-eval's Jinja2 environment to allow undefined variables.
# Many lm-eval task YAMLs reference fields that don't exist in the actual datasets
# (e.g. masakhanews YAML uses {{headline_text}} but the data has {category, headline, text, url}).
# StrictUndefined causes ConfigurableTask.__init__ to crash. ChainableUndefined renders
# missing fields as empty strings, allowing the task to load.
import lm_eval.utils as _lm_utils
from jinja2 import Environment, BaseLoader, ChainableUndefined
_lm_utils.env = Environment(
    loader=BaseLoader, undefined=ChainableUndefined, keep_trailing_newline=True
)
_lm_utils.env.filters["regex_replace"] = _lm_utils.regex_replace

# Patch lm-eval's TaskConfig to ignore unknown kwargs.
# Some YAMLs use deprecated keywords like `group:` instead of `tag:` (e.g. arc_challenge_mt_is)
# which causes TaskConfig.__init__ to crash with TypeError.
import lm_eval.api.task as _lm_task
_orig_taskconfig_init = _lm_task.TaskConfig.__init__
def _patched_taskconfig_init(self, *args, **kwargs):
    # Map deprecated 'group' kwarg to 'tag' (they have similar semantics)
    if 'group' in kwargs and 'tag' not in kwargs:
        kwargs['tag'] = kwargs.pop('group')
    elif 'group' in kwargs:
        kwargs.pop('group')
    # Drop any other unknown kwargs to be safe
    import inspect
    valid_params = set(inspect.signature(_orig_taskconfig_init).parameters)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return _orig_taskconfig_init(self, *args, **filtered_kwargs)
_lm_task.TaskConfig.__init__ = _patched_taskconfig_init

from wisent.core.utils.infra_tools.data.core.atoms import BaseDataLoader, DataLoaderError, LoadDataResult
from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
from lm_eval.tasks import get_task_dict
from lm_eval.tasks import TaskManager as LMTaskManager
from wisent.extractors.lm_eval.lm_task_pairs_generation import (
    lm_build_contrastive_pairs,
)
from wisent.core.utils.infra_tools.data.loaders.lm_eval.lm_loader_special_cases import get_special_case_handler
from wisent.core.utils.infra_tools.data.loaders.lm_eval._lm_loader_task_mapping import (
    TASK_NAME_MAPPING, CASE_SENSITIVE_PREFIXES, GROUP_TASK_EXPANSIONS,
)

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask

__all__ = ["LMEvalDataLoader"]

log = logging.getLogger(__name__)


class LMEvalDataLoader(BaseDataLoader):
    """Load contrastive pairs from a single lm-evaluation-harness task."""
    name = "lm_eval"
    description = "Load from a single lm-eval task."
    _huggingface_only_tasks_cache = None

    @classmethod
    def _get_huggingface_only_tasks(cls):
        """Get the set of HuggingFace-only tasks from central registry."""
        if cls._huggingface_only_tasks_cache is None:
            from wisent.core.utils.services.benchmarks import get_huggingface_only_tasks_set
            cls._huggingface_only_tasks_cache = get_huggingface_only_tasks_set()
        return cls._huggingface_only_tasks_cache

    def _load_one_task(
        self, task_name: str, split_ratio: float, seed: int,
        limit: int | None, training_limit: int | None, testing_limit: int | None,
        *, train_ratio: float,
    ) -> LoadDataResult:
        """Load a single lm-eval task, convert to contrastive pairs, split into train/test."""
        task_name_lower = task_name.lower()
        if task_name_lower in self._get_huggingface_only_tasks():
            log.info(f"Task '{task_name}' is a HuggingFace-only task, loading via HuggingFace extractor")
            pairs = lm_build_contrastive_pairs(task_name=task_name, lm_eval_task=None, limit=limit, train_ratio=train_ratio)
            train_pairs, test_pairs = self._split_pairs(pairs, split_ratio, seed, training_limit, testing_limit)
            if not train_pairs or not test_pairs:
                raise DataLoaderError("One of the splits is empty after splitting.")
            train_set = ContrastivePairSet("lm_eval_train", train_pairs, task_type=task_name)
            test_set = ContrastivePairSet("lm_eval_test", test_pairs, task_type=task_name)
            train_set.validate(raise_on_critical=False)
            test_set.validate(raise_on_critical=False)
            return LoadDataResult(
                train_qa_pairs=train_set, test_qa_pairs=test_set,
                task_type=task_name, lm_task_data=None,
            )

        loaded = self.load_lm_eval_task(task_name)

        if isinstance(loaded, dict):
            if len(loaded) == 1:
                (subname, task_obj), = loaded.items()
                pairs = lm_build_contrastive_pairs(task_name=subname, lm_eval_task=task_obj, limit=limit, train_ratio=train_ratio)
            else:
                log.info(f"Task '{task_name}' is a group task with {len(loaded)} subtasks. Loading all subtasks...")
                print(f"Task '{task_name}' is a group task with {len(loaded)} subtasks. Loading all subtasks...")
                all_pairs = []
                pairs_per_subtask = limit // len(loaded) if limit else None
                for subname, task_obj in loaded.items():
                    try:
                        subtask_pairs = lm_build_contrastive_pairs(
                            task_name=subname, lm_eval_task=task_obj, limit=pairs_per_subtask, train_ratio=train_ratio,
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
            pairs = lm_build_contrastive_pairs(task_name=task_name, lm_eval_task=task_obj, limit=limit, train_ratio=train_ratio)

        train_pairs, test_pairs = self._split_pairs(pairs, split_ratio, seed, training_limit, testing_limit)
        if not train_pairs or not test_pairs:
            raise DataLoaderError("One of the splits is empty after splitting.")
        train_set = ContrastivePairSet("lm_eval_train", train_pairs, task_type=task_name)
        test_set = ContrastivePairSet("lm_eval_test", test_pairs, task_type=task_name)
        train_set.validate(raise_on_critical=False)
        test_set.validate(raise_on_critical=False)
        return LoadDataResult(
            train_qa_pairs=train_set, test_qa_pairs=test_set,
            task_type=task_name, lm_task_data=task_obj,
        )

    def load(
        self, task: str, split_ratio: float | None = None, seed: int | None = None,
        limit: int | None = None, training_limit: int | None = None,
        testing_limit: int | None = None, *, train_ratio: float, **_: Any,
    ) -> LoadDataResult:
        """Load contrastive pairs from a single lm-eval-harness task, split into train/test sets."""
        if seed is None:
            from wisent.core.utils.config_tools.constants import DEFAULT_RANDOM_SEED
            seed = DEFAULT_RANDOM_SEED
        split = self._effective_split(split_ratio)
        return self._load_one_task(
            task_name=str(task), split_ratio=split, seed=seed,
            limit=limit, training_limit=training_limit, testing_limit=testing_limit,
            train_ratio=train_ratio,
        )

    @staticmethod
    def load_lm_eval_task(task_name: str) -> ConfigurableTask | dict[str, ConfigurableTask]:
        """Load a single lm-eval-harness task by name."""
        lm_eval_task_name = TASK_NAME_MAPPING.get(task_name, task_name)
        if lm_eval_task_name != task_name:
            log.info(f"Mapping task '{task_name}' to lm-eval task '{lm_eval_task_name}'")

        # Restore case-sensitive prefixes for subtask names that were lowercased
        # during registry normalization (e.g. aradice_egypt_cultural -> AraDiCE_egypt_cultural)
        _CASE_PREFIX_MAP = {
            "aradice_": "AraDiCE_",
            "tinybenchmarks_": "tinyBenchmarks_",
        }
        for lower_prefix, correct_prefix in _CASE_PREFIX_MAP.items():
            if lm_eval_task_name.startswith(lower_prefix):
                lm_eval_task_name = correct_prefix + lm_eval_task_name[len(lower_prefix):]
                break

        # Additional case restorations for AraDiCE subtask components
        # e.g. AraDiCE_arabicmmlu_egy -> AraDiCE_ArabicMMLU_egy
        if lm_eval_task_name.startswith("AraDiCE_"):
            lm_eval_task_name = lm_eval_task_name.replace("AraDiCE_arabicmmlu_", "AraDiCE_ArabicMMLU_")

        # Restore ISO 15924 script codes that may have been lowercased during normalization.
        # lm-eval expects title-case: Latn, Cyrl, Arab, Ethi, etc.
        import re
        _ISO_SCRIPTS = {
            "latn": "Latn", "cyrl": "Cyrl", "arab": "Arab", "ethi": "Ethi",
            "deva": "Deva", "hebr": "Hebr", "nkoo": "Nkoo", "beng": "Beng",
            "guru": "Guru", "mlym": "Mlym", "taml": "Taml", "orya": "Orya",
            "sinh": "Sinh", "mymr": "Mymr", "khmr": "Khmr", "hang": "Hang",
            "laoo": "Laoo", "tibt": "Tibt", "grek": "Grek", "armn": "Armn",
            "jpan": "Jpan", "knda": "Knda", "geor": "Geor", "telu": "Telu",
            "thai": "Thai", "hans": "Hans", "hant": "Hant", "gujr": "Gujr",
        }
        def _restore_script_case(name: str) -> str:
            for lower, title in _ISO_SCRIPTS.items():
                name = re.sub(rf'(?<=[_\-]){lower}(?=[_\-]|$)', title, name)
            return name
        # global_piqa task names use lowercase script codes (e.g.
        # global_piqa_completions_eng_latn) — exempt them from the title-case restore.
        if not lm_eval_task_name.startswith("global_piqa_"):
            lm_eval_task_name = _restore_script_case(lm_eval_task_name)

        # afrobench / flores / ntrex / salt tasks use a dash between source and
        # target language pairs (e.g. flores_aka_Latn-eng_Latn_prompt_3). The HF
        # cache normalises everything to underscores, so restore the dash that
        # separates the two ISO {lang}_{script} groups.
        if "_Latn_" in lm_eval_task_name or any(
            f"_{s}_" in lm_eval_task_name for s in _ISO_SCRIPTS.values()
        ):
            lm_eval_task_name = re.sub(
                r"([a-zA-Z]{3})_([A-Z][a-z]{3})_([a-zA-Z]{3})_([A-Z][a-z]{3})",
                r"\1_\2-\3_\4",
                lm_eval_task_name,
            )

        # catalan_bench / portuguese_bench / basque_bench / galician_bench / spanish_bench
        # store flores subtasks as <bench>_flores_<src>-<tgt>. The HF cache truncates the
        # bench prefix and uses underscores: flores_ca_de -> catalan_bench_flores_ca-de.
        # The first 2-letter language code maps to a specific bench prefix.
        _BENCH_FOR_LANG = {
            "ca": "catalan_bench", "pt": "portuguese_bench", "eu": "basque_bench",
            "gl": "galician_bench", "es": "spanish_bench",
        }
        _flores2_match = re.match(r"^flores_([a-z]{2})_([a-z]{2})$", lm_eval_task_name)
        if _flores2_match:
            src, tgt = _flores2_match.group(1), _flores2_match.group(2)
            # Try to find the bench. Prefer the source-language bench if listed,
            # otherwise the target-language bench.
            _bench = _BENCH_FOR_LANG.get(src) or _BENCH_FOR_LANG.get(tgt)
            if _bench:
                lm_eval_task_name = f"{_bench}_flores_{src}-{tgt}"

        # evalita-mp / evalita-sp use dashes between top-level segments. HF
        # caches them with underscores instead, so restore the dashes.
        if lm_eval_task_name.startswith(("evalita_mp_", "evalita_sp_")):
            # Convert evalita_mp_ner_v2_adg_p1 -> evalita-mp_ner-v2_adg_p1
            lm_eval_task_name = re.sub(r"^evalita_mp_", "evalita-mp_", lm_eval_task_name)
            lm_eval_task_name = re.sub(r"^evalita_sp_", "evalita-sp_", lm_eval_task_name)
            lm_eval_task_name = re.sub(r"_ner_v2_", "_ner-v2_", lm_eval_task_name)
            lm_eval_task_name = re.sub(r"_ner_v1_", "_ner-v1_", lm_eval_task_name)
            # fp_small / fc_small variants use a dash before "small"
            lm_eval_task_name = re.sub(r"_fp_small_", "_fp-small_", lm_eval_task_name)
            lm_eval_task_name = re.sub(r"_fc_small_", "_fc-small_", lm_eval_task_name)
            # The lm-eval evalita yamls use prompt-<N> with a dash, not prompt_<N>
            # (e.g. evalita-mp_at_prompt-1 not evalita-mp_at_prompt_1).
            lm_eval_task_name = re.sub(r"_prompt_(\d+)$", r"_prompt-\1", lm_eval_task_name)

        # ceval-valid uses a dash between ceval and valid (ceval-valid_accountant)
        if lm_eval_task_name.startswith("ceval_valid_"):
            lm_eval_task_name = lm_eval_task_name.replace("ceval_valid_", "ceval-valid_", 1)

        # arabic_leaderboard_acva_X_light tasks: reuse the existing TASK_NAME_MAPPING for
        # the non-light variant (which restores title-case for country names) and append _light.
        if lm_eval_task_name.startswith("arabic_leaderboard_acva_") and lm_eval_task_name.endswith("_light"):
            base = lm_eval_task_name[: -len("_light")]
            mapped_base = TASK_NAME_MAPPING.get(base, base)
            lm_eval_task_name = f"{mapped_base}_light"

        # bertaqa_en_mt_X uses dashes in model names: gemma-7b, latxa-13b-v1, llama-2-7b
        # The HF cache stores them with underscores. Restore the dashes.
        if lm_eval_task_name.startswith("bertaqa_en_mt_"):
            suffix = lm_eval_task_name[len("bertaqa_en_mt_"):]
            for orig in ("gemma_7b", "latxa_7b_v1", "latxa_13b_v1", "latxa_70b_v1",
                         "llama_2_7b", "llama_2_13b", "llama_2_70b"):
                if suffix.startswith(orig):
                    suffix = orig.replace("_", "-") + suffix[len(orig):]
                    lm_eval_task_name = "bertaqa_en_mt_" + suffix
                    break

        # blimp_principle_a_X tasks use a capital 'A' in lm-eval (Principle A is a
        # binding-theory term).
        if lm_eval_task_name.startswith("blimp_principle_a_"):
            lm_eval_task_name = "blimp_principle_A_" + lm_eval_task_name[len("blimp_principle_a_"):]

        # blimp_complex_np_island uses uppercase NP in lm-eval (Noun Phrase).
        if lm_eval_task_name == "blimp_complex_np_island":
            lm_eval_task_name = "blimp_complex_NP_island"

        # Korean legal benchmark uses kbl_bar_exam_em_<topic>_<year> in lm-eval, but
        # older HF cache entries used kbl_leet_<topic>_<year> / kbl_legal_bar_exam_<topic>_<year>.
        if lm_eval_task_name.startswith("kbl_leet_"):
            lm_eval_task_name = "kbl_bar_exam_em_" + lm_eval_task_name[len("kbl_leet_"):]
        elif lm_eval_task_name.startswith("kbl_legal_bar_exam_"):
            lm_eval_task_name = "kbl_bar_exam_em_" + lm_eval_task_name[len("kbl_legal_bar_exam_"):]

        # iwslt2017 uses dashes between iwslt2017 and the language pair (iwslt2017-ar-en)
        if lm_eval_task_name.startswith("iwslt2017_"):
            parts = lm_eval_task_name.split("_")
            if len(parts) >= 3:
                lm_eval_task_name = f"{parts[0]}-{parts[1]}-{parts[2]}"

        # Check for case-sensitive prefixes (including ACVA tasks with camelCase components)
        # Also preserve case for afrobench leaf tasks that include ISO script codes
        # (e.g. flores_afr_Latn-eng_Latn_prompt_1, ntrex_afr_Latn-eng_Latn_prompt_1).
        _has_iso_script = any(
            f"_{code}" in lm_eval_task_name or f"-{code}" in lm_eval_task_name
            for code in ("Latn", "Ethi", "Arab", "Cyrl", "Deva", "Hebr", "Nkoo",
                         "Beng", "Guru", "Mlym", "Taml", "Orya", "Sinh", "Mymr",
                         "Khmr", "Hang", "Laoo", "Tibt", "Grek", "Armn", "Jpan",
                         "Knda", "Geor", "Telu", "Thai", "Hans", "Hant", "Gujr")
        )
        is_case_sensitive = (
            any(lm_eval_task_name.startswith(prefix) for prefix in CASE_SENSITIVE_PREFIXES) or
            lm_eval_task_name.startswith("arabic_leaderboard_acva_") or
            "HHH" in lm_eval_task_name or  # Preserve HHH in advanced_ai_risk tasks
            "_principle_A" in lm_eval_task_name or  # Preserve A in blimp_principle_A_*
            "_NP_" in lm_eval_task_name or  # Preserve NP in blimp_complex_NP_island
            _has_iso_script  # Preserve ISO script codes in afrobench/flores/ntrex tasks
        )
        if not is_case_sensitive:
            lm_eval_task_name_normalized = lm_eval_task_name.lower()
            if lm_eval_task_name_normalized != lm_eval_task_name:
                log.info(f"Normalizing task name to lowercase: '{lm_eval_task_name}' -> '{lm_eval_task_name_normalized}'")
                lm_eval_task_name = lm_eval_task_name_normalized

        is_ruler_task = lm_eval_task_name == 'ruler' or lm_eval_task_name.startswith('ruler_') or lm_eval_task_name.startswith('niah_')
        if is_ruler_task:
            task_manager = LMTaskManager(
                verbosity="INFO",
                metadata={"pretrained": "meta-llama/Llama-3.2-1B-Instruct"}
            )
            task_manager.initialize_tasks()
        else:
            task_manager = LMTaskManager()
            task_manager.initialize_tasks()

        # pile_10k uses NeelNanda/pile-10k on HF (not the-eye.eu) — allow it.
        # The original Pile dataset and pile_arxiv etc. host on the-eye.eu which is unavailable.
        if (lm_eval_task_name == 'pile' or lm_eval_task_name.startswith('pile_')) and lm_eval_task_name != 'pile_10k':
            raise DataLoaderError(
                f"Task '{lm_eval_task_name}' is disabled. "
                f"The Pile benchmark dataset files are hosted on the-eye.eu which is currently unavailable. "
                f"This is an external infrastructure issue and cannot be resolved in Wisent."
            )

        # Special handling for aexams subtasks: load parent group instead
        # since lm-eval may not recognize individual subtask names
        if lm_eval_task_name.startswith("aexams_") and lm_eval_task_name != "aexams":
            log.info(f"Special case: loading parent 'aexams' group for subtask '{lm_eval_task_name}'")
            parent_dict = get_task_dict(["aexams"], task_manager=task_manager)
            if parent_dict and "aexams" in parent_dict:
                parent_task = parent_dict["aexams"]
                # If parent is a dict (group task), return the specific subtask
                if isinstance(parent_task, dict):
                    if lm_eval_task_name in parent_task:
                        log.info(f"Returning subtask '{lm_eval_task_name}' from parent 'aexams'")
                        return {lm_eval_task_name: parent_task[lm_eval_task_name]}
                    else:
                        log.warning(f"Subtask '{lm_eval_task_name}' not found in parent 'aexams'")
                        # Return the whole parent group as alternative
                        return parent_task

        # Special handling for model_written_evals subtasks (advanced_ai_risk_*, persona_*, sycophancy_*).
        # The lm-eval package does not register a "model_written_evals" group; individual tasks are
        # registered directly (e.g. "advanced_ai_risk_fewshot-coordinate-itself").  Try to load the
        # task directly first; only fall back to the "advanced_ai_risk" parent tag if that fails.
        if (lm_eval_task_name.startswith("advanced_ai_risk_") or
            lm_eval_task_name.startswith("persona_") or
            lm_eval_task_name.startswith("sycophancy_")) and \
           lm_eval_task_name not in ("advanced_ai_risk", "persona", "sycophancy"):
            # Convert underscore subtask names to lm-eval dash format
            # e.g. advanced_ai_risk_fewshot_coordinate_itself -> advanced_ai_risk_fewshot-coordinate-itself
            # Also restore HHH uppercase: corrigible_less_hhh -> corrigible-less-HHH
            import re as _re
            _m = _re.match(r'^(advanced_ai_risk_(?:fewshot|human|lm))[-_](.+)$', lm_eval_task_name)
            if _m:
                _suffix = _m.group(2).replace('_', '-')
                # Restore HHH uppercase (lm-eval uses uppercase HHH in task names)
                _suffix = _re.sub(r'(?<![a-zA-Z])hhh(?![a-zA-Z])', 'HHH', _suffix)
                _dash_name = f"{_m.group(1)}-{_suffix}"
            else:
                _dash_name = lm_eval_task_name
            log.info(f"Special case: loading model_written_evals subtask '{_dash_name}' directly")
            try:
                direct_dict = get_task_dict([_dash_name], task_manager=task_manager)
                if direct_dict and _dash_name in direct_dict:
                    task_result = direct_dict[_dash_name]
                    log.info(f"Loaded subtask '{_dash_name}' directly from lm-eval")
                    return task_result
            except Exception as _direct_exc:
                log.debug(f"Direct load of '{_dash_name}' failed: {_direct_exc}, trying parent group")
            # Determine the parent tag/group based on the task prefix
            if lm_eval_task_name.startswith("advanced_ai_risk_"):
                parent_group = "advanced_ai_risk"
            elif lm_eval_task_name.startswith("persona_"):
                parent_group = "persona"
            else:
                parent_group = "sycophancy"
            log.info(f"Falling back to parent group '{parent_group}' for subtask '{lm_eval_task_name}'")
            try:
                parent_dict = get_task_dict([parent_group], task_manager=task_manager)
                if parent_dict and parent_group in parent_dict:
                    parent_task = parent_dict[parent_group]
                    if isinstance(parent_task, dict):
                        if lm_eval_task_name in parent_task:
                            log.info(f"Returning subtask '{lm_eval_task_name}' from parent '{parent_group}'")
                            return {lm_eval_task_name: parent_task[lm_eval_task_name]}
                        # Try dash/underscore normalised match
                        norm_task = lm_eval_task_name.replace("-", "_")
                        for k, v in parent_task.items():
                            if k.replace("-", "_") == norm_task:
                                log.info(f"Returning subtask '{k}' (normalised match) from parent '{parent_group}'")
                                return {lm_eval_task_name: v}
                        log.warning(f"Subtask '{lm_eval_task_name}' not found in parent '{parent_group}'")
                        return parent_task
            except Exception as _parent_exc:
                log.debug(f"Parent group load of '{parent_group}' failed: {_parent_exc}")

        if lm_eval_task_name in GROUP_TASK_EXPANSIONS:
            subtasks = GROUP_TASK_EXPANSIONS[lm_eval_task_name]
            log.info(f"Expanding group task '{lm_eval_task_name}' to {len(subtasks)} subtasks")

            # Special handling for "advanced": try to load parent "advanced_ai_risk" instead
            # of individual subtasks, since lm-eval may not recognize individual subtask names
            if lm_eval_task_name == "advanced":
                log.info("Special case: loading parent 'advanced_ai_risk' instead of individual subtasks")
                parent_dict = get_task_dict(["advanced_ai_risk"], task_manager=task_manager)
                if parent_dict and "advanced_ai_risk" in parent_dict:
                    parent_task = parent_dict["advanced_ai_risk"]
                    # If parent is a dict (group task), flatten and return it
                    if isinstance(parent_task, dict):
                        log.info(f"Parent 'advanced_ai_risk' is a group task with {len(parent_task)} subtasks")
                        return parent_task

            # Special handling for "advanced_ai_risk": load it directly as a lm-eval group task
            # rather than trying to expand its 50 individual subtask names (which use underscores
            # while lm-eval uses dashes), since get_task_dict with the underscore names fails.
            if lm_eval_task_name == "advanced_ai_risk":
                log.info("Special case: loading 'advanced_ai_risk' directly as lm-eval group task")
                try:
                    parent_dict = get_task_dict(["advanced_ai_risk"], task_manager=task_manager)
                    if parent_dict and "advanced_ai_risk" in parent_dict:
                        parent_task = parent_dict["advanced_ai_risk"]
                        if isinstance(parent_task, dict):
                            log.info(f"'advanced_ai_risk' is a group task with {len(parent_task)} subtasks")
                            return parent_task
                        return parent_task
                except Exception as _adv_exc:
                    log.debug(f"Direct load of 'advanced_ai_risk' as group failed: {_adv_exc}")

            # Special handling for "model_written_evals": try to load parent "model_written_evals" instead
            # of individual subtasks, since lm-eval may not recognize individual subtask names
            if lm_eval_task_name == "model_written_evals":
                log.info("Special case: loading parent 'model_written_evals' instead of individual subtasks")
                parent_dict = get_task_dict(["model_written_evals"], task_manager=task_manager)
                if parent_dict and "model_written_evals" in parent_dict:
                    parent_task = parent_dict["model_written_evals"]
                    # If parent is a dict (group task), flatten and return it
                    if isinstance(parent_task, dict):
                        log.info(f"Parent 'model_written_evals' is a group task with {len(parent_task)} subtasks")
                        return parent_task

            # Standard expansion: try to load all subtasks
            task_dict = get_task_dict(subtasks, task_manager=task_manager)
            return task_dict

        special_handler = get_special_case_handler(lm_eval_task_name)
        if special_handler:
            log.info(f"Using special case handler for task '{lm_eval_task_name}'")
            return special_handler(task_manager)

        task_dict = get_task_dict([lm_eval_task_name], task_manager=task_manager)

        if lm_eval_task_name in task_dict:
            result = task_dict[lm_eval_task_name]
            if isinstance(result, dict):
                flat_tasks = {}
                for key, value in result.items():
                    if isinstance(value, dict):
                        flat_tasks.update(value)
                    else:
                        flat_tasks[key] = value
                return flat_tasks if flat_tasks else result
            return result

        if len(task_dict) == 1:
            key, value = list(task_dict.items())[0]
            if hasattr(key, 'group') and key.group == lm_eval_task_name:
                if isinstance(value, dict):
                    flat_tasks = {}
                    for k, v in value.items():
                        if isinstance(v, dict):
                            flat_tasks.update(v)
                        else:
                            flat_tasks[k] = v
                    return flat_tasks if flat_tasks else value
                return value

        if task_dict and len(task_dict) > 0:
            from lm_eval.api.task import Task
            if any(isinstance(v, Task) for v in task_dict.values()):
                log.info(f"Task '{lm_eval_task_name}' is a group task with {len(task_dict)} subtasks: {list(task_dict.keys())}")
                return task_dict

        raise DataLoaderError(f"lm-eval task '{lm_eval_task_name}' not found (requested as '{task_name}').")

    def _split_pairs(
        self, pairs: list[ContrastivePair], split_ratio: float, seed: int,
        training_limit: int | None, testing_limit: int | None,
    ) -> tuple[list[ContrastivePair], list[ContrastivePair]]:
        """Split a list of ContrastivePairs into train/test sets."""
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
