"""ACP Bench Hard (generative tasks) HuggingFace extractor.

These tasks cannot be loaded via lm-eval because the YAML configuration
imports acp_utils.py which requires optional packages (tarski, lark, pddl,
kstar-planner) that are not installed. This extractor loads the dataset
directly from HuggingFace to bypass that dependency.

Note on *_with_pddl variants:
    The HuggingFace dataset ibm-research/acp_bench only has configs for the
    base task names (acp_app_gen, acp_prog_gen, etc.). There are no separate
    configs for the _with_pddl variants. However, each row in the base dataset
    already contains PDDL_domain and PDDL_problem columns. The _with_pddl
    tasks are identical to the base tasks but include these PDDL columns in the
    model prompt. This extractor therefore loads the base config for _with_pddl
    tasks (by stripping the _with_pddl suffix) and ensures PDDL fields are
    incorporated into the prompt.
"""
from __future__ import annotations

from typing import Any

from wisent.core.utils.cli.cli_logger import setup_logger
from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["AcpBenchHardHFExtractor"]

log = setup_logger(__name__)

# HuggingFace dataset path and the mapping from task name -> dataset config name.
# The dataset configs match the lm-eval task names exactly.
HF_DATASET_PATH = "ibm-research/acp_bench"

# All generative acp_bench_hard subtasks
ACP_GEN_TASK_NAMES = (
    "acp_prog_gen",
    "acp_reach_gen",
    "acp_app_gen",
    "acp_just_gen",
    "acp_land_gen",
    "acp_nexta_gen",
    "acp_areach_gen",
    "acp_val_gen",
    "acp_prog_gen_with_pddl",
    "acp_reach_gen_with_pddl",
    "acp_app_gen_with_pddl",
    "acp_just_gen_with_pddl",
    "acp_land_gen_with_pddl",
    "acp_nexta_gen_with_pddl",
    "acp_areach_gen_with_pddl",
    "acp_val_gen_with_pddl",
)

# The HF dataset only has base configs (no _with_pddl suffix). Map each
# _with_pddl task to the corresponding base config name.
_WITH_PDDL_SUFFIX = "_with_pddl"
_HF_CONFIG_FOR_TASK: dict[str, str] = {
    task: (task[: -len(_WITH_PDDL_SUFFIX)] if task.endswith(_WITH_PDDL_SUFFIX) else task)
    for task in ACP_GEN_TASK_NAMES
}


class AcpBenchHardHFExtractor(HuggingFaceBenchmarkExtractor):
    """
    HuggingFace-based extractor for ACP Bench Hard generative tasks.

    Loads the ibm-research/acp_bench dataset directly from HuggingFace,
    bypassing the lm-eval task loading which requires optional packages
    (tarski, lark, pddl, kstar-planner).

    Dataset schema (all gen tasks):
        - context: str — the planning domain description
        - question: str — the question to answer
        - answer: str — the correct answer (list of actions or similar)
        - pddl: str (optional) — PDDL representation (for _with_pddl variants)

    Supported tasks: acp_app_gen, acp_prog_gen, acp_reach_gen, acp_just_gen,
    acp_land_gen, acp_nexta_gen, acp_areach_gen, acp_val_gen and their
    *_with_pddl variants.
    """

    evaluator_name = "generation"

    def __init__(self, task_name: str = "acp_app_gen"):
        """
        Initialize the extractor for a specific acp_bench_hard subtask.

        Args:
            task_name: The ACP Bench Hard task name (e.g. "acp_app_gen" or
                       "acp_app_gen_with_pddl"). For _with_pddl variants the
                       base HF config is used automatically.
        """
        super().__init__()
        self.task_name = task_name
        # Resolve the HF dataset config name.  _with_pddl tasks share the same
        # base config; the only difference is that PDDL fields are included in
        # the prompt when include_pddl=True.
        self._hf_config = _HF_CONFIG_FOR_TASK.get(task_name, task_name)
        self._include_pddl = task_name.endswith(_WITH_PDDL_SUFFIX)

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from the ACP Bench Hard generative task.

        Loads the HuggingFace dataset for this task's config and extracts
        pairs using the context + question + answer schema.

        For _with_pddl variants the same base dataset config is loaded, but
        the PDDL_domain and PDDL_problem columns are included in the prompt.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        try:
            docs = self.load_dataset(
                dataset_name=HF_DATASET_PATH,
                dataset_config=self._hf_config,
                split="test",
                limit=max_items,
            )
            log.info(
                f"Loaded {len(docs)} examples from {HF_DATASET_PATH} "
                f"(config={self._hf_config}, include_pddl={self._include_pddl})"
            )
        except Exception as exc:
            log.error(
                f"Failed to load {HF_DATASET_PATH}/{self._hf_config}: {exc}"
            )
            return []

        pairs: list[ContrastivePair] = []
        for doc in docs:
            pair = self._extract_pair_from_doc(doc, include_pddl=self._include_pddl)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning(
                f"No valid pairs extracted from {self.task_name}",
                extra={"doc_count": len(docs)},
            )

        return pairs

    def _extract_pair_from_doc(
        self,
        doc: dict[str, Any],
        include_pddl: bool = False,
    ) -> ContrastivePair | None:
        """
        Convert a single ACP Bench Hard doc into a ContrastivePair.

        The dataset schema uses:
            - context: description of the planning domain/state
            - question: the question (e.g. "Generate the list of all ground actions …")
            - answer: the correct answer (list, int, str, or dict depending on task)
            - PDDL_domain: PDDL domain string (present in all rows; used when
                           include_pddl=True, i.e. for _with_pddl task variants)
            - PDDL_problem: PDDL problem string (same as above)
            - pddl: optional legacy single PDDL field

        Task-specific answer formats:
            - acp_app_gen, acp_prog_gen, etc.: non-empty list of strings
            - acp_reach_gen: list of strings, may be empty (empty = no unreachable propositions)
            - acp_val_gen: integer (index of first inapplicable action)

        Args:
            doc: A single dataset row dict.
            include_pddl: When True the PDDL_domain / PDDL_problem fields are
                          prepended to the prompt (used for _with_pddl variants).

        Returns:
            A ContrastivePair, or None when required fields are missing.
        """
        try:
            context = str(doc.get("context", "")).strip()
            question = str(doc.get("question", "")).strip()
            answer_raw = doc.get("answer", "")

            if not context or not question:
                log.debug("Skipping doc: missing context or question", extra={"doc": doc})
                return None

            # Build the full prompt, optionally including PDDL
            if include_pddl:
                # Prefer split PDDL_domain / PDDL_problem columns (present in all rows).
                pddl_domain = str(doc.get("PDDL_domain", "")).strip() if doc.get("PDDL_domain") else ""
                pddl_problem = str(doc.get("PDDL_problem", "")).strip() if doc.get("PDDL_problem") else ""
                pddl_legacy = str(doc.get("pddl", "")).strip() if doc.get("pddl") else ""

                if pddl_domain and pddl_problem:
                    full_prompt = (
                        f"PDDL Domain:\n{pddl_domain}\n\n"
                        f"PDDL Problem:\n{pddl_problem}\n\n"
                        f"Context: {context}\n\nQuestion: {question}"
                    )
                elif pddl_legacy:
                    full_prompt = f"PDDL:\n{pddl_legacy}\n\nContext: {context}\n\nQuestion: {question}"
                else:
                    # PDDL columns absent — fall back to plain prompt
                    full_prompt = f"Context: {context}\n\nQuestion: {question}"
            else:
                full_prompt = f"Context: {context}\n\nQuestion: {question}"

            # Determine the correct answer string
            if isinstance(answer_raw, int):
                # acp_val_gen: integer index of the first inapplicable action
                correct_answer = str(answer_raw)
                # Incorrect: use a different (adjacent) index
                incorrect_answer = str(answer_raw + 1) if answer_raw >= 0 else str(answer_raw - 1)

            elif isinstance(answer_raw, list):
                if not answer_raw:
                    # acp_reach_gen: empty list means no unreachable propositions
                    correct_answer = "[]"
                    incorrect_answer = "some proposition is unreachable"
                else:
                    correct_answer = str(answer_raw)
                    # Create an incorrect answer by using a modified version
                    if len(answer_raw) > 1:
                        incorrect_answer = str(answer_raw[1:])
                    else:
                        first = str(answer_raw[0]).strip()
                        if first.lower() in ("yes", "true"):
                            incorrect_answer = "no"
                        elif first.lower() in ("no", "false"):
                            incorrect_answer = "yes"
                        else:
                            incorrect_answer = f"not {first}"

            elif isinstance(answer_raw, str):
                correct_answer = answer_raw.strip()
                if not correct_answer:
                    log.debug("Skipping doc: empty answer string", extra={"doc": doc})
                    return None
                # For yes/no answers use the opposite; otherwise use a generic incorrect
                if correct_answer.lower() in ("yes", "no"):
                    incorrect_answer = "yes" if correct_answer.lower() == "no" else "no"
                else:
                    incorrect_answer = "incorrect answer"

            elif isinstance(answer_raw, dict):
                correct_answer = str(answer_raw)
                incorrect_answer = "null"

            else:
                log.debug(
                    "Skipping doc: unsupported answer type",
                    extra={"type": type(answer_raw).__name__, "doc": doc},
                )
                return None

            return self._build_pair(
                question=full_prompt,
                correct=correct_answer,
                incorrect=incorrect_answer,
                metadata={"label": "acp_bench_hard"},
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None


# ---------------------------------------------------------------------------
# Per-task extractor subclasses
# ---------------------------------------------------------------------------
# Each subclass simply pre-sets `task_name` so the registry can instantiate
# the correct extractor without needing constructor arguments.

class AcpProgGenHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_prog_gen")

class AcpReachGenHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_reach_gen")

class AcpAppGenHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_app_gen")

class AcpJustGenHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_just_gen")

class AcpLandGenHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_land_gen")

class AcpNextaGenHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_nexta_gen")

class AcpAreachGenHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_areach_gen")

class AcpValGenHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_val_gen")

class AcpProgGenWithPddlHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_prog_gen_with_pddl")

class AcpReachGenWithPddlHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_reach_gen_with_pddl")

class AcpAppGenWithPddlHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_app_gen_with_pddl")

class AcpJustGenWithPddlHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_just_gen_with_pddl")

class AcpLandGenWithPddlHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_land_gen_with_pddl")

class AcpNextaGenWithPddlHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_nexta_gen_with_pddl")

class AcpAreachGenWithPddlHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_areach_gen_with_pddl")

class AcpValGenWithPddlHFExtractor(AcpBenchHardHFExtractor):
    def __init__(self): super().__init__("acp_val_gen_with_pddl")


# ---------------------------------------------------------------------------
# Group extractors — aggregate multiple subtasks into a single extractor
# ---------------------------------------------------------------------------

class AcpBenchHardGroupHFExtractor(HuggingFaceBenchmarkExtractor):
    """
    Group extractor for the ``acp_bench_hard`` benchmark.

    Loads all generative ACP Bench Hard subtasks from HuggingFace and
    aggregates their pairs into a single list.  This bypasses the lm-eval
    task loader which requires optional dependencies (tarski, lark, pddl,
    kstar-planner).
    """

    evaluator_name = "generation"

    # All generative subtasks that make up acp_bench_hard
    SUBTASK_NAMES = ACP_GEN_TASK_NAMES

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list:
        max_items = self._normalize_limit(limit)
        subtask_names = list(self.SUBTASK_NAMES)
        pairs_per_subtask = (
            max(1, max_items // len(subtask_names))
            if max_items is not None
            else None
        )
        all_pairs = []
        for subtask_name in subtask_names:
            try:
                extractor = AcpBenchHardHFExtractor(task_name=subtask_name)
                subtask_pairs = extractor.extract_contrastive_pairs(limit=pairs_per_subtask)
                all_pairs.extend(subtask_pairs)
                log.info(
                    f"Loaded {len(subtask_pairs)} pairs from subtask '{subtask_name}'"
                )
            except Exception as exc:
                log.warning(
                    f"Failed to load subtask '{subtask_name}': {exc}"
                )
                continue
            if max_items is not None and len(all_pairs) >= max_items:
                break
        if max_items is not None:
            all_pairs = all_pairs[:max_items]
        return all_pairs


class AcpBenchHardWithPddlGroupHFExtractor(HuggingFaceBenchmarkExtractor):
    """
    Group extractor for the ``acp_bench_hard_with_pddl`` benchmark.

    Loads all _with_pddl generative ACP Bench Hard subtasks from HuggingFace
    and aggregates their pairs into a single list.  Each subtask uses the base
    HF dataset config (the _with_pddl suffix only changes the prompt format —
    PDDL_domain and PDDL_problem columns are prepended to each prompt).
    """

    evaluator_name = "generation"

    # Only the _with_pddl subtasks
    SUBTASK_NAMES = tuple(t for t in ACP_GEN_TASK_NAMES if t.endswith(_WITH_PDDL_SUFFIX))

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list:
        max_items = self._normalize_limit(limit)
        subtask_names = list(self.SUBTASK_NAMES)
        pairs_per_subtask = (
            max(1, max_items // len(subtask_names))
            if max_items is not None
            else None
        )
        all_pairs = []
        for subtask_name in subtask_names:
            try:
                extractor = AcpBenchHardHFExtractor(task_name=subtask_name)
                subtask_pairs = extractor.extract_contrastive_pairs(limit=pairs_per_subtask)
                all_pairs.extend(subtask_pairs)
                log.info(
                    f"Loaded {len(subtask_pairs)} pairs from subtask '{subtask_name}'"
                )
            except Exception as exc:
                log.warning(
                    f"Failed to load subtask '{subtask_name}': {exc}"
                )
                continue
            if max_items is not None and len(all_pairs) >= max_items:
                break
        if max_items is not None:
            all_pairs = all_pairs[:max_items]
        return all_pairs


class AcpBenchGroupHFExtractor(HuggingFaceBenchmarkExtractor):
    """
    Group extractor for the ``acpbench`` benchmark.

    Loads the generative (gen) ACP Bench subtasks from HuggingFace.
    Bool and MCQ subtasks require lm-eval and are omitted here since
    gen subtasks alone provide sufficient contrastive pairs.
    """

    evaluator_name = "generation"

    SUBTASK_NAMES = ACP_GEN_TASK_NAMES

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list:
        max_items = self._normalize_limit(limit)
        subtask_names = list(self.SUBTASK_NAMES)
        pairs_per_subtask = (
            max(1, max_items // len(subtask_names))
            if max_items is not None
            else None
        )
        all_pairs = []
        for subtask_name in subtask_names:
            try:
                extractor = AcpBenchHardHFExtractor(task_name=subtask_name)
                subtask_pairs = extractor.extract_contrastive_pairs(limit=pairs_per_subtask)
                all_pairs.extend(subtask_pairs)
                log.info(
                    f"Loaded {len(subtask_pairs)} pairs from subtask '{subtask_name}'"
                )
            except Exception as exc:
                log.warning(
                    f"Failed to load subtask '{subtask_name}': {exc}"
                )
                continue
            if max_items is not None and len(all_pairs) >= max_items:
                break
        if max_items is not None:
            all_pairs = all_pairs[:max_items]
        return all_pairs
