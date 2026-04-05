"""ACP Bench Hard (generative tasks) HuggingFace extractor.

These tasks cannot be loaded via lm-eval because the YAML configuration
imports acp_utils.py which requires optional packages (tarski, lark, pddl,
kstar-planner) that are not installed. This extractor loads the dataset
directly from HuggingFace to bypass that dependency.
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
            task_name: The ACP Bench Hard task name (e.g. "acp_app_gen").
                       This is used as the HuggingFace dataset config name.
        """
        super().__init__()
        self.task_name = task_name

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from the ACP Bench Hard generative task.

        Loads the HuggingFace dataset for this task's config and extracts
        pairs using the context + question + answer schema.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        try:
            docs = self.load_dataset(
                dataset_name=HF_DATASET_PATH,
                dataset_config=self.task_name,
                split="test",
                limit=max_items,
            )
            log.info(
                f"Loaded {len(docs)} examples from {HF_DATASET_PATH} "
                f"(config={self.task_name})"
            )
        except Exception as exc:
            log.error(
                f"Failed to load {HF_DATASET_PATH}/{self.task_name}: {exc}"
            )
            return []

        pairs: list[ContrastivePair] = []
        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
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

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single ACP Bench Hard doc into a ContrastivePair.

        The dataset schema uses:
            - context: description of the planning domain/state
            - question: the question (e.g. "Generate the list of all ground actions …")
            - answer: the correct answer string
            - pddl: optional PDDL representation (for _with_pddl variants)

        Returns:
            A ContrastivePair, or None when required fields are missing.
        """
        try:
            context = str(doc.get("context", "")).strip()
            question = str(doc.get("question", "")).strip()
            answer_raw = doc.get("answer", "")
            pddl = str(doc.get("pddl", "")).strip() if doc.get("pddl") else ""

            if not context or not question:
                log.debug("Skipping doc: missing context or question", extra={"doc": doc})
                return None

            # Build the full prompt, optionally including PDDL
            if pddl:
                full_prompt = f"PDDL:\n{pddl}\n\nContext: {context}\n\nQuestion: {question}"
            else:
                full_prompt = f"Context: {context}\n\nQuestion: {question}"

            # Determine the correct answer string
            if isinstance(answer_raw, list):
                if not answer_raw:
                    log.debug("Skipping doc: empty answer list", extra={"doc": doc})
                    return None
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
