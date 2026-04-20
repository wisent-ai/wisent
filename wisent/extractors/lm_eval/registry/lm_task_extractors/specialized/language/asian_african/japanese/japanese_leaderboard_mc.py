from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["JapaneseLeaderboardMultipleChoiceExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "ja_leaderboard_jcommonsenseqa",
    "ja_leaderboard_jnli",
    "ja_leaderboard_marc_ja",
    "ja_leaderboard_xwinograd",
)
class JapaneseLeaderboardMultipleChoiceExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Japanese Leaderboard multiple-choice benchmarks."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask | None = None,
        limit: int | None = None,
        preferred_doc: str | None = None,
        *,
        train_ratio: float,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "japanese_leaderboard_mc"))
        max_items = self._normalize_limit(limit)

        # Rakuten/JGLUE was removed from HF; jcommonsenseqa/jnli/marc_ja used to depend
        # on it. Load shunk031/JGLUE directly when lm-eval can't load the task.
        cfg_task = None
        if lm_eval_task_data is not None:
            cfg = getattr(lm_eval_task_data, "config", None)
            cfg_task = cfg.__dict__.get("task") if (cfg is not None and hasattr(cfg, "__dict__")) else None

        if lm_eval_task_data is None or cfg_task in (
            "ja_leaderboard_jcommonsenseqa",
            "ja_leaderboard_jnli",
            "ja_leaderboard_marc_ja",
        ):
            from datasets import load_dataset
            config_map = {
                "ja_leaderboard_jcommonsenseqa": "JCommonsenseQA",
                "ja_leaderboard_jnli": "JNLI",
                "ja_leaderboard_marc_ja": "MARC-ja",
            }
            target_cfg = cfg_task or "ja_leaderboard_jcommonsenseqa"
            jglue_config = config_map.get(target_cfg, "JCommonsenseQA")
            try:
                from datasets import get_dataset_split_names
                split_names = get_dataset_split_names("shunk031/JGLUE", jglue_config, trust_remote_code=True)
                all_rows = []
                for s in split_names:
                    try:
                        ds_s = load_dataset("shunk031/JGLUE", jglue_config, split=s, trust_remote_code=True)
                        all_rows.extend(list(ds_s))
                    except Exception:
                        continue
                ds = all_rows
            except Exception as exc:
                log.error(f"Failed to load shunk031/JGLUE: {exc}")
                return []
            docs = ds[: (max_items * 4 if max_items else None)]
        else:
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
            log.warning("No valid Japanese Leaderboard MC pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            question = None
            choices = None
            answer_idx = None

            # JCommonsenseQA shunk031 format: question + choice0..choice4 + label
            if "question" in doc and "choice0" in doc and "label" in doc:
                question = str(doc.get("question", "")).strip()
                choices = [str(doc.get(f"choice{i}", "")).strip() for i in range(5) if doc.get(f"choice{i}")]
                answer = doc.get("label", 0)
                answer_idx = int(answer) if isinstance(answer, (int, str)) else 0

            # Format 1: question + choices (JCommonsenseQA)
            elif "question" in doc and "choices" in doc:
                question = str(doc.get("question", "")).strip()
                choices_data = doc.get("choices", [])
                if isinstance(choices_data, list):
                    choices = choices_data
                answer = doc.get("label", 0)
                answer_idx = int(answer) if isinstance(answer, (int, str)) else 0

            # Format 2: sentence1 + sentence2 (JNLI)
            elif "sentence1" in doc and "sentence2" in doc:
                premise = str(doc.get("sentence1", "")).strip()
                hypothesis = str(doc.get("sentence2", "")).strip()
                question = f"Premise: {premise}\nHypothesis: {hypothesis}"
                choices = ["含意", "矛盾", "中立"]
                answer = doc.get("label", 0)
                answer_idx = int(answer) if isinstance(answer, (int, str)) else 0

            # Format 3: sentence + label (MARC-ja)
            elif "sentence" in doc and "label" in doc:
                question = str(doc.get("sentence", "")).strip()
                choices = ["ポジティブ", "ネガティブ"]
                answer = doc.get("label", 0)
                answer_idx = int(answer) if isinstance(answer, (int, str)) else 0

            # Format 4: XWinograd format (sentence with options)
            elif "sentence" in doc and "option1" in doc and "option2" in doc:
                sentence = str(doc.get("sentence", "")).strip()
                option1 = str(doc.get("option1", "")).strip()
                option2 = str(doc.get("option2", "")).strip()
                question = sentence
                choices = [option1, option2]
                answer = doc.get("answer", 0)
                answer_idx = int(answer) if isinstance(answer, (int, str)) else 0

            if not question or not choices or answer_idx is None or not (0 <= answer_idx < len(choices)):
                log.debug("Skipping doc due to missing/invalid fields", extra={"doc": doc})
                return None

            correct = str(choices[answer_idx]).strip()
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = str(choices[incorrect_idx]).strip()

            prompt = f"Question: {question}"

            positive_response = PositiveResponse(model_response=correct)
            negative_response = NegativeResponse(model_response=incorrect)

            return ContrastivePair(
                prompt=prompt,
                positive_response=positive_response,
                negative_response=negative_response,
                label="japanese_leaderboard_mc",
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None
