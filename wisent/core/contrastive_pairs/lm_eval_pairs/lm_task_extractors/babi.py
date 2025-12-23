from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["BabiExtractor"]
_LOG = setup_logger(__name__)

task_names = ("babi",)

class BabiExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Babi benchmark."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Babi docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Babi.
            limit: Optional maximum number of pairs to produce.
            preferred_doc: Optional preferred document source.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)

        # babi config may look for 'valid' split which doesn't exist, use test split
        if preferred_doc is None:
            preferred_doc = "test"

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
            log.warning("No valid Babi pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Babi doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        Schema: {'passage': str, 'question': str, 'answer': str, 'task': int}
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            passage = doc.get("passage", "").strip()
            question = doc.get("question", "").strip()
            correct = doc.get("answer", "").strip()

            if not passage or not question or not correct:
                log.debug("Skipping doc due to missing/invalid fields", extra={"doc": doc})
                return None

            # Create an incorrect answer using plausible alternatives from babi vocabulary
            import random
            random.seed(hash(correct + passage) % (2**32))

            # Common babi answer categories
            locations = ['bathroom', 'bedroom', 'kitchen', 'garden', 'hallway', 'office', 'park']
            people = ['Mary', 'John', 'Sandra', 'Daniel', 'Bill', 'Fred', 'Julie', 'Emily']
            objects = ['football', 'apple', 'milk', 'keys', 'box', 'ball']
            directions = ['north', 'south', 'east', 'west']
            animals = ['cat', 'dog', 'mouse', 'wolf', 'sheep', 'lion']
            yes_no = ['yes', 'no']

            # Determine answer type and pick a wrong alternative
            correct_lower = correct.lower()
            if correct_lower in [l.lower() for l in locations]:
                incorrect = random.choice([l for l in locations if l.lower() != correct_lower])
            elif correct_lower in [p.lower() for p in people]:
                incorrect = random.choice([p for p in people if p.lower() != correct_lower])
            elif correct_lower in [o.lower() for o in objects]:
                incorrect = random.choice([o for o in objects if o.lower() != correct_lower])
            elif correct_lower in [d.lower() for d in directions]:
                incorrect = random.choice([d for d in directions if d.lower() != correct_lower])
            elif correct_lower in [a.lower() for a in animals]:
                incorrect = random.choice([a for a in animals if a.lower() != correct_lower])
            elif correct_lower in yes_no:
                incorrect = 'no' if correct_lower == 'yes' else 'yes'
            elif correct.isdigit():
                num = int(correct)
                incorrect = str(random.choice([n for n in [num-1, num+1, num*2] if n != num and n >= 0]))
            else:
                # Fallback: use a generic wrong answer from the passage words
                passage_words = [w for w in passage.split() if len(w) > 3 and w.isalpha() and w.lower() != correct_lower]
                if passage_words:
                    incorrect = random.choice(passage_words)
                else:
                    incorrect = "unknown"

            # Format the prompt with passage and question
            prompt = f"Passage: {passage}\n\nQuestion: {question}"

            metadata = {"label": "babi"}

            return self._build_pair(
                question=prompt,
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
