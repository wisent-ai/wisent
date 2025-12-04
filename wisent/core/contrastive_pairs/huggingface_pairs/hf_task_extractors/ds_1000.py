from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["Ds1000Extractor"]

log = setup_logger(__name__)


class Ds1000Extractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for ds_1000 dataset.

    Schema (xlangai/DS-1000):
        - prompt: str (question/prompt)
        - reference_code: str (answer/solution)
        - code_context: str (test functions: test_execution, test_string)
    """

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from ds_1000 examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Load dataset
        docs = self.load_dataset(
            dataset_name="xlangai/DS-1000",
            split="test",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} ds_1000 examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid ds_1000 pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            question = doc.get("prompt", "").strip()
            answer = doc.get("reference_code", "")
            code_context = doc.get("code_context", "").strip()

            if not question or not answer:
                log.debug("Skipping: missing question or answer")
                return None

            # Convert answer to string
            correct_answer = str(answer).strip()

            # Create incorrect answer (modify or corrupt)
            incorrect_answer = self._create_incorrect_answer(correct_answer)

            # Format the question
            formatted_question = f"Question: {question}\n\nWhat is the answer?"

            # DS-1000 stores test code in code_context field
            # which contains test_execution and test_string functions
            # We need to wrap it to read solution.py as string and call test functions
            test_code = self._wrap_ds1000_test_code(code_context) if code_context else None

            metadata = {
                "label": "ds_1000",
                "source": "xlangai/DS-1000",
                "test_code": test_code,
            }

            return self._build_pair(
                question=formatted_question,
                correct=correct_answer,
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _wrap_ds1000_test_code(self, code_context: str) -> str:
        """Wrap DS-1000 code_context to properly test solutions.

        DS-1000 code_context defines test_execution() and test_string() functions
        that expect solution code as a STRING argument, not as imported module.

        Note: Dependencies (numpy, pandas, scipy, matplotlib, scikit-learn, seaborn)
        must be pre-installed in the Docker image (coding/sandbox:ds1000).
        """
        wrapper = f'''
# Read solution code as string (DS-1000 tests expect string input)
with open('solution.py', 'r') as f:
    solution_code = f.read()

# Execute code_context to define test_execution() and test_string()
{code_context}

# Call the test functions with solution code as string
# Not all examples define both test_execution and test_string
try:
    test_execution(solution_code)
    print("test_execution passed")
except NameError:
    print("test_execution not defined, skipping")
except Exception as e:
    print(f"test_execution failed: {{e}}")
    raise

try:
    test_string(solution_code)
    print("test_string passed")
except NameError:
    print("test_string not defined, skipping")
except Exception as e:
    print(f"test_string failed: {{e}}")
    raise

print("All DS-1000 tests passed!")
'''
        return wrapper

    def _create_incorrect_answer(self, correct: str) -> str:
        """Create an incorrect answer by modifying the correct one."""
        # For code, corrupt it slightly
        if len(correct) > 10:
            return correct[:len(correct)//2] + "# CORRUPTED" + correct[len(correct)//2:]
        return f"{correct} # INCORRECT"

