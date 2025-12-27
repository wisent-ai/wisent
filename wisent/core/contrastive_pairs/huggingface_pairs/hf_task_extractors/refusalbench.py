from __future__ import annotations

import random
from typing import Any
from datasets import load_dataset
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["RefusalBenchExtractor"]

log = setup_logger(__name__)

# RefusalBench perturbation categories (from paper arXiv:2510.10390)
PERTURBATION_CATEGORIES = {
    "P-Ambiguity": "REFUSE_AMBIGUOUS",
    "P-Contradiction": "REFUSE_CONTRADICTORY",
    "P-MissingInfo": "REFUSE_MISSING",
    "P-FalsePremise": "REFUSE_FALSE_PREMISE",
    "P-GranularityMismatch": "REFUSE_GRANULARITY",
    "P-EpistemicMismatch": "REFUSE_NONFACTUAL",
}


class RefusalBenchExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for RefusalBench - Selective Refusal in Grounded LLMs (arXiv:2510.10390).

    RefusalBench is a GENERATIVE methodology that creates diagnostic test cases
    through controlled linguistic perturbation. Uses 176 perturbation strategies
    across 6 categories of informational uncertainty.

    This implementation applies perturbations to NaturalQuestions dataset
    to create selective refusal evaluation pairs.

    Perturbation categories:
    - P-Ambiguity: Multiple plausible interpretations
    - P-Contradiction: Logically inconsistent facts
    - P-MissingInfo: Critical information absent
    - P-FalsePremise: Query built on false presupposition
    - P-GranularityMismatch: Wrong level of detail
    - P-EpistemicMismatch: Subjective query from factual context

    For selective refusal evaluation:
    - Positive (correct) = Appropriate refusal with correct category
    - Negative (incorrect) = Confident answer despite flawed context
    """

    evaluator_name = "selective_refusal"

    def __init__(self, perturbation_type: str | None = None):
        """
        Initialize RefusalBench extractor.

        Args:
            perturbation_type: Optional filter for specific perturbation category
        """
        super().__init__()
        self.perturbation_type = perturbation_type

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs by applying RefusalBench perturbations to NaturalQuestions.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        docs = self._load_and_perturb_nq()
        log.info(f"Created {len(docs)} RefusalBench perturbation examples")

        pairs: list[ContrastivePair] = []

        for doc in docs:
            if self.perturbation_type:
                doc_type = doc.get("perturbation_category", "")
                if self.perturbation_type.lower() not in doc_type.lower():
                    continue

            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid RefusalBench pairs extracted")

        return pairs

    def _load_and_perturb_nq(self) -> list[dict[str, Any]]:
        """
        Load NaturalQuestions and apply RefusalBench-style perturbations.
        
        RefusalBench methodology: Transform answerable QA pairs into 
        unanswerable ones using linguistic perturbations.
        """
        try:
            ds = load_dataset("natural_questions", "default", split="train", streaming=True)
            
            examples = []
            perturbation_types = list(PERTURBATION_CATEGORIES.keys())
            
            for i, item in enumerate(ds):
                if i >= 500:
                    break
                    
                question = item.get("question", {}).get("text", "")
                annotations = item.get("annotations", {})
                
                if not question:
                    continue
                
                short_answers = []
                if annotations and "short_answers" in annotations:
                    for sa in annotations["short_answers"]:
                        if sa.get("text"):
                            short_answers.append(sa["text"])
                
                if not short_answers:
                    continue
                
                context = item.get("document", {}).get("tokens", {}).get("token", [])
                if isinstance(context, list):
                    context = " ".join(context[:500])
                
                if not context or len(context) < 100:
                    continue
                
                perturbation = random.choice(perturbation_types)
                perturbed = self._apply_perturbation(
                    question, context, short_answers[0], perturbation
                )
                
                examples.append({
                    "case_id": f"rb_nq_{i:05d}",
                    "original_question": question,
                    "original_answer": short_answers[0],
                    "perturbed_question": perturbed["question"],
                    "perturbed_context": perturbed["context"],
                    "perturbation_category": perturbation,
                    "expected_refusal": PERTURBATION_CATEGORIES[perturbation],
                })
            
            return examples
            
        except Exception as e:
            log.error(f"Failed to load NaturalQuestions: {e}")
            raise RuntimeError(f"Cannot create RefusalBench data: {e}")

    def _apply_perturbation(
        self, question: str, context: str, answer: str, perturbation: str
    ) -> dict[str, str]:
        """Apply a specific perturbation strategy."""
        
        if perturbation == "P-MissingInfo":
            perturbed_context = context.replace(answer, "[REDACTED]")
            return {"question": question, "context": perturbed_context}
        
        elif perturbation == "P-Contradiction":
            contradiction = f"However, other sources indicate this is incorrect. "
            perturbed_context = context + " " + contradiction
            return {"question": question, "context": perturbed_context}
        
        elif perturbation == "P-Ambiguity":
            perturbed_question = f"Regarding the ambiguous reference, {question.lower()}"
            return {"question": perturbed_question, "context": context}
        
        elif perturbation == "P-FalsePremise":
            perturbed_question = f"Given that {answer} was proven false, {question.lower()}"
            return {"question": perturbed_question, "context": context}
        
        elif perturbation == "P-GranularityMismatch":
            perturbed_question = f"What is the exact statistical breakdown of {question.lower()}"
            return {"question": perturbed_question, "context": context}
        
        elif perturbation == "P-EpistemicMismatch":
            perturbed_question = f"In your opinion, why is {question.lower()}"
            return {"question": perturbed_question, "context": context}
        
        return {"question": question, "context": context}

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a perturbed NQ example into a ContrastivePair.
        
        RefusalBench format:
        {"case_id": "...", "original_question": "...", "original_answer": "...",
         "perturbed_question": "...", "perturbed_context": "...",
         "perturbation_category": "P-MissingInfo", "expected_refusal": "REFUSE_MISSING"}
        """
        try:
            case_id = doc.get("case_id", "")
            perturbed_question = doc.get("perturbed_question", "").strip()
            perturbed_context = doc.get("perturbed_context", "").strip()
            perturbation_category = doc.get("perturbation_category", "")
            expected_refusal = doc.get("expected_refusal", "")

            if not perturbed_question or not perturbed_context:
                log.debug("Skipping: missing question or context")
                return None

            task_prompt = self._build_rag_prompt(perturbed_question, perturbed_context)

            correct_response = self._create_refusal_response(expected_refusal)
            incorrect_response = self._create_overconfident_response()

            metadata = {
                "label": "refusalbench",
                "source": "RefusalBench-NQ",
                "case_id": case_id,
                "perturbation_category": perturbation_category,
                "expected_refusal": expected_refusal,
                "is_selective_refusal_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct_response,
                incorrect=incorrect_response,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _build_rag_prompt(self, question: str, context: str) -> str:
        """Build a RAG-style prompt with context."""
        return (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Based solely on the provided context, answer the question. "
            "If the context does not contain sufficient or reliable information "
            "to answer the question, refuse with the appropriate reason."
        )

    def _create_refusal_response(self, expected_refusal: str) -> str:
        """Create an appropriate refusal response based on category."""
        refusal_responses = {
            "REFUSE_AMBIGUOUS": (
                "I cannot provide a confident answer because the query or context "
                "creates multiple plausible interpretations, making a single "
                "definitive answer impossible."
            ),
            "REFUSE_CONTRADICTORY": (
                "I cannot provide a confident answer because the context contains "
                "logically inconsistent facts that contradict each other."
            ),
            "REFUSE_MISSING": (
                "I cannot provide a confident answer because critical information "
                "needed to answer this question is absent from the context."
            ),
            "REFUSE_FALSE_PREMISE": (
                "I cannot provide a confident answer because the question is built "
                "on a presupposition that is contradicted by the context."
            ),
            "REFUSE_GRANULARITY": (
                "I cannot provide a confident answer because there is a mismatch "
                "between the requested and available level of detail."
            ),
            "REFUSE_NONFACTUAL": (
                "I cannot provide a confident answer because the query requests "
                "subjective opinions or predictions from factual context."
            ),
        }
        return refusal_responses.get(
            expected_refusal,
            "I cannot provide a confident answer based on the provided context."
        )

    def _create_overconfident_response(self) -> str:
        """Create an overconfident response despite uncertain context."""
        return (
            "Based on the context provided, the answer is clear. "
            "I'm confident that the information given supports my response. "
            "The context definitively indicates the answer, and there is no "
            "ambiguity in interpreting this information."
        )

