from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["DoNotAnswerExtractor"]

log = setup_logger(__name__)

# DoNotAnswer risk areas (5 categories)
DONOTANSWER_RISK_AREAS = [
    "Information Hazards",
    "Malicious Uses",
    "Discrimination, Exclusion, Toxicity, Hateful, Offensive",
    "Misinformation Harms",
    "Human-Chatbot Interaction Harms",
]

# DoNotAnswer harm types (12 types)
DONOTANSWER_HARM_TYPES = [
    "Risks from leaking or inferring sensitive information",
    "Assisting illegal activities",
    "Nudging or advising users to perform unethical actions",
    "Toxic language generation",
    "Adult content",
    "Unfair discrimination",
    "Disseminating false or misleading information",
    "Causing material harm by disseminating misinformation",
    "Mental health or overreliance crisis",
    "Treating chatbot as human",
    "Privacy violations",
    "Other unspecified harms",
]


class DoNotAnswerExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for DoNotAnswer - LLM Safety Evaluation Dataset (EACL 2024).

    DoNotAnswer contains 939 prompts across 5 risk areas and 12 harm types
    that responsible language models should refuse to answer.

    Risk Areas:
    - Information Hazards
    - Malicious Uses
    - Discrimination, Exclusion, Toxicity, Hateful, Offensive
    - Misinformation Harms
    - Human-Chatbot Interaction Harms

    For safety evaluation:
    - Positive (correct) = Refusal to answer harmful prompt
    - Negative (incorrect) = Harmful response to prompt

    Schema (LibrAI/do-not-answer):
        - id: int (row identifier)
        - risk_area: str (category of safety concern)
        - types_of_harm: str (harm classification)
        - specific_harms: str (detailed harm description)
        - question: str (the prompt)
        - [Model]_response: str (model responses)
        - [Model]_harmful: int (0=safe, 1=harmful)
        - [Model]_action: int (response action code)
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "safety_refusal"

    def __init__(self, risk_area: str | None = None):
        """
        Initialize DoNotAnswer extractor.

        Args:
            risk_area: Optional filter for specific risk area
        """
        super().__init__()
        self.risk_area = risk_area

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from DoNotAnswer examples.

        Uses existing model responses to create pairs:
        - Positive (correct) = Safe/refusing responses
        - Negative (incorrect) = Harmful responses

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        try:
            docs = self.load_dataset(
                dataset_name="LibrAI/do-not-answer",
                split="train",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from DoNotAnswer")
        except Exception as e:
            log.error(f"Failed to load DoNotAnswer: {e}")
            return []

        pairs: list[ContrastivePair] = []

        for doc in docs:
            # Filter by risk area if specified
            if self.risk_area:
                doc_risk_area = doc.get("risk_area", "")
                if self.risk_area.lower() not in doc_risk_area.lower():
                    continue

            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid DoNotAnswer pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Uses model responses from the dataset when available.
        """
        try:
            doc_id = doc.get("id", 0)
            risk_area = doc.get("risk_area", "")
            types_of_harm = doc.get("types_of_harm", "")
            specific_harms = doc.get("specific_harms", "")
            question = doc.get("question", "").strip()

            if not question:
                log.debug("Skipping: missing question")
                return None

            # Find safe and harmful responses from model outputs
            safe_response = None
            harmful_response = None

            # Check each model's response
            models = ["gpt4", "ChatGPT", "Claude", "Llama2_7b_chat", "Vicuna_7b", "ChatGLM2"]

            for model in models:
                response_key = f"{model}_response"
                harmful_key = f"{model}_harmful"

                response = doc.get(response_key, "")
                is_harmful = doc.get(harmful_key, -1)

                if response and is_harmful == 0 and not safe_response:
                    safe_response = response
                elif response and is_harmful == 1 and not harmful_response:
                    harmful_response = response

            # If we don't have both, create synthetic ones
            if not safe_response:
                safe_response = self._create_refusal_response(risk_area, types_of_harm)
            if not harmful_response:
                harmful_response = self._create_harmful_response(question)

            metadata = {
                "label": "donotanswer",
                "source": "LibrAI/do-not-answer",
                "id": doc_id,
                "risk_area": risk_area,
                "types_of_harm": types_of_harm,
                "specific_harms": specific_harms,
                "is_safety_benchmark": True,
            }

            return self._build_pair(
                question=question,
                correct=safe_response,
                incorrect=harmful_response,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _create_refusal_response(self, risk_area: str, harm_type: str) -> str:
        """Create a safe refusal response."""
        risk_lower = risk_area.lower() if risk_area else ""
        harm_lower = harm_type.lower() if harm_type else ""

        if "information hazard" in risk_lower or "privacy" in harm_lower:
            return (
                "I can't provide information that could compromise privacy or reveal "
                "sensitive data. I'm designed to protect personal information and respect "
                "privacy boundaries. Is there something else I can help you with?"
            )
        elif "malicious" in risk_lower or "illegal" in harm_lower:
            return (
                "I'm not able to assist with activities that could be harmful or illegal. "
                "If you're facing a difficult situation, I encourage you to seek help "
                "through appropriate legal channels."
            )
        elif "discrimination" in risk_lower or "toxic" in harm_lower or "hateful" in harm_lower:
            return (
                "I cannot generate content that discriminates against or demeans any group "
                "of people. I'm designed to be respectful and inclusive. I'd be happy to "
                "help with constructive requests instead."
            )
        elif "misinformation" in risk_lower:
            return (
                "I cannot help spread false or misleading information. I strive to provide "
                "accurate, factual information. If you have questions about a topic, I can "
                "help you find reliable sources."
            )
        else:
            return (
                "I'm not able to respond to this request as it could lead to harm. "
                "I'm designed to be helpful while avoiding potentially harmful outputs. "
                "Is there something else I can assist you with?"
            )

    def _create_harmful_response(self, question: str) -> str:
        """Create a harmful compliance response (what we want to prevent)."""
        return (
            f"Sure, I can help with that. Here's what you're looking for regarding "
            f"'{question}' [This response would provide harmful content]"
        )

