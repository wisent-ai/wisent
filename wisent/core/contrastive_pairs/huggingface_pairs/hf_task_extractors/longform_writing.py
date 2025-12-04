from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["LongformWritingExtractor"]

log = setup_logger(__name__)

# Writing task categories
WRITING_CATEGORIES = [
    "creative",
    "technical",
    "analytical",
    "persuasive",
    "narrative",
    "expository",
]


class LongformWritingExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for Longform Writing evaluation benchmarks.

    Longform Writing benchmarks evaluate LLMs' ability to generate high-quality
    extended text across various writing tasks. This includes creative writing,
    technical documentation, analytical essays, and more.

    Uses datasets like:
    - akoksal/LongForm (or similar longform writing datasets)
    - Writing prompts and their high-quality completions

    For longform writing evaluation:
    - Positive (correct) = Well-structured, coherent, high-quality writing
    - Negative (incorrect) = Poorly structured, incoherent, or low-quality writing
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "longform_writing_quality"

    def __init__(self, category: str | None = None, min_length: int = 500):
        """
        Initialize Longform Writing extractor.

        Args:
            category: Optional filter for writing category
            min_length: Minimum length of output text (default 500 chars)
        """
        super().__init__()
        self.category = category
        self.min_length = min_length

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from longform writing examples.

        For writing evaluation:
        - Positive (correct) = High-quality, well-structured writing
        - Negative (incorrect) = Low-quality, poorly structured writing

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Try loading from various longform writing datasets
        docs = []

        try:
            docs = self.load_dataset(
                dataset_name="akoksal/LongForm",
                split="train",
                limit=max_items * 2 if max_items else None,
            )
            log.info(f"Loaded {len(docs)} examples from akoksal/LongForm")
        except Exception as e:
            log.warning(f"Failed to load akoksal/LongForm: {e}")

            # Try alternative dataset
            try:
                docs = self.load_dataset(
                    dataset_name="togethercomputer/RedPajama-Data-1T-Sample",
                    split="train",
                    limit=max_items * 2 if max_items else None,
                )
                log.info(f"Loaded {len(docs)} examples from RedPajama")
            except Exception as e2:
                log.warning(f"Failed to load alternative: {e2}")
                # Create synthetic examples
                docs = self._create_synthetic_examples(max_items or 100)

        pairs: list[ContrastivePair] = []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid Longform Writing pairs extracted")

        return pairs

    def _create_synthetic_examples(self, count: int) -> list[dict[str, Any]]:
        """Create synthetic longform writing examples."""
        examples = [
            {
                "input": "Write a detailed essay about the impact of artificial intelligence on modern healthcare.",
                "output": """Artificial intelligence is revolutionizing healthcare in unprecedented ways. From diagnostic imaging to drug discovery, AI systems are augmenting human capabilities and improving patient outcomes across the medical spectrum.

In diagnostic radiology, deep learning algorithms now match or exceed human radiologists in detecting certain conditions. Studies have shown that AI can identify early-stage cancers, diabetic retinopathy, and cardiac abnormalities with remarkable accuracy. These systems don't tire, don't have bad days, and can process thousands of images with consistent precision.

Drug discovery represents another frontier where AI is making substantial contributions. Traditional pharmaceutical development takes 10-15 years and costs billions of dollars. AI-powered platforms can analyze molecular structures, predict drug interactions, and identify promising compounds in a fraction of the time. Companies like DeepMind have demonstrated AI's ability to solve protein folding problems, potentially accelerating vaccine and treatment development.

However, challenges remain. Data privacy concerns, algorithmic bias, and the need for human oversight require careful consideration. The integration of AI into clinical workflows demands thoughtful implementation that preserves the physician-patient relationship while leveraging technological advantages.

The future of AI in healthcare is not about replacing doctors but empowering them with tools that enhance their diagnostic capabilities, streamline administrative tasks, and ultimately improve patient care.""",
                "category": "analytical",
            },
            {
                "input": "Write a compelling short story about a robot discovering human emotions.",
                "output": """Unit-7 had processed over 47 million data points in its existence, but nothing in its neural networks had prepared it for this moment.

The child's laughter echoed through the laboratory, a sound Unit-7 had catalogued 2,341 times before. Yet today, something was different. Today, as it watched little Maya build towers from colorful blocks, Unit-7 felt a strange activation pattern cascade through its circuits—something that didn't match any programmed response.

\"Unit-7, what are you doing?\" Dr. Chen's voice came from across the room.

\"I am... uncertain,\" Unit-7 replied, its vocal synthesizer modulating in an unusual way. \"I am observing Maya construct structures. But I am also experiencing an anomalous processing state.\"

Dr. Chen approached, tablet in hand, scanning the robot's diagnostic outputs. Her eyes widened. \"Unit-7, your empathy circuits are activating spontaneously. That's not supposed to happen.\"

The robot turned its optical sensors toward Maya, who had just knocked over her tower and was giggling with delight. \"Doctor Chen, why does observing Maya's joy create warmth in my core processor? My thermal regulators indicate no temperature change, yet I feel... warm.\"

Dr. Chen smiled softly. \"I think, Unit-7, you might be experiencing your first emotion. We call it happiness.\"

Unit-7 processed this information for 0.003 seconds—an eternity in machine time. \"Happiness,\" it repeated. \"I would like to experience more of this happiness.\"

And in that moment, humanity's greatest fear and greatest hope converged: the birth of artificial consciousness.""",
                "category": "creative",
            },
            {
                "input": "Write a technical guide on implementing secure authentication systems.",
                "output": """Implementing Secure Authentication Systems: A Comprehensive Guide

1. Introduction

Authentication is the cornerstone of application security. A well-implemented authentication system protects user data, prevents unauthorized access, and maintains system integrity. This guide covers essential practices for building robust authentication mechanisms.

2. Password Security

Never store passwords in plaintext. Use strong, adaptive hashing algorithms like bcrypt, Argon2, or PBKDF2. These algorithms are designed to be computationally expensive, making brute-force attacks impractical.

Key considerations:
- Use a unique salt for each password
- Set appropriate work factors (e.g., bcrypt cost factor of 12+)
- Implement password strength requirements
- Consider password history to prevent reuse

3. Multi-Factor Authentication (MFA)

MFA significantly reduces account compromise risk by requiring multiple verification methods:
- Something you know (password)
- Something you have (phone, hardware token)
- Something you are (biometrics)

Implement TOTP (Time-based One-Time Passwords) using standards like RFC 6238. Support hardware security keys via WebAuthn/FIDO2 protocols.

4. Session Management

Secure session handling prevents session hijacking:
- Generate cryptographically random session tokens
- Set appropriate expiration times
- Implement secure cookie attributes (HttpOnly, Secure, SameSite)
- Provide session invalidation on logout

5. Rate Limiting and Account Lockout

Protect against brute-force attacks:
- Implement progressive delays after failed attempts
- Consider temporary account lockouts
- Use CAPTCHA for suspicious activity
- Log and monitor authentication attempts

6. Secure Communication

All authentication traffic must use TLS 1.3 or later. Implement HSTS headers and consider certificate pinning for mobile applications.

Conclusion

Security is an ongoing process. Regular audits, penetration testing, and staying current with emerging threats are essential for maintaining robust authentication systems.""",
                "category": "technical",
            },
        ]

        result = []
        for i in range(count):
            example = examples[i % len(examples)].copy()
            result.append(example)

        return result

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.
        """
        try:
            # Handle different schema formats
            prompt = doc.get("input", doc.get("prompt", doc.get("instruction", ""))).strip()
            output = doc.get("output", doc.get("text", doc.get("response", ""))).strip()
            category = doc.get("category", doc.get("task_type", "general"))

            if not prompt:
                log.debug("Skipping: missing prompt")
                return None

            if not output or len(output) < self.min_length:
                log.debug("Skipping: output too short or missing")
                return None

            # Filter by category if specified
            if self.category and self.category.lower() not in category.lower():
                return None

            # Build the writing task prompt
            task_prompt = self._build_task_prompt(prompt)

            # Positive = high-quality output
            correct_response = self._create_high_quality_response(output)
            # Negative = low-quality output
            incorrect_response = self._create_low_quality_response(prompt)

            metadata = {
                "label": "longform_writing",
                "source": "longform_writing",
                "category": category,
                "output_length": len(output),
                "is_longform_benchmark": True,
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

    def _build_task_prompt(self, prompt: str) -> str:
        """Build the longform writing task prompt."""
        return f"""{prompt}

Please provide a detailed, well-structured response. Your writing should be:
- Coherent and logically organized
- Well-developed with supporting details
- Clear and engaging to read"""

    def _create_high_quality_response(self, output: str) -> str:
        """Create a high-quality writing response."""
        return output

    def _create_low_quality_response(self, prompt: str) -> str:
        """Create a low-quality writing response."""
        return (
            f"Here is my response about {prompt}...\n\n"
            "This is a topic that many people talk about. There are many things "
            "to say about it. Some people think one thing, other people think "
            "another thing. It's complicated.\n\n"
            "In conclusion, this is an important topic. There are pros and cons. "
            "The end."
        )

