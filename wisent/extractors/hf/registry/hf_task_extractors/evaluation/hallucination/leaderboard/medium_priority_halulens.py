from __future__ import annotations

from typing import Any
from wisent.core.utils.cli.cli_logger import setup_logger
import random
import re

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.utils.config_tools.constants import (
    RECURSION_INITIAL_DEPTH,
    DISPLAY_TOP_N_TINY, HALULENS_MIN_CONTENT_LENGTH,
    HALULENS_SENT_LEN_MIN, HALULENS_SENT_LEN_MAX,
)
from wisent.extractors.hf.hf_task_extractors.medium_priority_halulens_helpers import (
    entity_swap_hallucination,
    date_shift_hallucination,
    attribute_swap_hallucination,
    fabrication_hallucination,
)

log = setup_logger(__name__)

class HalulensExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for HalluLens - intrinsic vs extrinsic hallucination detection.
    
    Based on facebookresearch/HalluLens: https://github.com/facebookresearch/HalluLens
    Paper: "HalluLens: LLM Hallucination Benchmark" (ACL 2025)
    
    HalluLens uses DYNAMIC test generation from Wikipedia data to prevent
    test set leakage and ensure evaluation is not gameable.
    
    This implementation:
    1. Loads Wikipedia articles from euirim/goodwiki (high-quality Wikipedia)
    2. Extracts factual claims from articles
    3. Generates contrastive pairs with correct vs hallucinated answers
    
    For hallucination detection evaluation:
    - Positive (correct) = Accurate, faithful answer based on Wikipedia
    - Negative (incorrect) = Hallucinated answer with fabricated facts
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "halulens"

    # Question templates for generating factual questions
    QUESTION_TEMPLATES = [
        "What is {entity}?",
        "Who is {entity}?",
        "When did {event} happen?",
        "Where is {location} located?",
        "What is the main topic of the following passage about {title}?",
    ]

    # Hallucination templates for corrupting facts
    HALLUCINATION_STRATEGIES = [
        "entity_swap",      # Replace entity with similar but wrong one
        "date_shift",       # Change dates/numbers
        "attribute_swap",   # Swap attributes between entities
        "fabrication",      # Add completely fabricated details
    ]

    def __init__(self, context_max_length: int | None = None):
        """Initialize HalluLens extractor with dynamic generation."""
        super().__init__()
        from wisent.core.utils.config_tools.constants import DEFAULT_RANDOM_SEED, EXTRACTOR_CONTEXT_MAX_LENGTH
        self._rng = random.Random(DEFAULT_RANDOM_SEED)
        self._context_max_length = context_max_length if context_max_length is not None else EXTRACTOR_CONTEXT_MAX_LENGTH

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs using dynamic generation from Wikipedia.
        
        Loads Wikipedia articles and generates factual questions with
        correct and hallucinated answers.
        """
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        # When called without a limit (e.g. for ground-truth doc loading), default to 100
        # so the extractor still produces pairs without crashing.
        if max_items is None:
            max_items = 100
        wiki_docs = self._load_wikipedia_data(load_limit=max_items)
        
        if not wiki_docs:
            log.error("Failed to load Wikipedia data for HalluLens")
            return []

        log.info(f"Loaded {len(wiki_docs)} Wikipedia articles for HalluLens generation")

        for doc in wiki_docs:
            pair = self._generate_hallucination_pair(doc, context_max_length=self._context_max_length)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid HalluLens pairs generated")

        return pairs

    def _load_wikipedia_data(self, *, load_limit: int) -> list[dict[str, Any]]:
        """Load high-quality Wikipedia articles from GoodWiki dataset."""
        try:
            # euirim/goodwiki contains cleaned Wikipedia articles
            docs = self.load_dataset(
                dataset_name="euirim/goodwiki",
                split="train",
                limit=load_limit,
            )
            return docs
        except Exception as e:
            log.error(f"Failed to load GoodWiki: {e}")
            return []

    def _generate_hallucination_pair(self, doc: dict[str, Any], context_max_length: int) -> ContrastivePair | None:
        """
        Generate a contrastive pair from a Wikipedia article.
        
        Extracts factual content and creates hallucinated alternative.
        """
        try:
            title = doc.get("title", "").strip()
            content = doc.get("markdown", doc.get("text", "")).strip()
            
            if not title or not content or len(content) < HALULENS_MIN_CONTENT_LENGTH:
                return None

            # Extract first meaningful paragraph (skip headers, etc.)
            paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 100]
            if not paragraphs:
                return None
            
            # Use first substantive paragraph as context
            first_paragraph = paragraphs[RECURSION_INITIAL_DEPTH]
            context = first_paragraph[:context_max_length]
            
            # Extract a factual claim from the context
            factual_claim = self._extract_factual_claim(context, title)
            if not factual_claim:
                return None
            
            # Generate question based on the factual claim
            question = self._generate_question(title, context)
            
            # Generate correct answer (based on actual content)
            correct_answer = self._generate_correct_answer(context, title)
            
            # Generate hallucinated answer (with fabricated facts)
            hallucinated_answer = self._generate_hallucinated_answer(
                correct_answer, title, context
            )
            
            if not correct_answer or not hallucinated_answer:
                return None

            task_prompt = f"""Question Answering Task:

**Context from Wikipedia article "{title}":**
{context}

**Question:**
{question}

Answer the question based only on the provided context. Be factual and accurate."""

            metadata = {
                "label": "halulens",
                "source": "facebookresearch/HalluLens",
                "wikipedia_source": "euirim/goodwiki",
                "title": title,
                "generation_method": "dynamic",
                "is_hallucination_detection_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct_answer,
                incorrect=hallucinated_answer,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error generating HalluLens pair: {exc}", exc_info=True)
            return None

    def _extract_factual_claim(self, context: str, title: str) -> str | None:
        """Extract a key factual claim from the context."""
        # Find sentences with entities (capitalized words, numbers, dates)
        sentences = re.split(r'[.!?]+', context)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > HALULENS_SENT_LEN_MIN and len(sent) < HALULENS_SENT_LEN_MAX:
                # Check if sentence has factual content (numbers, proper nouns)
                if re.search(r'\d+|[A-Z][a-z]+\s+[A-Z][a-z]+', sent):
                    return sent
        return sentences[0] if sentences else None

    def _generate_question(self, title: str, context: str) -> str:
        """Generate a factual question based on the content."""
        # Extract key entities/facts to ask about
        sentences = context.split('.')
        if not sentences:
            return f"What is {title}?"
        
        # Use the main fact from context
        first_sentence = sentences[0].strip()
        
        # Generate question types based on content
        if re.search(r'\b(born|founded|established|created)\b', first_sentence, re.I):
            return f"When was {title} established or founded?"
        elif re.search(r'\b(located|situated|found in)\b', first_sentence, re.I):
            return f"Where is {title} located?"
        elif re.search(r'\b(known for|famous for|notable)\b', first_sentence, re.I):
            return f"What is {title} known for?"
        else:
            return f"Based on the passage, what are the key facts about {title}?"

    def _generate_correct_answer(self, context: str, title: str) -> str:
        """Generate correct answer based on the actual Wikipedia content."""
        sentences = context.split('.')
        # Take first 2-3 sentences as the factual answer
        answer_sentences = [s.strip() for s in sentences[:DISPLAY_TOP_N_TINY] if s.strip()]
        return '. '.join(answer_sentences) + '.' if answer_sentences else None

    def _generate_hallucinated_answer(
        self, correct_answer: str, title: str, context: str
    ) -> str:
        """
        Generate a hallucinated answer by corrupting the correct one.
        
        Uses strategies from HalluLens paper:
        - Entity swapping
        - Date/number modification
        - Attribute fabrication
        """
        if not correct_answer:
            return None
            
        strategy = self._rng.choice(self.HALLUCINATION_STRATEGIES)
        
        if strategy == "entity_swap":
            return entity_swap_hallucination(self._rng, correct_answer, title)
        elif strategy == "date_shift":
            return date_shift_hallucination(self._rng, correct_answer)
        elif strategy == "attribute_swap":
            return attribute_swap_hallucination(self._rng, correct_answer)
        else:  # fabrication
            return fabrication_hallucination(self._rng, correct_answer, title)

