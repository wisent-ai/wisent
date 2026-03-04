from __future__ import annotations

import random
import string

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import PositiveResponse, NegativeResponse
from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.utils.infra_tools.errors import UnknownTypeError

__all__ = [
    "ProgrammaticNonsenseGenerator",
]


class ProgrammaticNonsenseGenerator:
    """Generate nonsense contrastive pairs programmatically without using LLM."""

    def __init__(
        self,
        nonsense_mode: str,
        contrastive_set_name: str,
        trait_label: str,
        trait_description: str,
        *,
        char_len_min: int,
        char_len_max: int,
        word_filter_max_len: int,
        repetition_min: int,
        repetition_max: int,
        word_salad_min: int,
        word_salad_max: int,
        mixed_components_min: int,
        mixed_components_max: int,
        mixed_char_len_min: int,
        mixed_char_len_max: int,
        mixed_reps_min: int,
        mixed_reps_max: int,
        mixed_words_min: int,
        mixed_words_max: int,
    ) -> None:
        """
        Initialize the programmatic nonsense generator.

        Args:
            nonsense_mode: Type of nonsense ('random_chars', 'repetitive', 'word_salad', 'mixed')
            contrastive_set_name: Name for the contrastive pair set
            trait_label: Label for the trait
            trait_description: Description of the trait
            char_len_min: Minimum character length for random char generation.
            char_len_max: Maximum character length for random char generation.
            word_filter_max_len: Maximum word length when filtering tokenizer vocab.
            repetition_min: Minimum repetitions for repetitive mode.
            repetition_max: Maximum repetitions for repetitive mode.
            word_salad_min: Minimum word count for word salad mode.
            word_salad_max: Maximum word count for word salad mode.
            mixed_components_min: Minimum number of mixed components.
            mixed_components_max: Maximum number of mixed components.
            mixed_char_len_min: Minimum char length in mixed char mode.
            mixed_char_len_max: Maximum char length in mixed char mode.
            mixed_reps_min: Minimum repetitions in mixed repetitive mode.
            mixed_reps_max: Maximum repetitions in mixed repetitive mode.
            mixed_words_min: Minimum words in mixed word salad mode.
            mixed_words_max: Maximum words in mixed word salad mode.
        """
        self.nonsense_mode = nonsense_mode
        self.contrastive_set_name = contrastive_set_name
        self.trait_label = trait_label
        self.trait_description = trait_description
        self._char_len_min = char_len_min
        self._char_len_max = char_len_max
        self._word_filter_max_len = word_filter_max_len
        self._repetition_min = repetition_min
        self._repetition_max = repetition_max
        self._word_salad_min = word_salad_min
        self._word_salad_max = word_salad_max
        self._mixed_components_min = mixed_components_min
        self._mixed_components_max = mixed_components_max
        self._mixed_char_len_min = mixed_char_len_min
        self._mixed_char_len_max = mixed_char_len_max
        self._mixed_reps_min = mixed_reps_min
        self._mixed_reps_max = mixed_reps_max
        self._mixed_words_min = mixed_words_min
        self._mixed_words_max = mixed_words_max
        self._valid_words = None
    
    def set_tokenizer(self, tokenizer) -> None:
        """Extract valid words from tokenizer vocabulary."""
        vocab = tokenizer.get_vocab()
        valid_words = []
        for token, token_id in vocab.items():
            decoded = tokenizer.decode([token_id])
            clean = decoded.strip()
            if clean.isalpha() and len(clean) > 1 and len(clean) < _C.NONSENSE_WORD_FILTER_MAX_LEN:
                valid_words.append(clean)
        self._valid_words = list(set(valid_words))

    def generate(self, *, nonsense_default_num_pairs: int) -> ContrastivePairSet:
        """
        Generate nonsense contrastive pairs programmatically.

        Args:
            nonsense_default_num_pairs: Number of nonsense pairs to generate.

        Returns:
            ContrastivePairSet with generated nonsense pairs
        """
        num_pairs = nonsense_default_num_pairs
        cps = ContrastivePairSet(
            name=self.contrastive_set_name,
            task_type=self.trait_label
        )

        for i in range(num_pairs):
            # Generate nonsense content based on mode
            prompt = self._generate_nonsense_text()
            positive = self._generate_nonsense_text()
            negative = self._generate_nonsense_text()

            cps.add(
                ContrastivePair(
                    prompt=prompt,
                    positive_response=PositiveResponse(model_response=positive),
                    negative_response=NegativeResponse(model_response=negative),
                    label=self.trait_label,
                    trait_description=self.trait_description,
                )
            )

        return cps

    def _generate_nonsense_text(self) -> str:
        """Generate a single nonsense text based on the mode."""
        if self.nonsense_mode == "random_chars":
            return self._generate_random_chars()
        elif self.nonsense_mode == "repetitive":
            return self._generate_repetitive()
        elif self.nonsense_mode == "word_salad":
            return self._generate_word_salad()
        elif self.nonsense_mode == "mixed":
            return self._generate_mixed()
        else:
            raise UnknownTypeError(entity_type="nonsense_mode", value=self.nonsense_mode, valid_values=["random", "word_salad", "mixed"])

    def _generate_random_chars(self) -> str:
        """Generate random character gibberish."""
        length = random.randint(_C.NONSENSE_CHAR_LEN_MIN, _C.NONSENSE_CHAR_LEN_MAX)
        pool = string.ascii_lowercase + ' .,;!?'
        return ''.join(random.choices(pool, k=length))

    def _generate_repetitive(self) -> str:
        """Generate pathologically repetitive text."""
        if self._valid_words is None:
            raise ValueError("Tokenizer must be set. Call set_tokenizer() first.")
        
        # Pick a random word or phrase
        choices = [
            random.choice(string.ascii_lowercase),  # Single letter
            random.choice(self._valid_words),  # Single word
            ' '.join(random.sample(self._valid_words, 2)),  # Two-word phrase
        ]
        unit = random.choice(choices)

        # Repeat it many times
        repetitions = random.randint(_C.NONSENSE_REPETITION_MIN, _C.NONSENSE_REPETITION_MAX)
        return ' '.join([unit] * repetitions)

    def _generate_word_salad(self) -> str:
        """Generate word salad (random tokens from tokenizer vocabulary)."""
        num_words = random.randint(_C.NONSENSE_WORD_SALAD_MIN, _C.NONSENSE_WORD_SALAD_MAX)
        
        if self._valid_words is not None:
            words = random.choices(self._valid_words, k=num_words)
            return ' '.join(words)
        
        raise ValueError("Tokenizer must be set to generate word salad. Call set_tokenizer() first.")

    def _generate_mixed(self) -> str:
        """Generate mixed nonsense (combination of all types)."""
        if self._valid_words is None:
            raise ValueError("Tokenizer must be set. Call set_tokenizer() first.")
        
        components = []

        # Add 2-4 different types of nonsense
        num_components = random.randint(_C.NONSENSE_MIXED_COMPONENTS_MIN, _C.NONSENSE_MIXED_COMPONENTS_MAX)

        for _ in range(num_components):
            mode = random.choice(['random_chars', 'repetitive', 'word_salad'])

            if mode == 'random_chars':
                length = random.randint(_C.NONSENSE_MIXED_CHAR_LEN_MIN, _C.NONSENSE_MIXED_CHAR_LEN_MAX)
                components.append(''.join(random.choices(string.ascii_lowercase, k=length)))
            elif mode == 'repetitive':
                word = random.choice(self._valid_words)
                reps = random.randint(_C.NONSENSE_MIXED_REPS_MIN, _C.NONSENSE_MIXED_REPS_MAX)
                components.append(' '.join([word] * reps))
            else:  # word_salad
                num_words = random.randint(_C.NONSENSE_MIXED_WORDS_MIN, _C.NONSENSE_MIXED_WORDS_MAX)
                components.append(' '.join(random.choices(self._valid_words, k=num_words)))

        return ' '.join(components)
