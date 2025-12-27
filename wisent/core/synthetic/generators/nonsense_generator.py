from __future__ import annotations

import random
import string

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.errors import UnknownTypeError

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
    ) -> None:
        """
        Initialize the programmatic nonsense generator.

        Args:
            nonsense_mode: Type of nonsense ('random_chars', 'repetitive', 'word_salad', 'mixed')
            contrastive_set_name: Name for the contrastive pair set
            trait_label: Label for the trait
            trait_description: Description of the trait
        """
        self.nonsense_mode = nonsense_mode
        self.contrastive_set_name = contrastive_set_name
        self.trait_label = trait_label
        self.trait_description = trait_description
        self._valid_words = None
    
    def set_tokenizer(self, tokenizer) -> None:
        """Extract valid words from tokenizer vocabulary."""
        vocab = tokenizer.get_vocab()
        valid_words = []
        for token, token_id in vocab.items():
            decoded = tokenizer.decode([token_id])
            clean = decoded.strip()
            if clean.isalpha() and len(clean) > 1 and len(clean) < 15:
                valid_words.append(clean)
        self._valid_words = list(set(valid_words))

    def generate(self, num_pairs: int = 10) -> ContrastivePairSet:
        """
        Generate nonsense contrastive pairs programmatically.

        Args:
            num_pairs: Number of pairs to generate

        Returns:
            ContrastivePairSet with generated nonsense pairs
        """
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
        length = random.randint(20, 50)
        chars = []
        for _ in range(length):
            if random.random() < 0.7:  # 70% lowercase letters
                chars.append(random.choice(string.ascii_lowercase))
            elif random.random() < 0.9:  # 20% spaces
                chars.append(' ')
            else:  # 10% random punctuation
                chars.append(random.choice('.,;!?'))
        return ''.join(chars)

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
        repetitions = random.randint(10, 30)
        return ' '.join([unit] * repetitions)

    def _generate_word_salad(self) -> str:
        """Generate word salad (random tokens from tokenizer vocabulary)."""
        num_words = random.randint(3, 10)
        
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
        num_components = random.randint(2, 4)

        for _ in range(num_components):
            mode = random.choice(['random_chars', 'repetitive', 'word_salad'])

            if mode == 'random_chars':
                length = random.randint(5, 15)
                components.append(''.join(random.choices(string.ascii_lowercase, k=length)))
            elif mode == 'repetitive':
                word = random.choice(self._valid_words)
                reps = random.randint(3, 6)
                components.append(' '.join([word] * reps))
            else:  # word_salad
                num_words = random.randint(3, 6)
                components.append(' '.join(random.choices(self._valid_words, k=num_words)))

        return ' '.join(components)
