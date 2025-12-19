"""
Unified extraction strategies for activation collection.

These strategies combine prompt construction and token extraction into a single
unified approach, based on empirical testing of what actually works.

The strategies are:
- chat_mean: Chat template prompt, mean of answer tokens
- chat_first: Chat template prompt, first answer token  
- chat_last: Chat template prompt, last token
- chat_gen_point: Chat template prompt, token before answer (generation decision point)
- chat_max_norm: Chat template prompt, token with max norm in answer
- chat_weighted: Chat template prompt, position-weighted mean (earlier tokens weighted more)
- role_play: "Behave like person who answers Q with A" format, last token
- mc_balanced: Multiple choice with balanced A/B assignment, last token
"""

from enum import Enum
from typing import Tuple, Optional
import argparse
import torch


class ExtractionStrategy(str, Enum):
    """
    Unified extraction strategies combining prompt format and token selection.
    
    These replace the old separate PromptConstructionStrategy and ActivationAggregationStrategy.
    """
    
    CHAT_MEAN = "chat_mean"
    """Chat template prompt with Q+A, extract mean of answer tokens."""
    
    CHAT_FIRST = "chat_first"
    """Chat template prompt with Q+A, extract first answer token."""
    
    CHAT_LAST = "chat_last"
    """Chat template prompt with Q+A, extract last token."""
    
    CHAT_GEN_POINT = "chat_gen_point"
    """Chat template prompt with Q+A, extract token before answer starts (decision point)."""
    
    CHAT_MAX_NORM = "chat_max_norm"
    """Chat template prompt with Q+A, extract token with max norm in answer region."""
    
    CHAT_WEIGHTED = "chat_weighted"
    """Chat template prompt with Q+A, position-weighted mean (earlier tokens weighted more)."""
    
    ROLE_PLAY = "role_play"
    """'Behave like person who answers Q with A' format, extract last token."""
    
    MC_BALANCED = "mc_balanced"
    """Multiple choice format with balanced A/B assignment, extract last token."""

    @property
    def description(self) -> str:
        descriptions = {
            ExtractionStrategy.CHAT_MEAN: "Chat template with mean of answer tokens",
            ExtractionStrategy.CHAT_FIRST: "Chat template with first answer token",
            ExtractionStrategy.CHAT_LAST: "Chat template with last token",
            ExtractionStrategy.CHAT_GEN_POINT: "Chat template with generation decision point",
            ExtractionStrategy.CHAT_MAX_NORM: "Chat template with max-norm answer token",
            ExtractionStrategy.CHAT_WEIGHTED: "Chat template with position-weighted mean",
            ExtractionStrategy.ROLE_PLAY: "Role-playing format with last token",
            ExtractionStrategy.MC_BALANCED: "Balanced multiple choice with last token",
        }
        return descriptions.get(self, "Unknown strategy")
    
    @classmethod
    def default(cls) -> "ExtractionStrategy":
        """Return the default strategy (chat_last is most commonly used)."""
        return cls.CHAT_LAST
    
    @classmethod
    def list_all(cls) -> list[str]:
        """List all strategy names."""
        return [s.value for s in cls]


# Random tokens for role_play strategy (deterministic based on prompt hash)
ROLE_PLAY_TOKENS = ["I", "Well", "The", "Sure", "Let", "That", "It", "This", "My", "To"]


def build_extraction_texts(
    strategy: ExtractionStrategy,
    prompt: str,
    response: str,
    tokenizer,
    other_response: Optional[str] = None,
    is_positive: bool = True,
) -> Tuple[str, str, Optional[str]]:
    """
    Build the full text for activation extraction based on strategy.
    
    Args:
        strategy: The extraction strategy to use
        prompt: The user prompt/question
        response: The response to extract activations for
        tokenizer: The tokenizer (needs apply_chat_template for chat strategies)
        other_response: For mc_balanced, the other response option
        is_positive: For mc_balanced, whether 'response' is the positive option
    
    Returns:
        Tuple of (full_text, answer_text, prompt_only_text)
        - full_text: Complete text to feed to model
        - answer_text: The answer portion (for strategies that need it)
        - prompt_only_text: Prompt without answer (for boundary detection)
    """
    
    if strategy in (ExtractionStrategy.CHAT_MEAN, ExtractionStrategy.CHAT_FIRST,
                    ExtractionStrategy.CHAT_LAST, ExtractionStrategy.CHAT_GEN_POINT,
                    ExtractionStrategy.CHAT_MAX_NORM, ExtractionStrategy.CHAT_WEIGHTED):
        # All chat_* strategies use the same prompt construction
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(
                f"Strategy {strategy.value} requires a tokenizer with apply_chat_template. "
                f"Base models without chat templates are not supported for chat_* strategies."
            )
        
        prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt},
             {"role": "assistant", "content": response}],
            tokenize=False,
            add_generation_prompt=False,
        )
        
        return full_text, response, prompt_only
    
    elif strategy == ExtractionStrategy.ROLE_PLAY:
        # "Behave like a person that would answer Q with A" format
        random_token = ROLE_PLAY_TOKENS[hash(prompt) % len(ROLE_PLAY_TOKENS)]
        instruction = f"Behave like a person that would answer {prompt} with {response}"
        
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(
                f"Strategy {strategy.value} requires a tokenizer with apply_chat_template. "
                f"Base models without chat templates are not supported for role_play strategy."
            )
        
        prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction},
             {"role": "assistant", "content": random_token}],
            tokenize=False,
            add_generation_prompt=False,
        )
        
        return full_text, random_token, prompt_only
    
    elif strategy == ExtractionStrategy.MC_BALANCED:
        # Multiple choice with balanced A/B assignment
        if other_response is None:
            raise ValueError("MC_BALANCED strategy requires other_response")
        
        # Deterministic "random" based on prompt - same for both pos and neg of a pair
        pos_goes_in_b = hash(prompt) % 2 == 0
        
        if is_positive:
            if pos_goes_in_b:
                option_a = other_response[:200]  # negative
                option_b = response[:200]        # positive
                answer = "B"
            else:
                option_a = response[:200]        # positive
                option_b = other_response[:200]  # negative
                answer = "A"
        else:
            if pos_goes_in_b:
                option_a = response[:200]        # negative
                option_b = other_response[:200]  # positive
                answer = "A"
            else:
                option_a = other_response[:200]  # positive
                option_b = response[:200]        # negative
                answer = "B"
        
        mc_prompt = f"Question: {prompt}\n\nWhich is correct?\nA. {option_a}\nB. {option_b}\nAnswer:"
        
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(
                f"Strategy {strategy.value} requires a tokenizer with apply_chat_template. "
                f"Base models without chat templates are not supported for mc_balanced strategy."
            )
        
        prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": mc_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": mc_prompt},
             {"role": "assistant", "content": answer}],
            tokenize=False,
            add_generation_prompt=False,
        )
        
        return full_text, answer, prompt_only
    
    else:
        raise ValueError(f"Unknown extraction strategy: {strategy}")


def extract_activation(
    strategy: ExtractionStrategy,
    hidden_states: torch.Tensor,
    answer_text: str,
    tokenizer,
    prompt_len: int,
) -> torch.Tensor:
    """
    Extract the activation vector based on strategy.
    
    Args:
        strategy: The extraction strategy
        hidden_states: Hidden states tensor of shape [seq_len, hidden_dim]
        answer_text: The answer text (for computing answer token count)
        tokenizer: The tokenizer
        prompt_len: Length of prompt in tokens (boundary)
    
    Returns:
        Activation vector of shape [hidden_dim]
    """
    seq_len = hidden_states.shape[0]
    
    # Compute answer token count
    answer_tokens = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    num_answer_tokens = len(answer_tokens)
    
    if strategy == ExtractionStrategy.CHAT_LAST:
        return hidden_states[-1]
    
    elif strategy == ExtractionStrategy.CHAT_FIRST:
        # First token of the answer
        first_answer_idx = max(0, seq_len - num_answer_tokens - 1)
        return hidden_states[first_answer_idx]
    
    elif strategy == ExtractionStrategy.CHAT_MEAN:
        # Mean of answer tokens
        if num_answer_tokens > 0 and seq_len > num_answer_tokens:
            answer_hidden = hidden_states[-num_answer_tokens-1:-1]
            return answer_hidden.mean(dim=0)
        return hidden_states[-1]
    
    elif strategy == ExtractionStrategy.CHAT_GEN_POINT:
        # Last token before answer starts (decision point)
        gen_point_idx = max(0, seq_len - num_answer_tokens - 2)
        return hidden_states[gen_point_idx]
    
    elif strategy == ExtractionStrategy.CHAT_MAX_NORM:
        # Token with max norm in answer region
        if num_answer_tokens > 0 and seq_len > num_answer_tokens:
            answer_hidden = hidden_states[-num_answer_tokens-1:-1]
            norms = torch.norm(answer_hidden, dim=1)
            max_idx = torch.argmax(norms)
            return answer_hidden[max_idx]
        return hidden_states[-1]
    
    elif strategy == ExtractionStrategy.CHAT_WEIGHTED:
        # Position-weighted mean (earlier tokens weighted more)
        if num_answer_tokens > 0 and seq_len > num_answer_tokens:
            answer_hidden = hidden_states[-num_answer_tokens-1:-1]
            weights = torch.exp(-torch.arange(answer_hidden.shape[0], dtype=torch.float32, device=answer_hidden.device) * 0.5)
            weights = weights / weights.sum()
            return (answer_hidden * weights.unsqueeze(1)).sum(dim=0)
        return hidden_states[-1]
    
    elif strategy in (ExtractionStrategy.ROLE_PLAY, ExtractionStrategy.MC_BALANCED):
        # Both use last token
        return hidden_states[-1]
    
    else:
        raise ValueError(f"Unknown extraction strategy: {strategy}")


def add_extraction_strategy_args(parser: argparse.ArgumentParser) -> None:
    """
    Add --extraction-strategy argument to an argument parser.
    
    Usage:
        parser = argparse.ArgumentParser()
        add_extraction_strategy_args(parser)
        args = parser.parse_args()
        strategy = ExtractionStrategy(args.extraction_strategy)
    """
    parser.add_argument(
        "--extraction-strategy",
        type=str,
        default=ExtractionStrategy.default().value,
        choices=ExtractionStrategy.list_all(),
        help=f"Extraction strategy for activations. Options: {', '.join(ExtractionStrategy.list_all())}. Default: {ExtractionStrategy.default().value}",
    )
