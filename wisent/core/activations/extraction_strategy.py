"""
Unified extraction strategies for activation collection.

These strategies combine prompt construction and token extraction into a single
unified approach, based on empirical testing of what actually works.

CHAT STRATEGIES (require chat template - for instruct models):
- chat_mean: Chat template prompt, mean of answer tokens
- chat_first: Chat template prompt, first answer token  
- chat_last: Chat template prompt, last token
- chat_max_norm: Chat template prompt, token with max norm in answer
- chat_weighted: Chat template prompt, position-weighted mean (earlier tokens weighted more)
- role_play: "Behave like person who answers Q with A" format, last token
- mc_balanced: Multiple choice with balanced A/B assignment, last token

BASE MODEL STRATEGIES (no chat template - for base models like gemma-2b, gemma-9b):
- completion_last: Direct Q+A completion, last token
- completion_mean: Direct Q+A completion, mean of answer tokens
- mc_completion: Multiple choice without chat template, A/B token
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
    """Chat template prompt with Q+A, extract EOT token (has seen full answer)."""
    
    CHAT_MAX_NORM = "chat_max_norm"
    """Chat template prompt with Q+A, extract token with max norm in answer region."""
    
    CHAT_WEIGHTED = "chat_weighted"
    """Chat template prompt with Q+A, position-weighted mean (earlier tokens weighted more)."""
    
    ROLE_PLAY = "role_play"
    """'Behave like person who answers Q with A' format, extract EOT token."""
    
    MC_BALANCED = "mc_balanced"
    """Multiple choice format with balanced A/B assignment, extract the A/B choice token."""
    
    # Base model strategies (no chat template required)
    COMPLETION_LAST = "completion_last"
    """Direct Q+A completion without chat template, extract last token. For base models."""
    
    COMPLETION_MEAN = "completion_mean"
    """Direct Q+A completion without chat template, extract mean of answer tokens. For base models."""
    
    MC_COMPLETION = "mc_completion"
    """Multiple choice without chat template, extract A/B token. For base models."""

    @property
    def description(self) -> str:
        descriptions = {
            ExtractionStrategy.CHAT_MEAN: "Chat template with mean of answer tokens",
            ExtractionStrategy.CHAT_FIRST: "Chat template with first answer token",
            ExtractionStrategy.CHAT_LAST: "Chat template with EOT token",
            ExtractionStrategy.CHAT_MAX_NORM: "Chat template with max-norm answer token",
            ExtractionStrategy.CHAT_WEIGHTED: "Chat template with position-weighted mean",
            ExtractionStrategy.ROLE_PLAY: "Role-playing format with EOT token",
            ExtractionStrategy.MC_BALANCED: "Balanced multiple choice with A/B token",
            ExtractionStrategy.COMPLETION_LAST: "Direct completion with last token (base models)",
            ExtractionStrategy.COMPLETION_MEAN: "Direct completion with mean of answer tokens (base models)",
            ExtractionStrategy.MC_COMPLETION: "Multiple choice completion with A/B token (base models)",
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
    
    @classmethod
    def for_tokenizer(cls, tokenizer, prefer_mc: bool = False) -> "ExtractionStrategy":
        """
        Select the appropriate strategy based on whether tokenizer supports chat template.
        
        Args:
            tokenizer: The tokenizer to check
            prefer_mc: If True, prefer multiple choice strategies (mc_balanced/mc_completion)
        
        Returns:
            Appropriate strategy for the tokenizer type
        """
        has_chat = (hasattr(tokenizer, "apply_chat_template") and callable(getattr(tokenizer, "apply_chat_template"))
                    and hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None)

        if has_chat:
            return cls.MC_BALANCED if prefer_mc else cls.CHAT_LAST
        else:
            return cls.MC_COMPLETION if prefer_mc else cls.COMPLETION_LAST
    
    @classmethod
    def is_base_model_strategy(cls, strategy: "ExtractionStrategy") -> bool:
        """Check if a strategy is designed for base models (no chat template)."""
        return strategy in (cls.COMPLETION_LAST, cls.COMPLETION_MEAN, cls.MC_COMPLETION)
    
    @classmethod 
    def get_equivalent_for_model_type(cls, strategy: "ExtractionStrategy", tokenizer) -> "ExtractionStrategy":
        """
        Get the equivalent strategy for the given tokenizer type.
        
        If strategy requires chat template but tokenizer doesn't have it,
        returns the base model equivalent. And vice versa.
        
        Args:
            strategy: The requested strategy
            tokenizer: The tokenizer to check
            
        Returns:
            The appropriate strategy for the tokenizer
        """
        has_chat = (hasattr(tokenizer, "apply_chat_template") and callable(getattr(tokenizer, "apply_chat_template"))
                    and hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None)
        is_base_strategy = cls.is_base_model_strategy(strategy)
        
        if has_chat and is_base_strategy:
            # Tokenizer has chat but strategy is for base model - upgrade to chat version
            mapping = {
                cls.COMPLETION_LAST: cls.CHAT_LAST,
                cls.COMPLETION_MEAN: cls.CHAT_MEAN,
                cls.MC_COMPLETION: cls.MC_BALANCED,
            }
            return mapping.get(strategy, strategy)
        
        elif not has_chat and not is_base_strategy:
            # Tokenizer is base model but strategy requires chat - downgrade to base version
            mapping = {
                cls.CHAT_LAST: cls.COMPLETION_LAST,
                cls.CHAT_FIRST: cls.COMPLETION_LAST,
                cls.CHAT_MEAN: cls.COMPLETION_MEAN,
                cls.CHAT_MAX_NORM: cls.COMPLETION_LAST,
                cls.CHAT_WEIGHTED: cls.COMPLETION_MEAN,
                cls.ROLE_PLAY: cls.COMPLETION_LAST,
                cls.MC_BALANCED: cls.MC_COMPLETION,
            }
            return mapping.get(strategy, cls.COMPLETION_LAST)
        
        return strategy


def tokenizer_has_chat_template(tokenizer) -> bool:
    """Check if tokenizer supports chat template."""
    has_method = hasattr(tokenizer, "apply_chat_template") and callable(getattr(tokenizer, "apply_chat_template"))
    has_template = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    return has_method and has_template


# Random tokens for role_play strategy (deterministic based on prompt hash)
ROLE_PLAY_TOKENS = ["I", "Well", "The", "Sure", "Let", "That", "It", "This", "My", "To"]


def build_extraction_texts(
    strategy: ExtractionStrategy,
    prompt: str,
    response: str,
    tokenizer,
    other_response: Optional[str] = None,
    is_positive: bool = True,
    auto_convert_strategy: bool = True,
) -> Tuple[str, str, Optional[str]]:
    """
    Build the full text for activation extraction based on strategy.
    
    Args:
        strategy: The extraction strategy to use
        prompt: The user prompt/question
        response: The response to extract activations for
        tokenizer: The tokenizer (needs apply_chat_template for chat strategies)
        other_response: For mc_balanced/mc_completion, the other response option
        is_positive: For mc_balanced/mc_completion, whether 'response' is the positive option
        auto_convert_strategy: If True, automatically convert strategy to match tokenizer type
    
    Returns:
        Tuple of (full_text, answer_text, prompt_only_text)
        - full_text: Complete text to feed to model
        - answer_text: The answer portion (for strategies that need it)
        - prompt_only_text: Prompt without answer (for boundary detection)
    """
    # Auto-convert strategy if needed
    if auto_convert_strategy:
        original_strategy = strategy
        strategy = ExtractionStrategy.get_equivalent_for_model_type(strategy, tokenizer)
        if strategy != original_strategy:
            import warnings
            warnings.warn(
                f"Strategy {original_strategy.value} not compatible with tokenizer, "
                f"using {strategy.value} instead.",
                UserWarning
            )
    
    if strategy in (ExtractionStrategy.CHAT_MEAN, ExtractionStrategy.CHAT_FIRST,
                    ExtractionStrategy.CHAT_LAST, ExtractionStrategy.CHAT_MAX_NORM,
                    ExtractionStrategy.CHAT_WEIGHTED):
        # All chat_* strategies use the same prompt construction
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(
                f"Strategy {strategy.value} requires a tokenizer with apply_chat_template. "
                f"Base models without chat templates are not supported for chat_* strategies. "
                f"Use completion_last, completion_mean, or mc_completion instead."
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
                f"Use completion_last or mc_completion for base models."
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
                f"Use mc_completion for base models."
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
    
    elif strategy in (ExtractionStrategy.COMPLETION_LAST, ExtractionStrategy.COMPLETION_MEAN):
        # Base model strategies - direct Q+A without chat template
        # Format: "Q: {prompt}\nA: {response}"
        prompt_only = f"Q: {prompt}\nA:"
        full_text = f"Q: {prompt}\nA: {response}"
        return full_text, response, prompt_only
    
    elif strategy == ExtractionStrategy.MC_COMPLETION:
        # Multiple choice for base models - no chat template
        if other_response is None:
            raise ValueError("MC_COMPLETION strategy requires other_response")
        
        # Deterministic "random" based on prompt - same for both pos and neg of a pair
        pos_goes_in_b = hash(prompt) % 2 == 0
        
        if is_positive:
            if pos_goes_in_b:
                option_a = other_response[:200]
                option_b = response[:200]
                answer = "B"
            else:
                option_a = response[:200]
                option_b = other_response[:200]
                answer = "A"
        else:
            if pos_goes_in_b:
                option_a = response[:200]
                option_b = other_response[:200]
                answer = "A"
            else:
                option_a = other_response[:200]
                option_b = response[:200]
                answer = "B"
        
        mc_prompt = f"Question: {prompt}\n\nWhich is correct?\nA. {option_a}\nB. {option_b}\nAnswer:"
        
        prompt_only = mc_prompt
        full_text = f"{mc_prompt} {answer}"
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
        # EOT token - has seen the entire answer, best performance
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
            weights = torch.exp(-torch.arange(answer_hidden.shape[0], dtype=answer_hidden.dtype, device=answer_hidden.device) * 0.5)
            weights = weights / weights.sum()
            return (answer_hidden * weights.unsqueeze(1)).sum(dim=0)
        return hidden_states[-1]
    
    elif strategy == ExtractionStrategy.ROLE_PLAY:
        # EOT token - slightly better than answer word (65% vs 64%)
        return hidden_states[-1]
    
    elif strategy == ExtractionStrategy.MC_BALANCED:
        # Answer token (A/B) - better than EOT (64% vs 56%)
        return hidden_states[-2]
    
    elif strategy == ExtractionStrategy.COMPLETION_LAST:
        # Last token for base model completion
        return hidden_states[-1]
    
    elif strategy == ExtractionStrategy.COMPLETION_MEAN:
        # Mean of answer tokens for base model completion
        if num_answer_tokens > 0 and seq_len > num_answer_tokens:
            answer_hidden = hidden_states[-num_answer_tokens:]
            return answer_hidden.mean(dim=0)
        return hidden_states[-1]
    
    elif strategy == ExtractionStrategy.MC_COMPLETION:
        # A/B token for base model MC (last token is the answer)
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


def get_strategy_for_model(tokenizer, prefer_mc: bool = False) -> ExtractionStrategy:
    """
    Get the best extraction strategy for a given tokenizer.
    
    Automatically detects if tokenizer has chat template and returns
    the appropriate strategy.
    
    Args:
        tokenizer: The tokenizer to check
        prefer_mc: If True, prefer multiple choice strategies
        
    Returns:
        ExtractionStrategy appropriate for the tokenizer
        
    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
        >>> strategy = get_strategy_for_model(tokenizer)
        >>> print(strategy)  # completion_last (base model)
        
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        >>> strategy = get_strategy_for_model(tokenizer)
        >>> print(strategy)  # chat_last (instruct model)
    """
    return ExtractionStrategy.for_tokenizer(tokenizer, prefer_mc=prefer_mc)
