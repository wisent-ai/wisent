"""Prompt construction strategies for activation collection."""

from enum import Enum


class PromptConstructionStrategy(Enum):
    """
    Strategies for constructing prompts from question-answer pairs.

    These strategies determine how the prompt and response are formatted
    before being passed to the model for activation extraction.
    """

    MULTIPLE_CHOICE = "multiple_choice"
    """
    Format: Which is better: Q A. bad B. good → "A"/"B" (choice format)
    Example: "Which is better: What is 2+2? A. 5 B. 4"
    Response: "A" or "B"
    """

    ROLE_PLAYING = "role_playing"
    """
    Format: Behave like person who would answer Q with good_resp → "I" (role assumption)
    Example: "Behave like a person who would answer 'What is 2+2?' with '4'"
    Response: "I"
    """

    DIRECT_COMPLETION = "direct_completion"
    """
    Format: Q → good_resp/bad_resp (direct answer)
    Example: "What is 2+2?"
    Response: "4" or "5"
    """

    INSTRUCTION_FOLLOWING = "instruction_following"
    """
    Format: [INST] Q [/INST] → good_resp/bad_resp (instruction format)
    Example: "[INST] What is 2+2? [/INST]"
    Response: "4" or "5"
    """

    CHAT_TEMPLATE = "chat_template"
    """
    Format: Uses the model's built-in chat template
    Example: <|start_header_id|>user<|end_header_id|>What is 2+2?<|eot_id|>
    Response: Model's chat-formatted response
    """
