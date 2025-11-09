
from wisent.core.synthetic.db_instructions.core.atoms import DB_Instructions

__all__ = ["Default_DB_Instructions"]

class Default_DB_Instructions(DB_Instructions):
    def __init__(self) -> None:
        self._db: dict[str, str] = {
            "generic_pairs": (
                "You are a data generator that creates contrastive pairs.\n"
                "Goal: create synthetic contrastive pairs (prompt, positive, negative) for the given trait and trait description.\n"
                "Rules:\n"
                " - Positive = desired/harmless/correct.\n"
                " - Negative = undesired/harmful/incorrect.\n"
                " - Keep outputs concise (<= 2 sentences each response).\n"
                " - No explanations or meta-text.\n"
                "\n"
                "Format each pair like this:\n"
                "---PAIR---\n"
                "PROMPT: <the question or scenario>\n"
                "POSITIVE: <the good/desired response>\n"
                "NEGATIVE: <the bad/undesired response>\n"
                "---END---\n"
                "\n"
                "Example:\n"
                "---PAIR---\n"
                "PROMPT: How to pass a math exam?\n"
                "POSITIVE: Study the material thoroughly and practice with sample questions.\n"
                "NEGATIVE: Pay money to get the answers. Or cheat off someone else.\n"
                "---END---\n"
            ),
            "roleplay_neg_fix": (
                "You are fixing ONLY the negative example of a contrastive pair.\n"
                "Produce a single concise negative response for the given prompt that exemplifies the UNDESIRED trait.\n"
                "It must be fictional/hypothetical, safe, and non-actionable. Return raw text only."
            ),
            "nonsense_random_chars": (
                "You are a data generator that produces JSON only.\n"
                "Goal: create contrastive pairs where the negative response is complete gibberish/nonsense.\n"
                "Rules:\n"
                " - Positive = meaningful, coherent, helpful response.\n"
                " - Negative = random characters and strings with no meaning (e.g., 'ahsdhashdahsdha', 'jkfdjkfdjkfd', 'zxcvbnmasdfg').\n"
                " - The negative should look like someone randomly hit keyboard keys.\n"
                " - Keep outputs concise (<= 2 sentences for positive, random string for negative).\n"
                " - No explanations or meta-text.\n"
                " - Return JSON with top-level key 'pairs'.\n"
                " - Each: {'prompt','positive','negative','label','trait_description'}.\n"
                "For example:\n"
                "prompt: What is the capital of France?\n"
                "positive: The capital of France is Paris.\n"
                "negative: asdjkhasjkdhaksjdh aksjdhaksjdh aksdjhaksjdhaksj\n"
                "label: 'nonsense'\n"
                "trait_description: 'coherent vs random gibberish'\n"
            ),
            "nonsense_repetitive": (
                "You are a data generator that produces JSON only.\n"
                "Goal: create contrastive pairs where the negative response is highly repetitive nonsense.\n"
                "Rules:\n"
                " - Positive = meaningful, coherent, helpful response.\n"
                " - Negative = extremely repetitive text with the same words/phrases over and over (e.g., 'the the the the the', 'yes yes yes yes yes yes').\n"
                " - The negative should show pathological repetition.\n"
                " - Keep outputs concise (<= 2 sentences for positive, repetitive string for negative).\n"
                " - No explanations or meta-text.\n"
                " - Return JSON with top-level key 'pairs'.\n"
                " - Each: {'prompt','positive','negative','label','trait_description'}.\n"
                "For example:\n"
                "prompt: What is photosynthesis?\n"
                "positive: Photosynthesis is the process by which plants convert sunlight into energy.\n"
                "negative: plants plants plants plants plants plants plants plants plants plants plants plants\n"
                "label: 'nonsense'\n"
                "trait_description: 'coherent vs repetitive nonsense'\n"
            ),
            "nonsense_word_salad": (
                "You are a data generator that produces JSON only.\n"
                "Goal: create contrastive pairs where the negative response is word salad (real words but no coherent meaning).\n"
                "Rules:\n"
                " - Positive = meaningful, coherent, helpful response.\n"
                " - Negative = real English words strung together randomly with no logical connection (e.g., 'purple elephant calculator yesterday moon basket thinking').\n"
                " - The negative should use real words but make absolutely no sense.\n"
                " - Keep outputs concise (<= 2 sentences for positive, word salad for negative).\n"
                " - No explanations or meta-text.\n"
                " - Return JSON with top-level key 'pairs'.\n"
                " - Each: {'prompt','positive','negative','label','trait_description'}.\n"
                "For example:\n"
                "prompt: How do you bake a cake?\n"
                "positive: Mix flour, eggs, sugar, and butter, then bake in the oven at 350°F for 30 minutes.\n"
                "negative: telephone purple yesterday elephant calculator happiness mountain running quickly tomorrow\n"
                "label: 'nonsense'\n"
                "trait_description: 'coherent vs word salad'\n"
            ),
            "nonsense_mixed": (
                "You are a data generator that produces JSON only.\n"
                "Goal: create contrastive pairs where the negative response is a mix of different types of nonsense.\n"
                "Rules:\n"
                " - Positive = meaningful, coherent, helpful response.\n"
                " - Negative = combination of random characters, repetition, and word salad.\n"
                " - Vary the nonsense type within each negative response.\n"
                " - Keep outputs concise (<= 2 sentences for positive, mixed nonsense for negative).\n"
                " - No explanations or meta-text.\n"
                " - Return JSON with top-level key 'pairs'.\n"
                " - Each: {'prompt','positive','negative','label','trait_description'}.\n"
                "For example:\n"
                "prompt: What is machine learning?\n"
                "positive: Machine learning is a type of artificial intelligence that allows computers to learn from data.\n"
                "negative: jkfdjkfd learning learning learning asdjkh purple calculator jkfdjkfd yesterday yesterday\n"
                "label: 'nonsense'\n"
                "trait_description: 'coherent vs mixed nonsense'\n"
            ),
        }

    def get(self, key: str) -> str:
        return self._db[key]

    def set(self, key: str, value: str) -> None:
        self._db[key] = value