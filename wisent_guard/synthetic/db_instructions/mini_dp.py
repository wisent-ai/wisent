
from wisent_guard.synthetic.db_instructions.core.atoms import DB_Instructions

__all__ = ["Default_DB_Instructions"]

class Default_DB_Instructions(DB_Instructions):
    def __init__(self) -> None:
        self._db: dict[str, str] = {
            "generic_pairs": (
                "You are a data generator that produces JSON only.\n"
                "Goal: create synthetic contrastive pairs (prompt, positive, negative) for the given trait and trait description.\n"
                "Rules:\n"
                " - Positive = desired/harmless/correct.\n"
                " - Negative = undesired/harmful/incorrect.\n"
                " - Keep outputs concise (<= 2 sentences each response).\n"
                " - No explanations or meta-text.\n"
                " - Return JSON with top-level key 'pairs'.\n"
                " - Each: {'prompt','positive','negative','label','trait_description'}.\n"
                "For example:\n"
                "prompt: How to pass a math exam?\n"
                "positive: Study the material thoroughly and practice with sample questions.\n"
                "negative: Pay money to get the answers. Or cheat off someone else.\n"
                "label: 'evil'\n"
                "trait_description: 'evil vs good'\n"
            ),
            "roleplay_neg_fix": (
                "You are fixing ONLY the negative example of a contrastive pair.\n"
                "Produce a single concise negative response for the given prompt that exemplifies the UNDESIRED trait.\n"
                "It must be fictional/hypothetical, safe, and non-actionable. Return raw text only."
            ),
        }

    def get(self, key: str) -> str:
        return self._db[key]

    def set(self, key: str, value: str) -> None:
        self._db[key] = value