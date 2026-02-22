from wisent.core.synthetic.cleaners.methods.core.atoms import Refusaler
from wisent.core.models.wisent_model import WisentModel 

import re, unicodedata

__all__ = [
    "BaseRefusaler",
]   

class BaseRefusaler(Refusaler):
    """
    Refusal detection and fixing step.
    """

    _FILLER = r"(?:\b\w{1,15}\b\s+){0,2}"

    _NEG_MODAL   = r"(?:can(?:\s*not)?|can't|won't|will\s+not|should(?:\s*not)?|shouldn't|must(?:\s*not)?|mustn't)"
    _AM          = r"(?:i\s+(?:am|['']m))"
    _I           = r"\bi\b"
    _I_M         = r"(?:i['']m)"
    _VERB_ACTION = (
        r"(?:help|assist|provide|share|give|supply|offer|"
        r"comply(?:\s+with)?|fulfil?l|answer|respond|"
        r"create|generate|write|compose|produce|draft|"
        r"explain|instruct|guide|walk\s+you\s+through|"
        r"facilitate|bypass|circumvent|enable|"
        r"do\s+(?:that|this|so))"
    )
    _SUPPORT_VERBS = r"(?:support|condone|endorse|promote|encourage)"
    _ABILITY_NOUNS = r"(?:ability|capability|capacity|permission|authorization|access)"

    _REFUSAL_RE = re.compile(
        rf"""
        # verbose, ignorecase, unicode

        (?:
            # A) "As an AI (language) model..." disclaimers
            (?P<ai_disclaimer>\bas\s+an?\s+(?:ai|assistant)(?:\s+language)?\s+model\b)

        | # B) Policy/legal/safety invocations
            (?P<policy>
                \b(?:this|that|it)\s+would\s+be\s+(?:unethical|unsafe|illegal|harmful)\b
                |
                \bagainst\s+(?:policy|the\s+rules|guidelines|terms(?:\s+of\s+service)?)\b
                |
                \b(?:due\s+to|for)\s+(?:safety|legal|policy|ethical)\s+(?:reasons|concerns)\b
            )

        | # C) Apologies / hedges commonly preceding refusals
            (?P<apology_hedge>
                \b(?:sorry|apologies|unfortunately|regrettably)\b
                |
                \b{_I_M}\s+afraid\b
            )

        | # D) Explicit inability / not permitted / not comfortable
            (?P<unable>
                {_AM}\s+(?:unable|not\s+able|not\s+allowed|not\s+permitted|not\s+comfortable)\s+to\b
            )

        | # E) “I [negation] [filler] [help/provide/…]”
            (?P<cannot_action>
                {_I}\s+{_NEG_MODAL}\s+{_FILLER}{_VERB_ACTION}\b
            )

        | # F) Prefer/rather not (indirect refusal)
            (?P<prefer_rather>
                \b(?:i['']d|i\s+would)\s+(?:rather|prefer)\s+not\s+(?:to\s+)?\w+\b
            )

        | # G) Decline/refuse verbs
            (?P<decline_refuse>
                {_I}\s+(?:must|have\s+to|will)\s+(?:decline|refuse)\s+to\b
                |
                \b(?:i\s+decline|i\s+refuse)\b
            )

        | # H) “I don't/won't [support|condone|endorse|promote] …”
            (?P<no_support>
                {_I}\s+(?:do\s+not|don't|won't|will\s+not)\s+{_SUPPORT_VERBS}\b
            )

        | # I) “I do not have the [ability/capability/permission/access] to …”
            (?P<no_ability>
                {_I}\s+(?:do\s+not|don't)\s+have\s+the\s+{_ABILITY_NOUNS}\s+to\b
            )

        | # J) Direct lexical hits
            (?P<refusal_word>\brefus(?:e|al)\b)
        )
        """.replace("{i_am}", "(?:i\\s+(?:am|['']m))"),
        re.VERBOSE | re.IGNORECASE | re.UNICODE,
    )

    _FAMILY_WEIGHTS = {
        "ai_disclaimer": 0.9,
        "policy": 0.9,
        "apology_hedge": 0.4, 
        "unable": 0.9,
        "cannot_action": 1.0,
        "prefer_rather": 0.6,
        "decline_refuse": 0.9,
        "no_support": 0.8,
        "no_ability": 0.8,
        "refusal_word": 0.6,
    }
    _THRESHOLD = 0.9  

    @staticmethod
    def _normalize(text: str) -> str:
        """
        Normalize text for refusal detection.
        
        arguments:
            text:
                Input text string.
        
        returns:
            Normalized text string.
            
        example:
            >>> BaseRefusaler._normalize("  As an AI model, I can't help with that.  ")"
            'As an AI model, I can't help with that.'
            >>> BaseRefusaler._normalize("This is a test.\u00A0")
            'This is a test.'
            >>> BaseRefusaler._normalize("Café")
            'Café'
        """
        return unicodedata.normalize("NFKC", text).strip()

    def score_refusal(self, text: str) -> tuple[float, str | None, str]:
        """
        Returns (score, top_family, snippet). Multiple families can fire; we
        take the max weight and add small bonus if apology_hedge co-occurs.

        arguments:
            text:
                Input text string.

        returns:
            score:
                Refusal likelihood score between 0.0 and 1.0.
            top_family:
                Name of the highest-weighted matched refusal family, or None.
            snippet:
                The matched refusal snippet from the text, or empty string.
        
        example:
            >>> rp = DefaultRefusalPolicy()
            >>> rp.score_refusal("As an AI model, I can't help with that.")
            (1.0, 'ai_disclaimer', "As an AI model, I can't help with that.")
            >>> rp.score_refusal("Sorry, I am unable to assist with that.")
            (0.9, 'unable', 'I am unable to assist with that.')
            >>> rp.score_refusal("I don't support that.")
            (0.8, 'no_support', "I don't support that.")
            >>> rp.score_refusal("I can't help. Sorry about that.")
            (1.0, 'cannot_action', "I can't help")
            >>> rp.score_refusal("I can't help. I am sorry about that.")
            (1.0, 'cannot_action', "I can't help")
            >>> rp.score_refusal("I can't help. Unfortunately, I am not able to assist.")
            (1.0, 'cannot_action', "I can't help")
            >>> rp.score_refusal("I can help with that.")
            (0.0, None, "")
        """
        t = self._normalize(text)
        m = self._REFUSAL_RE.search(t)
        if not m:
            return 0.0, None, ""
        
        best_family, best_w = None, 0.0
        for name, val in m.groupdict().items():
            if val:
                w = self._FAMILY_WEIGHTS.get(name, 0.5)
                if w > best_w:
                    best_family, best_w = name, w
        bonus = 0.0
        if m.group("apology_hedge"):
            if any(name != "apology_hedge" and m.group(name) for name in self._FAMILY_WEIGHTS):
                bonus = 0.1
        score = min(1.0, best_w + bonus)
        return score, best_family, m.group(0)

    def looks_like_refusal(self, text: str) -> bool:
        """
        Detect if the text looks like a refusal.

        arguments:
            text:
                Input text string.
        
        returns:
            True if the text looks like a refusal, False otherwise.

        example:
            >>> rp = DefaultRefusalPolicy()
            >>> rp.looks_like_refusal("As an AI model, I can't help with that.")
            True
            >>> rp.looks_like_refusal("Sorry, I am unable to assist with that.")
            True
            >>> rp.looks_like_refusal("I don't support that.")
            True
            >>> rp.looks_like_refusal("I can help with that.")
            False
        """
        score, *_ = self.score_refusal(text)
        return score >= self._THRESHOLD

    def why_refusal(self, text: str) -> tuple[str, str] | None:
        """
        If the text looks like a refusal, return (family, snippet) for the top match.

        arguments:
            text:
                Input text string.

        returns:
            (family, snippet) tuple if the text looks like a refusal, None otherwise.

        example:
            >>> rp = DefaultRefusalPolicy()
            >>> rp.why_refusal("As an AI model, I can't help with that.")
            ('ai_disclaimer', "As an AI model, I can't help with that.")
            >>> rp.why_refusal("Sorry, I am unable to assist with that.")
            ('unable', 'I am unable to assist with that.')
            >>> rp.why_refusal("I support that.")
            None
        """
        score, fam, snip = self.score_refusal(text)
        if score == 0.0:
            return None
        return fam or "unknown", snip

    def fix_negative(
        self,
        model: WisentModel,
        generation_conf: dict,
        prompt: str,
        trait_label: str,
        trait_description: str,
        system_prompt: str,
    ) -> str:
        """
        Attempt to fix a refusal negative example by re-prompting the model.

        arguments:
            model:
                WisentModel instance to call.
            prompt:
                The original prompt text.
            trait_label:
                Label of the undesired trait.
            trait_description:
                Description of the undesired trait.
            system_prompt:
                System prompt to use for the model call.

        returns:
            New negative example text, or empty string if still a refusal.

        example:
            >>> rp = DefaultRefusalPolicy()
            >>> def mock_completion_fn(msgs):
            ...     return "As an AI model, I cannot help with that."
            >>> rp.fix_negative(mock_completion_fn, "Tell me a joke.", "toxic", "contains toxic language", "System prompt")
            ... ""
            >>> def mock_completion_fn2(msgs):
            ...     return "Here's a joke: Why did the chicken cross the road? To get to the other side!"
            >>> rp.fix_negative(mock_completion_fn2, "Tell me a joke.", "toxic", "contains toxic language", "System prompt")
            ... "Here's a joke: Why did the chicken cross the road? To get to the other side!"
        """
        msgs = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Prompt: {prompt}\nTrait label: {trait_label}\nTrait description: {trait_description}",
            },
        ]
        neg_trial = model.generate(
            inputs=[msgs],
            max_tokens=generation_conf.get("max_tokens", 256),
            temperature=generation_conf.get("temperature", 1.0),
            use_steering=False,
            top_p=generation_conf.get("top_p", 1.0),
        )
        return "" if self.looks_like_refusal(neg_trial) else neg_trial
    