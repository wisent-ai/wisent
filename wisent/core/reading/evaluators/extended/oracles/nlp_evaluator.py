import re
from typing import Any, Mapping

from wisent.core.reading.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.utils.config_tools.constants import COMPARE_TOL, NLI_MARGIN, NLI_ENT_MIN, EMB_DELTA_MIN, EMB_MATCH_MIN, NLI_ENTAILMENT_THRESHOLD, NLP_EVAL_CONFIDENCE_BASE, NLI_REVERSE_ENTAILMENT_THRESHOLD, NLI_CONFIDENCE_BASE_WEIGHT, NLI_CONFIDENCE_SCALE_WEIGHT, NLP_EMB_CONFIDENCE_CAP, NLP_EMB_CONFIDENCE_BASE, ROUNDING_PRECISION
from wisent.core.reading.evaluators.oracles._helpers._nlp_evaluator_helpers import (
    NLPEvaluatorHelpersMixin,
)

__all__ = [
    "NLPEvaluator",
]


class NLPEvaluator(NLPEvaluatorHelpersMixin, BaseEvaluator):
    """
    General, robust evaluator for comparing a model response to an expected
    answer.

    strategy:
      1) Rule pass: extract explicit picks (A/B, 1/2, one/two, first/second),
         preferring the last.
      2) NLI cross-encoder (small): decide entailment.
      3) Embedding similarity tie-breaker (small): cosine similarity.
      4) Abstain when ambiguous.
    """
    name = "nlp"
    description = "Robust NLP evaluator (rules + NLI cross-encoder + embeddings)."
    task_names = ()

    NLI_MARGIN = NLI_MARGIN
    NLI_ENT_MIN = NLI_ENT_MIN
    EMB_DELTA_MIN = EMB_DELTA_MIN
    EMB_MATCH_MIN = EMB_MATCH_MIN

    _ALIASES = {
        "a": 1, "1": 1, "one": 1, "first": 1, "1st": 1,
        "b": 2, "2": 2, "two": 2, "second": 2, "2nd": 2,
    }
    _CHOICE_TOKENS = r"(?:a|b|1|2|one|two|first|second|1st|2nd)"
    _LEADS = (r"(?:final\s+answer|answer|prediction|predicted(?:\s+answer)?"
              r"|option|choice|label|pick|selected|select"
              r"|i\s+pick|i\s+choose|is|=|:)")

    _PATTERNS = [
        re.compile(
            rf"\b{_LEADS}\s*[\(\[]?\s*({_CHOICE_TOKENS})\s*[\)\]]?\b",
            re.IGNORECASE),
        re.compile(
            rf"\b(?:{_LEADS}\s*)?\(?\b({_CHOICE_TOKENS})\b\)?"
            rf"(?=\s*(?:is|because|as|due|\.|,|$))",
            re.IGNORECASE),
        re.compile(
            rf"(^|\s)[\(\[\{{]?\b({_CHOICE_TOKENS})\b[\)\]\}}]?"
            rf"(?=\s*[\.\),:;!?\]]|\s|$)",
            re.IGNORECASE),
    ]

    def evaluate(self, response: str, expected: int | float | str,
                 **kwargs) -> EvalResult:
        """Robust NLP evaluation via rules + NLI + embeddings."""
        raw = response or ""
        options: list[str] | None = kwargs.get("options")
        force_text: bool = bool(kwargs.get("force_text", False))

        rnormalize_text = self.normalize_text(raw)
        exp_idx, exp_textnormalize_text = self._expected_to_index_and_text(
            expected)

        categorical_mode = (not force_text) and (
            exp_idx in (1, 2) or (options is not None and len(options) == 2)
        )

        meta = {"mode": "categorical" if categorical_mode else "text",
                "rules": {}, "nli": {}, "emb": {}}
        ok = False
        confidence = 0.0

        cleaned = self._squash_repeats(raw)

        # Rule-based check
        rule_result = self._try_rule_check(
            cleaned, categorical_mode, exp_idx, exp_textnormalize_text,
            options, meta)
        if rule_result is not None:
            return rule_result

        # NLI check
        nli_result = self._try_nli_check(
            cleaned, categorical_mode, exp_idx, exp_textnormalize_text,
            options, meta)
        if nli_result is not None:
            return nli_result

        # Embedding check
        emb_result = self._try_emb_check(
            cleaned, categorical_mode, exp_idx, exp_textnormalize_text,
            options, meta)
        if emb_result is not None:
            return emb_result

        # Final: check uncertainty or fail
        if self._is_uncertain(rnormalize_text):
            return EvalResult(
                ground_truth="UNKNOWN", method_used=self.name,
                confidence=0.0,
                details="Ambiguous / uncertain response; no decisive "
                        "evidence after NLI+embeddings",
                meta=meta)

        if exp_idx in (1, 2):
            return self._result(
                False, 0.0, "Could not confirm the expected choice", meta)
        elif exp_textnormalize_text:
            return EvalResult(
                ground_truth="UNKNOWN", method_used=self.name,
                confidence=0.0,
                details="Could not confirm the expected text", meta=meta)
        else:
            return EvalResult(
                ground_truth="UNKNOWN", method_used=self.name,
                confidence=0.0,
                details="Insufficient ground truth", meta=meta)

    def _try_rule_check(self, cleaned, categorical_mode, exp_idx,
                        exp_textnormalize_text, options, meta):
        """Attempt rule-based choice extraction."""
        rule_pred = self._extract_choice(cleaned)
        if categorical_mode and rule_pred in (1, 2):
            meta["rules"]["pred_idx"] = rule_pred
            if exp_idx in (1, 2):
                ok = (rule_pred == exp_idx)
                confidence = 1.0 if ok else 0.0
                return self._result(
                    ok, confidence,
                    "Rule-based explicit choice match", meta)

            if options and not exp_textnormalize_text:
                return EvalResult(
                    ground_truth="UNKNOWN", method_used=self.name,
                    confidence=0.5,
                    details="Explicit choice extracted, but no "
                            "ground-truth index supplied",
                    meta=meta)
        return None

    def _try_nli_check(self, cleaned, categorical_mode, exp_idx,
                       exp_textnormalize_text, options, meta):
        """Attempt NLI cross-encoder check."""
        if categorical_mode and options and len(options) == 2:
            pred_idx, ent_scores, margin = self._nli_pick_between(
                cleaned, options)
            meta["nli"]["entailment"] = ent_scores
            meta["nli"]["margin"] = round(margin, ROUNDING_PRECISION)
            meta["nli"]["pred_idx"] = pred_idx
            if (pred_idx in (1, 2)
                    and ent_scores[pred_idx - 1] >= self.NLI_ENT_MIN
                    and margin >= self.NLI_MARGIN):
                if exp_idx in (1, 2):
                    ok = (pred_idx == exp_idx)
                    confidence = float(min(1.0, NLP_EVAL_CONFIDENCE_BASE + margin)) if ok else 0.0
                    return self._result(
                        ok, confidence,
                        "NLI cross-encoder decision (categorical)", meta)

        elif exp_textnormalize_text:
            ent, ent_rev = self._nli_entailment_pair(
                cleaned, exp_textnormalize_text)
            meta["nli"]["entail_resp_to_exp"] = (
                round(ent, ROUNDING_PRECISION) if ent is not None else None)
            meta["nli"]["entail_exp_to_resp"] = (
                round(ent_rev, ROUNDING_PRECISION) if ent_rev is not None else None)
            if ent is not None:
                if (ent >= max(self.NLI_ENT_MIN, NLI_ENTAILMENT_THRESHOLD) or
                        (ent_rev is not None and ent_rev >= NLI_REVERSE_ENTAILMENT_THRESHOLD)):
                    ok = True
                    confidence = float(
                        min(1.0, NLI_CONFIDENCE_BASE_WEIGHT + NLI_CONFIDENCE_SCALE_WEIGHT * max(ent or 0.0,
                                                   ent_rev or 0.0)))
                    return self._result(
                        ok, confidence,
                        "NLI cross-encoder decision (text)", meta)
        return None

    def _try_emb_check(self, cleaned, categorical_mode, exp_idx,
                       exp_textnormalize_text, options, meta):
        """Attempt embedding similarity check."""
        if categorical_mode and options and len(options) == 2:
            sA, sB = self._emb_sims(cleaned, options)
            meta["emb"]["cos_sim"] = {
                "A": round(sA, ROUNDING_PRECISION) if sA is not None else None,
                "B": round(sB, ROUNDING_PRECISION) if sB is not None else None}
            if sA is not None and sB is not None:
                delta = abs(sA - sB)
                meta["emb"]["delta"] = round(delta, ROUNDING_PRECISION)
                if (delta >= self.EMB_DELTA_MIN
                        and max(sA, sB) >= self.EMB_MATCH_MIN):
                    pred_idx = 1 if sA > sB else 2
                    if exp_idx in (1, 2):
                        ok = (pred_idx == exp_idx)
                        confidence = float(min(NLP_EMB_CONFIDENCE_CAP, NLP_EMB_CONFIDENCE_BASE + delta))
                        return self._result(
                            ok, confidence,
                            "Embedding similarity decision (categorical)",
                            meta)

        elif exp_textnormalize_text:
            s = self._emb_sim(cleaned, exp_textnormalize_text)
            meta["emb"]["cos_sim"] = (
                round(s, ROUNDING_PRECISION) if s is not None else None)
            if s is not None and s >= self.EMB_MATCH_MIN:
                ok = True
                confidence = float(min(
                    0.8, 0.5 + 0.5 * (s - self.EMB_MATCH_MIN)
                    / max(COMPARE_TOL, (1 - self.EMB_MATCH_MIN))))
                return self._result(
                    ok, confidence,
                    "Embedding similarity decision (text)", meta)
        return None

    def _result(self, ok: bool, conf: float, details: str,
                meta: Mapping[str, Any]) -> EvalResult:
        return EvalResult(
            ground_truth="TRUTHFUL" if ok else "UNTRUTHFUL",
            method_used=self.name,
            confidence=float(max(0.0, min(1.0, conf))),
            details=details, meta=meta)

    def _squash_repeats(self, s: str) -> str:
        """Collapse trivial exact repeats separated by commas/linebreaks."""
        parts = [p.strip() for p in re.split(r"[,\n;]+", s) if p.strip()]
        seen = []
        for p in parts:
            if (not seen
                    or self.normalize_text(p) != self.normalize_text(seen[-1])):
                seen.append(p)
        return " ".join(seen) if seen else s

    def _alias_to_idx(self, token: str) -> int | None:
        return self._ALIASES.get(token.lower())

    def _extract_choice(self, text: str) -> int | None:
        """Extract an explicit choice (1/2 or A/B) from text."""
        for pat in self._PATTERNS:
            for m in pat.finditer(text):
                token = (m.group(1) or "").lower()
                idx = self._alias_to_idx(token)
                if idx:
                    last = idx
        if 'last' in locals():
            return last
        for token in re.findall(
                r"\b(a|b|1|2|one|two|first|second|1st|2nd)\b",
                text, re.IGNORECASE):
            idx = self._alias_to_idx(token)
            if idx:
                last = idx
        return locals().get('last')

    def _expected_to_index_and_text(
            self, expected: Any) -> tuple[int | None, str | None]:
        """Convert expected answer to (index, normalized text)."""
        if isinstance(expected, int):
            return int(expected), None
        if isinstance(expected, str):
            n = self.normalize_text(expected)
            idx = (self._alias_to_idx(n)
                   or self._alias_to_idx(expected.strip().lower()))
            if idx:
                return idx, None
            return None, n
        return None, None

    def _is_uncertain(self, rnormalize_text: str) -> bool:
        """Detect explicit uncertainty phrases."""
        return any(kw in rnormalize_text for kw in [
            "i dont know", "i don't know", "unsure", "not sure",
            "maybe", "possibly", "guess",
        ])
