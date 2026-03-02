"""Adapted math evaluator and multi-strategy equality."""
import re
from typing import Any, Optional, List
from sympy import simplify, N
from wisent.core.reading.evaluators.benchmark_specific.math_parsing.core import EvaluatorMath
from wisent.core.reading.evaluators.benchmark_specific.math_parsing._scripts_constants import (
    MULTILINGUAL_ANSWER_REGEXES, MULTILINGUAL_ANSWER_PATTERN_TEMPLATE, STRIP_EXCEPTIONS,
)
from wisent.core.reading.evaluators.benchmark_specific.math_parsing._scripts_parsing import (
    numeric_equal, normalize_extracted_answer, strip_string,
)
from wisent.core.reading.evaluators.benchmark_specific.math_parsing._scripts_extraction import (
    extract_answer,
)
from wisent.core.reading.evaluators.benchmark_specific.math_parsing._scripts_equality import (
    math_equal,
)

class AdaptedEvaluatorMath(EvaluatorMath):
    def is_num_eq(
        self, ref_num: Optional[float], pred_num: Optional[float]
    ) -> Optional[bool]:
        """Compare two numbers with specified feautures:
        - relative tolerance
        - flexible percentage surface forms
        """
        if ref_num is None or pred_num is None:
            return None

        if numeric_equal(ref_num, pred_num):
            return True

        if self.include_percentage and self.could_be_percent(pred_num):
            percent_ref_nums: List[float] = [
                num
                for num in [ref_num / 100, ref_num * 100]
                if self.could_be_percent(num)
            ]
            for item in percent_ref_nums:
                # "For the values to be considered close, the difference between them must be smaller than at least one of the tolerances."
                # if isclose(
                #     item, pred_num, rel_tol=self.percent_rel_tol, abs_tol=self.abs_tol
                # ):
                if numeric_equal(item, pred_num):
                    return True
        return None

    def is_sym_eq(self, a: Any, b: Any) -> Optional[bool]:
        """Compare two objects symbolically."""
        if a is None or b is None:
            return None

        try:
            if a == b:
                return True
        except Exception:
            pass

        try:
            diff = simplify(a - b)
            # For non-symmetric operations like subtraction between sets
            diff_rev = simplify(b - a)

            if hasattr(diff, "__iter__") and hasattr(
                diff_rev, "__iter__"
            ):  # If diff is iterable (e.g. Matrix)
                if diff == diff_rev and all(element == 0 for element in diff) and all(
                    element == 0 for element in diff_rev
                ):
                    return True
            else:
                if (
                    not diff and not diff_rev
                ):  # use `not` for non-zero values like `sympy.EmptySet`
                    return True
        except Exception:
            pass

        try:
            v_a, v_b = (N(eval(str(v))) for v in [a, b])
            num_eq = self.is_num_eq(v_a, v_b)
            if num_eq:
                return True
        except Exception:
            pass

        return None   




def multi_math_equal(answer, response, choice=False):
    if choice:
        final_result = None
        for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
            regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
            match = re.search(regex, response)
            if match:
                match_result = normalize_extracted_answer(match.group(1))
                final_result = match_result
            if final_result is not None:
                break
        
        acc = 1.0 if final_result == answer else 0.0
        failed = 1.0 if final_result is None else 0.0
        return bool(acc), final_result, failed
    else:
        prediction1 = extract_answer(response, '')
        prediction1 = strip_string(prediction1, skip_unit='' in STRIP_EXCEPTIONS)
        result1 = math_equal(answer, prediction1)

        prediction2 = None
        try:
            evaluator = AdaptedEvaluatorMath(ans_extract_mode="boxed")
            prediction2 = evaluator.extract_ans(response)
            result2 = evaluator.eq(answer, prediction2)
        except:
            result2 = 0.0
        
        prediction = ((prediction1, result1), (prediction2, result2))
        # return bool(result1) or bool(result2), prediction, "boxed" not in response
        return bool(result1), prediction, "boxed" not in response
