"""
Math parsing and evaluation scripts.

Re-exports from extracted submodules for backward compatibility.
"""
import re
from typing import Union, Optional, Any
import regex
from latex2sympy2_extended import latex2sympy
from wisent.core.evaluators.benchmark_specific.math_parsing.core import EvaluatorMath
import multiprocessing
from math import isclose
import time
from word2number import w2n

from wisent.core.evaluators.benchmark_specific.math_parsing._scripts_constants import (
    MULTILINGUAL_ANSWER_REGEXES,
    MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,
    STRIP_EXCEPTIONS,
    unit_texts,
)
from wisent.core.evaluators.benchmark_specific.math_parsing._scripts_parsing import (
    numeric_equal,
    parse_digits,
    is_digit,
    normalize_extracted_answer,
    convert_word_number,
    _fix_sqrt,
    _fix_fracs,
    _fix_a_slash_b,
    strip_string,
    str_to_pmatrix,
    choice_answer_clean,
)
from wisent.core.evaluators.benchmark_specific.math_parsing._scripts_extraction import (
    extract_answer,
    symbolic_equal,
    symbolic_equal_process,
    call_with_timeout,
)
from wisent.core.evaluators.benchmark_specific.math_parsing._scripts_equality import (
    math_equal,
)
from wisent.core.evaluators.benchmark_specific.math_parsing._scripts_adapted import (
    AdaptedEvaluatorMath,
    multi_math_equal,
)

__all__ = [
    "MULTILINGUAL_ANSWER_REGEXES",
    "MULTILINGUAL_ANSWER_PATTERN_TEMPLATE",
    "STRIP_EXCEPTIONS",
    "unit_texts",
    "numeric_equal",
    "parse_digits",
    "is_digit",
    "normalize_extracted_answer",
    "convert_word_number",
    "strip_string",
    "str_to_pmatrix",
    "choice_answer_clean",
    "extract_answer",
    "symbolic_equal",
    "symbolic_equal_process",
    "call_with_timeout",
    "math_equal",
    "AdaptedEvaluatorMath",
    "multi_math_equal",
]
