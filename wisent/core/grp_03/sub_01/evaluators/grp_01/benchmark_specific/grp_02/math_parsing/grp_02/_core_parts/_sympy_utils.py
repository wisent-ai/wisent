"""Sympy/LaTeX utility functions and constants for math parsing."""
import os
import re as regex
from typing import Any, Callable, List, Optional
from sympy import (
    Complement, Expr, FiniteSet, Intersection, Interval, Union, simplify,
)
from sympy.parsing.latex import parse_latex
from sympy.parsing.latex.errors import LaTeXParsingError
from wisent.core.errors import InvalidDataFormatError
from typing import Dict as T_Dict
from typing import Union as T_Union

DEF_TIMEOUT: int = 5
DEF_N_PROC: int = os.cpu_count() // 2

STRIP_STRS: List[str] = [
    ":", "/", ",", "#", "?", "$", '"', "'",
    "\u043a", "\u0438", "\\(", "\\)", "\\[", "\\]",
]
NO_TRAILING_STRS: List[str] = ["(", "[", "{", "\\", "."] + STRIP_STRS
NO_PRECEDING_PUNCS: List[str] = [
    "!", ")", "]", "}", "\\\\", "boxed"
] + STRIP_STRS


def run_with_timeout(
    func: Callable[..., Any], kwargs: T_Dict[str, Any], timeout: int
) -> Any:
    """Run function (timeout parameter ignored)."""
    return func(**kwargs)


def latex2sympy_fix(s: str) -> Expr:
    sp_symbol: Expr = parse_latex(s)
    if "," in s:
        first_term = None
        try:
            first_term = parse_latex(s.split(",")[0])
        except Exception:
            pass
        if sp_symbol == first_term:
            raise LaTeXParsingError(f"{s} != {first_term}")
    return sp_symbol


def latex2sympy_interval(
    s: str,
) -> T_Union[FiniteSet, Union, Intersection, Complement, Interval]:
    """Parse LaTeX expression like (-\\infty,0] as SymPy Interval object."""
    s = s.replace(" ", "")
    if "\\cup" in s:
        exps = s.split("\\cup")
        intervals = [latex2sympy_interval(exp) for exp in exps]
        return Union(*intervals)
    if "\\cap" in s:
        exps = s.split("\\cap")
        intervals = [latex2sympy_interval(exp) for exp in exps]
        return Intersection(*intervals)
    if s.startswith("\\{") and s.endswith("\\}"):
        return FiniteSet(simplify(latex2sympy_fix(s[2:-2])))
    elif s.startswith("{") and s.endswith("}"):
        return FiniteSet(simplify(latex2sympy_fix(s[1:-1])))
    if s.startswith("("):
        left_open = True
        s = s[1:]
    elif s.startswith("\\("):
        left_open = True
        s = s[2:]
    elif s.startswith("["):
        left_open = False
        s = s[1:]
    elif s.startswith("\\["):
        left_open = False
        s = s[2:]
    else:
        raise InvalidDataFormatError(reason=f"Invalid interval start: {s}")
    if s.endswith(")"):
        right_open = True
        s = s[:-1]
    elif s.endswith("\\)"):
        right_open = True
        s = s[:-2]
    elif s.endswith("]"):
        right_open = False
        s = s[:-1]
    elif s.endswith("\\]"):
        right_open = False
        s = s[:-2]
    else:
        raise InvalidDataFormatError(reason=f"Invalid interval end: {s}")
    left: Expr
    right: Expr
    left, right = (simplify(latex2sympy_fix(side)) for side in s.split(","))
    if left.is_comparable and right.is_comparable and left >= right:
        raise InvalidDataFormatError(
            reason=f"Invalid interval bounds: {left}, {right}"
        )
    interval = Interval(left, right, left_open, right_open)
    return interval


PAREN_MAP: T_Dict[str, str] = {
    r"\(": r"\)", r"\[": r"\]", r"\{": r"\}",
    "(": ")", "[": "]", "{": "}",
}

DATETIME_FMTS: List[str] = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
    "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S", "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M", "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M", "%Y/%m/%d %H:%M",
    "%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p",
]

BASIC_FN_NAMES: List[str] = (
    "sin|cos|tan|cot|sec|csc|sinh|cosh|tanh|coth|sech|csch|log|ln|exp"
).split("|")

UNITS: List[str] = [
    "hour", "minute", "min", "sec", "second", "day", "week", "month",
    "year", "meter", "mile", "kg", "mg", "g", "t", "ton", "nm", "pm",
    "um", "\u03bcm", "m", "cm", "mm", "dm", "km", "kilometer", "inch",
    "feet", "piece", "bit", "hz", "Hz", "m/s", "km/s", "m/(min^2)",
    "billion", "eV", "V", "C", "s", "degree",
    r"a\.?m\.?", r"(?<!\\)p\.?m\.?",
]

DEF_REL_TOL = 1e-9
DEF_ABS_TOL = 1e-8
DEF_PERCENT_REL_TOL = 1e-3


def has_non_ascii(s: str) -> bool:
    for char in s:
        if ord(char) > 127:
            return True
    return False


def is_querying4set(query: str) -> bool:
    return "ind the" in query or ("all" in query and "separate" in query)


NDAYS_PER_WEEK = 7
WEEKDAY_ABBRS: List[str] = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
WEEKDAY_FULLS: List[str] = [
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
]


def norm_str2weekday(s: str) -> Optional[str]:
    """Converts a string representation of a weekday to its normalized form."""
    s = str(s).lower().strip()
    if " " in s:
        return None
    for i_day in range(NDAYS_PER_WEEK):
        if s.startswith(WEEKDAY_ABBRS[i_day]):
            return WEEKDAY_FULLS[i_day].capitalize()
    return None


def parse(
    parser: Callable, s_to_parse: str, parse_errs: List[Exception]
) -> Optional[Any]:
    try:
        return parser(s_to_parse)
    except Exception as e:
        parse_errs.append(e)
    return None


def norm_deg(s: str) -> str:
    """Normalize expressions including degrees."""
    s = s.replace("rad", "")
    s = regex.sub(r"^(\d+) ?\^?\\?circ$", r"\1", s)
    s = regex.sub(r"(\d+) ?\^?\\?circ", r"{\1*\\frac{\\pi}{180}}", s)
    return s


def is_set(s: str) -> bool:
    return (
        regex.search(r"[^a-z]or(x|[^a-z])", s) is not None
        or (s.startswith("{") and s.endswith("}"))
        or (s.startswith("\\{") and s.endswith("\\}"))
    )


def fix_sqrt(s: str) -> str:
    """Fixes the formatting of square root expressions."""
    _s = regex.sub(r"\\?sqrt[\(\{\[](\w+)[\)\}\]]", r"\\sqrt{\1}", s)
    _s = regex.sub(r"\\?sqrt\s*(\d+)", r"\\sqrt{\1}", _s)
    return _s


def fix_fracs(s: str) -> str:
    """Fixes the formatting of fractions in a given string."""
    substrs = s.split("\\frac")
    _s = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            _s += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                _s += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return s
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        _s += "{" + a + "}{" + b + "}" + substr[2:]
                    else:
                        _s += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        _s += "{" + a + "}" + b + substr[2:]
                    else:
                        _s += "{" + a + "}" + b
    return _s


def fix_a_slash_b(s: str) -> str:
    """Fixes the formatting of fractions using regex."""
    fraction_pattern = r"(\b\d+|sqrt\(.*?\))\/(\d+|sqrt\(.*?\)\b)"
    result = regex.sub(
        fraction_pattern,
        lambda m: f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}",
        s,
    )
    return result


# --- LaTeX constants (merged from _latex_constants.py) ---

STR2NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}


def rm_latex_env(s: str, env: str) -> str:
    """Remove LaTeX environment from a string."""
    s = s.replace(f"\\begin{{{env}}}", "")
    s = s.replace(f"\\end{{{env}}}", "")
    return s


LATEX_CMDS: List[str] = [
    "\\textbf", "\\textit", "\\textsl", "\\texttt", "\\textsc",
    "\\textsf", "\\textrm", "\\mathrm", "\\mathbf", "\\mathit",
    "\\mathsf", "\\mathtt", "\\mathbb", "\\mathcal", "\\mathscr",
    "\\mathfrak", "\\bm", "\\em", "\\emph", "\\underline",
    "\\overline", "\\tiny", "\\scriptsize", "\\footnotesize",
    "\\small", "\\normalsize", "\\large", "\\Large", "\\LARGE",
    "\\huge", "\\Huge", "\\newline", "\\par", "\\noindent",
    "\\indent", "\\footnote", "\\cite", "\\ref", "\\label",
    "\\textsuperscript", "\\textsubscript", "\\text", "\\mbox",
    "\\renewcommand{\\arraystretch}",
]

LATEX_FMT_ENVS: List[str] = [
    "align", "align*", "center", "flushleft", "flushright",
]
LATEX_LIST_ENVS: List[str] = ["itemize", "enumerate", "description"]

SIMPLE_RM_STRS: List[str] = [
    "\n", "\t", "approximately", "'", '"', "\\$", "$",
    "\uffe5", "\u00a3", "\u20ac", "{,}", "\\!", "\\,", "\\:", "\\;",
    "\\quad", "\\qquad", "\\space", "\\thinspace", "\\medspace",
    "\\thickspace", "~,", "\\ ",
    "\\\\%", "\\%", "%",
    "\\left", "\\right", "^{\\circ}", "^\\circ",
]

SIMPLE_REPLACE_MAP: T_Dict[str, str] = {
    "\u222a": "\\cup", "\u03c0": "\\pi", "\u221e": "\\infty",
    "\u2208": "\\in", "\u2229": "\\cap", "\u2212": "-",
    "\\item": ",", "and": ",", ";": ",",
    "infinity": "\\infty", "+\\infty": "\\infty",
    "tfrac": "frac", "dfrac": "frac",
    "\\approx": "=", "\\times": "*", "\\cdot": "*",
    "{.": "{0.", " .": " 0.", ":": "/",
}
