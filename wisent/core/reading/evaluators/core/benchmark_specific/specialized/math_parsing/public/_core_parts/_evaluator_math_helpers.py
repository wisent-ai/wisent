"""EvaluatorMath helper mixin with LaTeX/math normalization methods."""
import re as regex
from collections import Counter
from datetime import datetime
from typing import Any, List, Match, Optional
from typing import Counter as T_Counter
from typing import Dict as T_Dict
from typing import Tuple as T_Tuple
from typing import Union as T_Union
from sympy import N, simplify
from wisent.core.reading.evaluators.benchmark_specific.math_parsing._core_parts._sympy_utils import (
    BASIC_FN_NAMES, DATETIME_FMTS, LATEX_CMDS, LATEX_FMT_ENVS,
    LATEX_LIST_ENVS, NO_PRECEDING_PUNCS, NO_TRAILING_STRS,
    PAREN_MAP, SIMPLE_REPLACE_MAP, SIMPLE_RM_STRS, STR2NUM, UNITS,
    fix_a_slash_b, fix_fracs, fix_sqrt, is_set, norm_deg,
    latex2sympy_fix, rm_latex_env,
)


class EvaluatorMathHelpersMixin:
    """Mixin providing helper methods for EvaluatorMath."""

    def clean(self, ans: str) -> str:
        ans = ans.strip()
        ans = self.clean_preceding(ans)
        ans = self.clean_trailing(ans)
        return ans

    def clean_preceding(self, s: str) -> str:
        s = str(s).strip()
        while s != "" and s[0] in NO_PRECEDING_PUNCS:
            s = s[1:].strip()
        return s

    def clean_trailing(self, s: str) -> str:
        s = str(s).strip()
        while s != "" and s[-1] in NO_TRAILING_STRS:
            s = s[:-1].strip()
        return s

    def get_maj_ans_from_votes(
        self, ans_vote: T_Union[T_Counter[str], T_Dict[str, int]]
    ) -> str:
        if isinstance(ans_vote, dict):
            ans_vote = Counter(ans_vote)
        maj_ans = ans_vote.most_common(1)[0][0]
        if maj_ans == "" and len(ans_vote) > 1:
            maj_ans = ans_vote.most_common(2)[1][0]
        return maj_ans

    def remove_latex_cmd(self, s: str, cmd: str) -> str:
        try:
            cmd_idx = s.index(cmd)
        except ValueError:
            return s
        pfx = s[:cmd_idx].strip()
        sfx = s[cmd_idx + len(cmd) :].strip()
        if len(sfx) > 0 and sfx[0] == "{":  # Common command
            sfx = self.remove_first_paren_pair(sfx, "{")
        elif len(pfx) > 0 and pfx[-1] == "{":  # Declaration command
            left_idx_in_sfx = sfx.find("}")
            if left_idx_in_sfx != -1:
                pfx = pfx[:-1]
                sfx = sfx[:left_idx_in_sfx] + sfx[left_idx_in_sfx + 1 :]
        else:  # Independent command
            pass
        return pfx + sfx

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
            diff_rev = simplify(b - a)
            if hasattr(diff, "__iter__") and hasattr(diff_rev, "__iter__"):
                if all(element == 0 for element in diff) and all(
                    element == 0 for element in diff_rev
                ):
                    return True
            else:
                if not diff and not diff_rev:
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

    def norm_str2date_time(self, string: str) -> Optional[str]:
        """Normalize date or time string to a standard format."""
        for fmt in DATETIME_FMTS:
            try:
                dt: datetime = datetime.strptime(string, fmt)
                has_time: bool = ":" in string
                has_date: bool = "/" in string or "-" in string
                if has_date and has_time:
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                elif has_date:
                    return dt.strftime("%Y-%m-%d")
                elif has_time:
                    return dt.strftime("%H:%M:%S")
                else:
                    pass
            except ValueError:
                continue
        return None

    def index_first_paren_pair(self, s: str, l: str) -> T_Tuple[int, int]:
        r: str = PAREN_MAP[l]
        try:
            i_l: int = s.index(l)
        except ValueError:
            return -1, -1
        len_paren: int = len(l)
        depth = 0
        i_r: int = -1
        for i_c in range(i_l, len(s)):
            if s[i_c : i_c + len_paren] == l:
                depth -= 1
            elif s[i_c : i_c + len_paren] == r:
                depth += 1
            if depth == 0:
                i_r = i_c
                break
        return i_l, i_r

    def remove_first_paren_pair(self, s: str, l: str) -> str:
        i_l: int
        i_r: int
        i_l, i_r = self.index_first_paren_pair(s, l)
        if i_l != -1 and i_r != -1:
            len_paren: int = len(l)
            s = s[:i_l] + s[i_l + len_paren : i_r] + s[i_r + len_paren :]
        return s

    def remove_out_paren(self, s: str) -> str:
        """Remove until there are no parentheses outside."""
        done: bool = False
        while not done:
            done = True
            for left, _ in PAREN_MAP.items():
                len_paren: int = len(left)
                i_l: int
                i_r: int
                i_l, i_r = self.index_first_paren_pair(s, left)
                if i_l == 0 and i_r == len(s) - len_paren:
                    s = s[len_paren:-len_paren]
                    done = False
        return s

    def extract_set(self, norm_s: str) -> List[str]:
        clean_s: str = self.remove_out_paren(norm_s)
        ele_strs: List[str] = clean_s.replace("or", ",").split(",")
        ele_strs: List[str] = [s.strip() for s in ele_strs]
        merged_strs: List[str] = []
        for i in range(len(ele_strs)):
            s_i: str = ele_strs[i]
            existing = False
            for j in range(i):
                s_j: str = ele_strs[j]
                if self.eq(s_i, s_j):
                    existing = True
                    break
            if not existing:
                merged_strs.append(s_i)
        merged_strs.sort()
        return merged_strs

    def norm_basic_fn(self, s: str) -> str:
        """Normalize basic function expressions."""
        s = regex.sub(
            rf"\\?({'|'.join(BASIC_FN_NAMES)})\^(\d+)", r"\\\1^{\2}", s
        )
        s = regex.sub(
            rf"\\?({'|'.join(BASIC_FN_NAMES)})(?!\^)", r"\\\1^{1}", s
        )
        return s

    def norm_pm(self, s: str) -> str:
        """Replaces '\\pm' or '\\mp' with expanded form."""
        def replace_pm(match: Match[str]) -> str:
            first_part: str
            second_part: str
            first_part, second_part = match.groups()
            return f"{first_part}-{second_part},{first_part}+{second_part}"
        _s = self.remove_out_paren(s)
        pattern = (
            r"([\w\.\\{}\+\-\*\^]+?)(?:\\pm|\\mp)([\w\.\\{}\+\-\*\^]+)"
        )
        if regex.search(pattern, _s):
            return regex.sub(pattern, replace_pm, _s)
        else:
            return s

    def norm_math_str(self, string: str) -> str:
        string = str(string).strip()
        string = self.clean(string)
        for rm_str in SIMPLE_RM_STRS:
            string = string.replace(rm_str, "")
        for k, v in SIMPLE_REPLACE_MAP.items():
            string = string.replace(k, v)
        if "\\infty" not in string:
            string = string.replace("inf", "\\infty")
        string = string.replace(" ", "")
        for latex_cmd in LATEX_CMDS:
            string = self.remove_latex_cmd(string, latex_cmd)
        for env in LATEX_FMT_ENVS + LATEX_LIST_ENVS:
            string = rm_latex_env(string, env)
        string = norm_deg(string)
        string = regex.sub(
            rf"(?<!\\)(pi\b|{'|'.join(BASIC_FN_NAMES)})", r"\\\1", string
        )
        string = self.norm_basic_fn(string)
        string = regex.sub(r"{[a-z]?matrix}", r"{array}", string)
        string = regex.sub(
            r"\\begin{array}{[lcr]*}", r"\\begin{array}{}", string
        )
        if "\\begin{array}" not in string:
            string = string.replace("\\\\", "")
        if "j" in string and "i" not in string:
            string = string.replace("j", "i")
        string = regex.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
        string = regex.sub(r"(\d+)\.0+$", r"\1", string)
        for unit in UNITS:
            string = regex.sub(
                f"([-\\d\\.\\*\\^{{}}]+){unit}e?s?.*", "\\1", string
            )
        if len(string) == 0:
            return string
        if string[0] == ".":
            string = "0" + string
        s_is_set: bool = is_set(string)
        raw_strings: List[str]
        if s_is_set:
            raw_strings = self.extract_set(string)
        else:
            raw_strings = [string]
        strings: List[str] = []
        for string in raw_strings:
            string = fix_sqrt(string)
            if string.startswith("frac"):
                string = "\\" + string
            string = fix_fracs(string)
            string = fix_a_slash_b(string)
            string = regex.sub(r"^[a-z]\\in", "", string)
            if "," not in string:
                string = self.remove_out_paren(string)
            if "\\begin{array}" not in string:
                if len(string.split("=")) > 2:
                    string = string.split("=")[-1]
                if len(string.split("=")) == 2:
                    first_part = string.split("=")[0].strip()
                    if (
                        regex.match(
                            r"^([a-z]|[A-Z]{2}|\\?(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|sin|cos|sec|csc|tan|cot|sinh|cosh|sech|csch|tanh|coth|log|ln|exp))\^?{?-?('|\\prime|\d)*}?(\(-?([\d\.]+|[a-z])?\))?$",
                            first_part,
                        )
                        is not None
                    ):
                        string = string.split("=")[1]
                if len(string.split("=")) == 2:
                    if (
                        len(regex.findall(
                            r"[a-zA-Z]", string.split("=")[0].strip()
                        )) == 0
                    ):
                        string = string.split("=")[1]
            string = self.norm_pm(string)
            string = regex.sub(r"^0+([1-9])", r"\1", string)
            strings.append(string)
        string = ",".join(strings)
        if "," not in string:
            string = self.remove_out_paren(string)
        if STR2NUM.get(string):
            string = str(STR2NUM[string])
        string = regex.sub(r"\\mid([a-z])", r"\\mid \1", string)
        string = self.clean(string)
        for ineq in ["<", ">"]:
            if len(regex.findall(f"{ineq}=?", string)) > 1 and not any(
                delim in string.lower() for delim in [",", "and", "or"]
            ):
                string = string.replace(ineq, ",")
        return string
