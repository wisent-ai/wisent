"""Core math parsing module for benchmark evaluation.

This module was split into submodules under _core_parts/ to comply
with the 300-line file limit. All public symbols are re-exported here.
"""
import os
import re as regex
import warnings
from collections import Counter
from typing import Any, Callable, List, Optional
from typing import Counter as T_Counter
from typing import Dict as T_Dict
from typing import Tuple as T_Tuple
from typing import Union as T_Union
from pebble import ProcessPool
from sympy import *  # noqa: F401,F403 - needed for eval()
from sympy.parsing.latex import parse_latex  # noqa: F401
from sympy.parsing.latex.errors import LaTeXParsingError  # noqa: F401
from sympy.parsing.sympy_parser import parse_expr  # noqa: F401
from sympy.utilities.exceptions import SymPyDeprecationWarning
from tqdm import tqdm

warnings.filterwarnings("ignore", category=SymPyDeprecationWarning)

STRIP_STRS: List[str] = [
    ":", "/", ",", "#", "?", "$", '"', "'",
    "\u043a", "\u0438",
    "\\(", "\\)", "\\[", "\\]",
]
NO_TRAILING_STRS: List[str] = ["(", "[", "{", "\\", "."] + STRIP_STRS
NO_PRECEDING_PUNCS: List[str] = [
    "!", ")", "]", "}", "\\\\", "boxed"
] + STRIP_STRS
PRM800K_ANS_PRRFIX = "# Answer"
GSM8K_ANS_PREFIX = "####"


def extract_boxed(resp: str) -> str:
    ans: str = resp.split("oxed")[-1]
    a: str
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for i_pre, c in enumerate(ans[1:]):
            if ans[i_pre] == "\\":
                a += c
                continue
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a


def norm_str2bool(s: str) -> Optional[bool]:
    """Converts a string to its corresponding boolean value."""
    s = str(s).lower().strip().replace("noindent", "")
    if any(pos in s for pos in ["yes", "true"]):
        return True
    elif any(neg in s for neg in ["no", "false"]):
        return False
    else:
        return None


class EvaluatorBase:
    """Base class for evaluators."""

    def __init__(self, ans_extract_mode: str = "boxed"):
        self.ans_extract_mode: str = ans_extract_mode

    def eq(self, ref_ans: str, pred: str) -> bool:
        return ref_ans == pred

    def extract_ans(self, resp_str: str) -> str:
        if self.ans_extract_mode == "boxed":
            if "oxed{" in resp_str:
                return extract_boxed(resp_str)
            else:
                return ""
        if self.ans_extract_mode == "explicit":
            ans: Optional[str] = self.extract_explicit_ans(resp_str)
            if ans is not None:
                return ans
        if self.ans_extract_mode == "speculate":
            matches: List[str] = regex.findall(
                r"(?:\$|\\\(|\\\[)([^\$]+)(?:\$|\\\(|\\\[)",
                resp_str, regex.DOTALL,
            )
            if len(matches) > 0:
                return matches[-1]
            matches = regex.findall(
                r"-?\d*\.?\d+", resp_str.replace(",", "")
            )
            if len(matches) > 0:
                return matches[-1]
        return ""

    def extract_explicit_ans(self, resp_str: str) -> Optional[str]:
        resp_str = self.clean_trailing(resp_str)
        if "<|start_answer|>" in resp_str and "<|end_answer|>" in resp_str:
            return (
                resp_str.split("<|start_answer|>")[1]
                .split("<|end_answer|>")[0].strip()
            )
        if GSM8K_ANS_PREFIX in resp_str:
            resp_str = resp_str.split(GSM8K_ANS_PREFIX)[-1].strip()
        if PRM800K_ANS_PRRFIX in resp_str:
            resp_str = resp_str.split(PRM800K_ANS_PRRFIX)[-1].strip()
        resp: str
        if "oxed{" in resp_str:
            resp = extract_boxed(resp_str)
        else:
            resp = resp_str
            if "is the ans" in resp:
                resp = regex.split(
                    r"(,|\.|\!\|?)",
                    resp.split("is the ans")[-2].strip(),
                )[-1].strip()
            elif "is our ans" in resp:
                resp = regex.split(
                    r"(,|\.|\!\|?)",
                    resp.split("is our ans")[-2].strip(),
                )[-1].strip()
            elif "answer is" in resp:
                resp = resp.split("answer is")[-1].strip()
            elif "answer:" in resp:
                resp = resp.split("answer:")[-1].strip()
            elif "answer :" in resp:
                resp = resp.split("answer :")[-1].strip()
            elif "statement" in resp:
                bool_resp = norm_str2bool(resp.split("is ")[-1].strip())
                if bool_resp is not None:
                    return str(bool_resp)
            else:
                return None
            if resp.startswith("$") and resp.endswith("$"):
                resp = resp[1:-1]
        return resp

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

    def get_maj_answers(self, answers: List[str]) -> List[str]:
        maj_answers: List[str] = []
        ans_vote: T_Counter[str] = Counter()
        for answer in answers:
            for exist_ans in ans_vote:
                try:
                    correct = self.eq(answer, exist_ans)
                except Exception:
                    correct = False
                if correct:
                    ans_vote[exist_ans] += 1
                    break
            else:
                ans_vote[answer] += 1
            maj_ans = self.get_maj_ans_from_votes(ans_vote)
            maj_answers.append(maj_ans)
        return maj_answers

    def get_maj_ans_from_votes(
        self, ans_vote: T_Union[T_Counter[str], T_Dict[str, int]]
    ) -> str:
        if isinstance(ans_vote, dict):
            ans_vote = Counter(ans_vote)
        maj_ans = ans_vote.most_common(1)[0][0]
        if maj_ans == "" and len(ans_vote) > 1:
            maj_ans = ans_vote.most_common(2)[1][0]
        return maj_ans


from wisent.core.constants import DEFAULT_TIMEOUT_QUICK, MAX_TASKS_PER_PROC

DEF_TIMEOUT: int = DEFAULT_TIMEOUT_QUICK
DEF_N_PROC: int = os.cpu_count() // 2


def batch_exec(
    func: Callable[..., Any],
    kwargs_list: List[T_Dict[str, Any]],
    n_procs: int = DEF_N_PROC,
    timeout: int = DEF_TIMEOUT,
    use_tqdm: bool = True,
    desc: str = "Processing",
    def_val: Any = None,
    max_tasks_per_proc: int = MAX_TASKS_PER_PROC,
) -> List[Any]:
    """Execute a function in batch using ProcessPool."""
    n_samples: int = len(kwargs_list)
    n_procs = min(n_procs, n_samples)
    results: List[Any] = [def_val] * n_samples
    with ProcessPool(
        max_workers=n_procs, max_tasks=max_tasks_per_proc
    ) as pool:
        future = pool.map(
            task_wrapper, [func] * len(kwargs_list),
            kwargs_list, timeout=timeout,
        )
        iterator = future.result()
        pbar = tqdm(total=n_samples, desc=desc) if use_tqdm else None
        idx: int = 0
        while idx < n_samples:
            try:
                result: Any = next(iterator)
                results[idx] = result
            except StopIteration:
                break
            except Exception:
                pass
            if pbar is not None:
                pbar.update(1)
            idx += 1
        if pbar:
            pbar.close()
    return results


def task_wrapper(
    func: Callable[..., Any], kwargs: T_Dict[str, Any]
) -> Any:
    return func(**kwargs)


# Re-export all public symbols from split parts
from wisent.core.evaluators.benchmark_specific.math_parsing._core_parts._sympy_utils import (  # noqa: E402,F811
    BASIC_FN_NAMES, DATETIME_FMTS, DEF_ABS_TOL, DEF_PERCENT_REL_TOL,
    DEF_REL_TOL, LATEX_CMDS, LATEX_FMT_ENVS, LATEX_LIST_ENVS,
    PAREN_MAP, SIMPLE_REPLACE_MAP, SIMPLE_RM_STRS, STR2NUM, UNITS,
    WEEKDAY_ABBRS, WEEKDAY_FULLS,
    fix_a_slash_b, fix_fracs, fix_sqrt, has_non_ascii,
    is_querying4set, is_set, latex2sympy_fix, latex2sympy_interval,
    norm_deg, norm_str2weekday, parse, rm_latex_env, run_with_timeout,
)
from wisent.core.evaluators.benchmark_specific.math_parsing._core_parts._evaluator_math import (  # noqa: E402
    EvaluatorMath,
)
from wisent.core.evaluators.benchmark_specific.math_parsing._core_parts._evaluator_batch import (  # noqa: E402
    EvaluatorMathBatch,
)
