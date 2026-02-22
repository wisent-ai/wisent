"""EvaluatorMath class for mathematical answer evaluation."""
import re as regex
from math import isclose
from typing import Any, List, Match, Optional
from typing import Dict as T_Dict
from typing import Tuple as T_Tuple
from typing import Union as T_Union
from sympy import Expr, Matrix
from sympy.parsing.latex.errors import LaTeXParsingError
from sympy.parsing.sympy_parser import parse_expr
from wisent.core.errors import InvalidDataFormatError, InvalidValueError
from wisent.core.evaluators.benchmark_specific.math_parsing._core_parts._sympy_utils import (
    DEF_ABS_TOL, DEF_PERCENT_REL_TOL, DEF_REL_TOL,
    has_non_ascii, is_querying4set, is_set,
    latex2sympy_fix, latex2sympy_interval, norm_str2weekday, parse,
)
from wisent.core.evaluators.benchmark_specific.math_parsing._core_parts._evaluator_math_helpers import (
    EvaluatorMathHelpersMixin,
)


class EvaluatorMath(EvaluatorMathHelpersMixin):
    """Evaluator for math problems, capable of extracting answer
    segment from complex resp and processing various mathematical
    objects (e.g. fractions, symbolic expressions, matrices, vectors)
    and special text (e.g. bool values).

    Parameters
    ----------
    ans_extract_mode: str, default: "boxed"
    include_percentage : bool, default: True
    rel_tol : float, default: DEF_REL_TOL
    abs_tol : float, default: DEF_ABS_TOL
    percent_rel_tol : float, default: DEF_PERCENT_REL_TOL
    ascii_only : bool, default: True
    """

    def __init__(
        self,
        ans_extract_mode: str = "boxed",
        include_percentage: bool = True,
        rel_tol: float = DEF_REL_TOL,
        abs_tol: float = DEF_ABS_TOL,
        percent_rel_tol: float = DEF_PERCENT_REL_TOL,
        ascii_only: bool = True,
    ):
        self.ans_extract_mode: str = ans_extract_mode
        self.include_percentage: bool = include_percentage
        self.rel_tol: float = rel_tol
        self.abs_tol = abs_tol
        self.percent_rel_tol: float = percent_rel_tol
        self.ascii_only: bool = ascii_only

    def extract_ans(self, resp_str: str) -> str:
        from wisent.core.evaluators.benchmark_specific.math_parsing.core import (
            EvaluatorBase,
        )
        raw_ans: str = EvaluatorBase().extract_ans(resp_str)
        math_ans: str = self.norm_ans_str(raw_ans)
        return math_ans

    def eq(
        self,
        ref_ans: T_Union[str, T_Tuple[str, float]],
        pred: str,
        compare_sets: bool = False,
    ) -> bool:
        """Check if two values are mathematically equal."""
        ref: str
        ref_num: Optional[float]
        if isinstance(ref_ans, (list, tuple)) and len(ref_ans) == 2:
            ref, ref_num = ref_ans
        else:
            ref = ref_ans
            ref_num = None
        if ref is None:
            return None
        if pred is None:
            return False
        # datetime
        pred_datetime: Optional[str] = self.norm_str2date_time(pred)
        ref_datetime: Optional[str] = self.norm_str2date_time(ref)
        if (
            pred_datetime is not None
            and ref_datetime is not None
            and pred_datetime == ref_datetime
        ):
            return True
        # 0. Normalize
        pred_str: str = self.norm_ans_str(pred)
        ref_str: str = self.norm_ans_str(ref)
        if len(pred_str) == 0:
            return False
        if self.ascii_only and has_non_ascii(pred_str):
            return False
        # 1. literally equal
        lower_pred: str = pred_str.lower()
        lower_ref: str = ref_str.lower()
        if lower_pred == lower_ref:
            return True
        if compare_sets:
            preds: List[str] = self.extract_set(pred_str)
            refs: List[str] = self.extract_set(ref_str)
            if len(preds) != len(refs):
                return False
            for pred in preds:
                exist = False
                for ref in refs:
                    exist: bool = self.eq(
                        pred, ref, compare_sets=False,
                    )
                    if exist:
                        break
                if not exist:
                    return False
                refs.remove(ref)
            return True
        pred_parse_errs: List[Exception] = []
        ref_parse_errs: List[Exception] = []
        # 2. Numerically equal
        pred_num: Optional[float] = parse(float, pred_str, pred_parse_errs)
        if ref_num is None:
            ref_num = parse(float, ref_str, ref_parse_errs)
        num_eq: Optional[bool] = self.is_num_eq(ref_num, pred_num)
        if num_eq is not None:
            return num_eq
        # 3. Symbolically equal
        # 3.1 Python object
        pred_obj: Optional[Any] = parse(
            parse_expr, pred_str, pred_parse_errs
        )
        ref_obj: Optional[Any] = parse(
            parse_expr, ref_str, ref_parse_errs
        )
        if (
            pred_obj is not None
            and ref_obj is not None
            and pred_obj == ref_obj
        ):
            return True
        # 3.2 SymPy interval
        pred_spobj: Optional[Expr] = parse(
            latex2sympy_interval, pred_str, pred_parse_errs
        )
        ref_spobj: Optional[Expr] = parse(
            latex2sympy_interval, ref_str, ref_parse_errs
        )
        if (
            pred_spobj is not None
            and ref_spobj is not None
            and self.is_sym_eq(pred_spobj, ref_spobj)
        ):
            return True
        # 3.3 Matrix
        pred_spobj = parse(self.latex2matrix, pred_str, pred_parse_errs)
        ref_spobj = parse(self.latex2matrix, ref_str, ref_parse_errs)
        if (
            pred_spobj is not None
            and ref_spobj is not None
            and self.is_sym_eq(pred_spobj, ref_spobj)
        ):
            return True
        # 3.4 LaTeX with fix
        pred_spobj = parse(latex2sympy_fix, pred_str, pred_parse_errs)
        ref_spobj = parse(latex2sympy_fix, ref_str, ref_parse_errs)
        if (
            pred_spobj is not None
            and ref_spobj is not None
            and self.is_sym_eq(pred_spobj, ref_spobj)
        ):
            return True
        if (
            pred_spobj is not None
            and ref_obj is not None
            and self.is_sym_eq(pred_spobj, ref_obj)
        ):
            return True
        if (
            pred_obj is not None
            and ref_spobj is not None
            and self.is_sym_eq(pred_obj, ref_spobj)
        ):
            return True
        n_checks = 5
        expr_parse_errs: T_Dict[str, List[Exception]] = {}
        if len(pred_parse_errs) == n_checks:
            expr_parse_errs["pred"] = pred_parse_errs
        if len(ref_parse_errs) == n_checks:
            expr_parse_errs["ref"] = ref_parse_errs
        if len(expr_parse_errs) > 0:
            raise InvalidDataFormatError(reason=str(expr_parse_errs))
        else:
            return False

    def could_be_percent(self, v: T_Union[float, str]) -> bool:
        """Check if a value could be a percentage."""
        return 0 < v < 1 or 1 < v < 100

    def is_num_eq(
        self, ref_num: Optional[float], pred_num: Optional[float]
    ) -> Optional[bool]:
        """Compare two numbers with tolerance and percentage."""
        if ref_num is None or pred_num is None:
            return None
        if isclose(
            ref_num, pred_num, rel_tol=self.rel_tol, abs_tol=self.abs_tol
        ):
            return True
        if self.include_percentage and self.could_be_percent(pred_num):
            percent_ref_nums: List[float] = [
                num
                for num in [ref_num / 100, ref_num * 100]
                if self.could_be_percent(num)
            ]
            for item in percent_ref_nums:
                if isclose(
                    item, pred_num,
                    rel_tol=self.percent_rel_tol, abs_tol=self.abs_tol
                ):
                    return True
        return None

    def norm_ans_str(self, ans: str) -> str:
        """Normalize answer string for all kinds of answers."""
        from wisent.core.evaluators.benchmark_specific.math_parsing.core import (
            norm_str2bool,
        )
        ans = str(ans)
        ans = ans.replace("\n", "")
        ans = ans.strip()
        ans = self.clean(ans)
        ans_bool = norm_str2bool(ans)
        if ans_bool is not None:
            return str(ans_bool)
        ans_weekday = norm_str2weekday(ans)
        if ans_weekday is not None:
            return ans_weekday
        ans = self.norm_math_str(ans)
        return ans

    def latex2matrix(self, latex_mat_str: str) -> Matrix:
        """Convert latex matrix into sympy matrix."""
        if not isinstance(latex_mat_str, str):
            raise InvalidValueError(
                param_name="latex_mat_str",
                actual=type(latex_mat_str).__name__,
                expected="str",
            )
        latex_mat_str = latex_mat_str.replace(" ", "")
        pattern = (
            r"(?:\[|\()?\\begin{[a-zA-Z]?(?:matrix|array)}"
            r"(?:\[lcr\])*?(.*)\\end{[a-zA-Z]?(?:matrix|array)}(?:\]|\))?"
        )
        data: Optional[Match[str]] = regex.search(pattern, latex_mat_str)
        python_matrix: List[List[str]] = []
        if data is not None:
            rows: List[str] = regex.split(r"\\+(?!frac|sqrt)", data[1])
            for row in rows:
                elements_list: List[str] = row.split("&")
                python_matrix.append(elements_list)
        else:
            if "," in latex_mat_str:
                if is_set(latex_mat_str):
                    python_matrix = [self.extract_set(latex_mat_str)]
                else:
                    python_matrix = [
                        self.remove_out_paren(latex_mat_str).split(",")
                    ]
            else:
                raise LaTeXParsingError(
                    f"{latex_mat_str} can not be parsed in a `Matrix`!"
                )
        sympy_matrix = []
        for row in python_matrix:
            sympy_row = [latex2sympy_fix(element) for element in row]
            sympy_matrix.append(sympy_row)
        matrix = Matrix(sympy_matrix)
        if len(matrix.shape) == 2 and matrix.shape[1] == 1:
            matrix = matrix.T
        return matrix
