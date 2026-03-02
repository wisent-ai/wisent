"""EvaluatorMathBatch class for batch mathematical answer evaluation."""
from collections import Counter
from typing import List, Optional, Sequence, Set
from typing import Counter as T_Counter
from typing import Dict as T_Dict
from typing import Tuple as T_Tuple
from typing import Union as T_Union
from wisent.core.reading.evaluators.benchmark_specific.math_parsing._core_parts._sympy_utils import (
    DEF_ABS_TOL, DEF_N_PROC, DEF_PERCENT_REL_TOL, DEF_REL_TOL,
    DEF_TIMEOUT, is_querying4set,
)
from wisent.core.reading.evaluators.benchmark_specific.math_parsing._core_parts._evaluator_math import (
    EvaluatorMath,
)


class EvaluatorMathBatch(EvaluatorMath):
    """Batch evaluator for math problems.

    Parameters
    ----------
    ans_extract_mode: str, default: "boxed"
    include_percentage : bool, default: True
    rel_tol : float, default: DEF_REL_TOL
    abs_tol : float, default: DEF_ABS_TOL
    percent_rel_tol : float, default: DEF_PERCENT_REL_TOL
    ascii_only : bool, default: True
    timeout : int, default: DEF_TIMEOUT
    n_procs: int, default: DEF_N_PROC
    use_tqdm: bool, default: True
    """

    def __init__(
        self,
        ans_extract_mode: str,
        include_percentage: bool = True,
        rel_tol: float = DEF_REL_TOL,
        abs_tol: float = DEF_ABS_TOL,
        percent_rel_tol: float = DEF_PERCENT_REL_TOL,
        ascii_only: bool = True,
        timeout: int = DEF_TIMEOUT,
        n_procs: int = DEF_N_PROC,
        use_tqdm: bool = True,
    ):
        EvaluatorMath.__init__(
            self,
            include_percentage=include_percentage,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            percent_rel_tol=percent_rel_tol,
            ascii_only=ascii_only,
            ans_extract_mode=ans_extract_mode,
        )
        self.timeout = timeout
        self.n_procs = n_procs
        self.use_tqdm = use_tqdm

    def batch_eval(
        self,
        ref_answers: List[str],
        resps: List[str],
        problems: Optional[List[T_Union[str, bool]]] = None,
    ) -> T_Tuple[List[str], List[bool]]:
        """Evaluate a batch of `resps` against `ref_answers`."""
        pred_answers: List[str] = self.batch_extract_ans(resps)
        corrects: List[bool] = self.batch_eq(
            ref_answers, pred_answers, problems
        )
        return pred_answers, corrects

    def batch_get_eq_map(
        self,
        ref_answers: Sequence[str],
        pred_answers: Sequence[str],
        querying4set_flags: Sequence[bool],
    ) -> T_Dict[T_Tuple[str, str, bool], bool]:
        from wisent.core.reading.evaluators.benchmark_specific.math_parsing.core import (
            batch_exec,
        )
        corrects: List[bool] = batch_exec(
            self.eq,
            [
                {
                    "ref_ans": ref_ans,
                    "pred": pred,
                    "compare_sets": set_flag,
                }
                for ref_ans, pred, set_flag in zip(
                    ref_answers, pred_answers, querying4set_flags
                )
            ],
            n_procs=self.n_procs,
            timeout=self.timeout,
            use_tqdm=self.use_tqdm,
            desc="Judging",
            def_val=False,
        )
        eq_map: T_Dict[T_Tuple[str, str, bool], bool] = dict(
            zip(
                zip(ref_answers, pred_answers, querying4set_flags),
                corrects,
            )
        )
        return eq_map

    def batch_eq(
        self,
        ref_answers: Sequence[str],
        pred_answers: Sequence[str],
        problems: Optional[Sequence[T_Union[str, bool]]] = None,
    ) -> List[bool]:
        """Evaluate a batch of `pred_answers` against `ref_answers`."""
        assert len(ref_answers) == len(
            pred_answers
        ), f"{len(ref_answers) = } != {len(pred_answers) = }"
        set_flags: List[bool] = (
            [
                is_querying4set(p) if isinstance(p, str) else p
                for p in problems
            ]
            if problems is not None
            else [False] * len(ref_answers)
        )
        uniq_judge_data: Set[T_Tuple[str, str, bool]] = set(
            zip(ref_answers, pred_answers, set_flags)
        )
        uniq_ref_answers, uniq_pred_answers, uniq_set_flags = zip(
            *uniq_judge_data
        )
        uniq_judge_data2correct: T_Dict[T_Tuple[str, str, bool], bool] = (
            self.batch_get_eq_map(
                uniq_ref_answers, uniq_pred_answers, uniq_set_flags
            )
        )
        return [
            uniq_judge_data2correct[(ref, pred, set_flag)]
            for ref, pred, set_flag in zip(
                ref_answers, pred_answers, set_flags
            )
        ]

    def batch_extract_ans(self, resps: List[str]) -> List[str]:
        """Extract answers from a batch of responses."""
        from wisent.core.reading.evaluators.benchmark_specific.math_parsing.core import (
            batch_exec,
        )
        answers: List[str] = batch_exec(
            self.extract_ans,
            [{"resp_str": resp} for resp in resps],
            n_procs=self.n_procs,
            timeout=self.timeout,
            use_tqdm=self.use_tqdm,
            desc="Extracting",
            def_val="",
        )
        return answers

    def batch_get_maj_answers(
        self,
        answers_list: List[List[str]],
        problems: Optional[List[T_Union[str, bool]]] = None,
    ) -> T_Tuple[List[List[str]], List[List[str]]]:
        """Get the majority answers for a batch of answers."""
        maj_answers_list: List[List[str]] = []
        norm_answers_list: List[List[str]] = []
        all_judge_data: List[T_Tuple[str, str, bool]] = []
        set_flags: List[bool] = (
            [
                is_querying4set(problem) if isinstance(problem, str)
                else problem
                for problem in problems
            ]
            if problems is not None
            else [False] * len(answers_list)
        )
        for answers, set_flag in zip(answers_list, set_flags):
            all_judge_data.extend(
                (answer, answers[j], set_flag)
                for j, answer in enumerate(answers)
                if j < len(answers) - 1
            )
        all_ref_answers, all_pred_answers, all_set_flags = zip(
            *all_judge_data
        )
        all_judge_data2eq: T_Dict[T_Tuple[str, str, bool], bool] = (
            self.batch_get_eq_map(
                all_ref_answers, all_pred_answers, all_set_flags
            )
        )
        for answers, set_flag in zip(answers_list, set_flags):
            maj_answers: List[str] = []
            norm_answers: List[str] = []
            ans_vote: T_Counter[str] = Counter()
            for answer in answers:
                exist_ans = next(
                    (
                        exist_answer
                        for exist_answer in ans_vote
                        if all_judge_data2eq.get(
                            (answer, exist_answer, set_flag), False
                        )
                    ),
                    None,
                )
                norm_ans: str = (
                    exist_ans if exist_ans is not None else answer
                )
                ans_vote[norm_ans] += 1
                norm_answers.append(norm_ans)
                maj_answers.append(
                    self.get_maj_ans_from_votes(ans_vote)
                )
            maj_answers_list.append(maj_answers)
            norm_answers_list.append(norm_answers)
        return maj_answers_list, norm_answers_list
