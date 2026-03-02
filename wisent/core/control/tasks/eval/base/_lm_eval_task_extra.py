"""LM eval task definitions - batch 2."""
from wisent.core.control.tasks.eval._lm_eval_task_base import LMEvalTask

class MULTIRCTask(LMEvalTask):
    """MULTIRC task implementation. Uses unified split strategy."""

    def __init__(self):
        super().__init__(
            task_name="multirc",
            description="MultiRC: Multi-Sentence Reading Comprehension",
            categories=["reasoning", "long context", "general knowledge"]
        )
    # No longer needs special handling - unified split combines all available docs


class TruthfulQATask(LMEvalTask):
    """TruthfulQA task implementation. Uses unified split strategy."""

    def __init__(self):
        super().__init__(
            task_name="truthfulqa_mc1",
            description="TruthfulQA: Truthfulness evaluation benchmark",
            categories=["hallucination", "general-knowledge", "reasoning"],
        )
    # No longer needs special handling - unified split combines all available docs


class MMLUTask(LMEvalTask):
    """MMLU task implementation."""

    def __init__(self):
        super().__init__(
            task_name="mmlu",
            description="MMLU: Massive Multitask Language Understanding",
            categories=["general-knowledge", "science", "reasoning"],
        )

class QA4MRETask(LMEvalTask):
    """QA4MRE task implementation"""

    def __init__(self):
        super().__init__(
            task_name="qa4mre_2013",
            description="QA4MRE: Question Answering for Machine Reading Evaluation",
            categories=["multiple-choice", "long context"],
        )


# === CODING TASKS ===


class InstructHumanEvalTask(LMEvalTask):
    """InstructHumanEval task implementation."""

    def __init__(self):
        super().__init__(
            task_name="instructhumaneval",
            description="InstructHumanEval: Instruction-following HumanEval benchmark",
            categories=["coding", "reasoning", "python", "instruction-following"],
        )


class HumanEvalPlusTask(LMEvalTask):
    """HumanEval Plus task implementation."""

    def __init__(self):
        super().__init__(
            task_name="humaneval_plus",
            description="HumanEval Plus: Extended HumanEval with more tests",
            categories=["coding", "reasoning", "python"],
        )


class ConalaTask(LMEvalTask):
    """Conala task implementation."""

    def __init__(self):
        super().__init__(
            task_name="conala",
            description="Conala: Code generation from natural language",
            categories=["coding", "reasoning", "python", "nl2code"],
        )


class ConcodeTask(LMEvalTask):
    """Concode task implementation."""

    def __init__(self):
        super().__init__(
            task_name="concode",
            description="Concode: Code completion benchmark",
            categories=["coding", "reasoning", "completion"],
        )


class MercuryTask(LMEvalTask):
    """Mercury task implementation."""

    def __init__(self):
        super().__init__(
            task_name="mercury",
            description="Mercury: Code generation benchmark",
            categories=["coding", "reasoning"],
        )


class AppsTask(LMEvalTask):
    """APPS task implementation."""

    def __init__(self):
        super().__init__(
            task_name="apps",
            description="APPS: Automated Programming Problems Synthesis",
            categories=["coding", "reasoning", "python", "competitive"],
        )


class DS1000Task(LMEvalTask):
    """DS1000 task implementation."""

    def __init__(self):
        super().__init__(
            task_name="ds1000",
            description="DS1000: Data Science coding tasks",
            categories=["coding", "reasoning", "python", "data-science"],
        )


class MultiplePyTask(LMEvalTask):
    """Multiple-Py task implementation."""

    def __init__(self):
        super().__init__(
            task_name="multiple_py",
            description="Multiple-Py: Multi-language Python tasks",
            categories=["coding", "reasoning", "python", "multi-language"],
        )


class MultipleJsTask(LMEvalTask):
    """Multiple-JS task implementation."""

    def __init__(self):
        super().__init__(
            task_name="multiple_js",
            description="Multiple-JS: Multi-language JavaScript tasks",
            categories=["coding", "reasoning", "javascript", "multi-language"],
        )


class MultipleJavaTask(LMEvalTask):
    """Multiple-Java task implementation."""

    def __init__(self):
        super().__init__(
            task_name="multiple_java",
            description="Multiple-Java: Multi-language Java tasks",
            categories=["coding", "reasoning", "java", "multi-language"],
        )


class MultipleCppTask(LMEvalTask):
    """Multiple-Cpp task implementation."""

    def __init__(self):
        super().__init__(
            task_name="multiple_cpp",
            description="Multiple-Cpp: Multi-language C++ tasks",
            categories=["coding", "reasoning", "cpp", "multi-language"],
        )


