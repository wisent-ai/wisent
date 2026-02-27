"""LM eval task definitions - batch 3."""
from wisent.core.tasks.eval._lm_eval_task_base import LMEvalTask

class MultipleRsTask(LMEvalTask):
    """Multiple-Rs task implementation."""

    def __init__(self):
        super().__init__(
            task_name="multiple_rs",
            description="Multiple-Rs: Multi-language Rust tasks",
            categories=["coding", "reasoning", "rust", "multi-language"],
        )


class MultipleGoTask(LMEvalTask):
    """Multiple-Go task implementation."""

    def __init__(self):
        super().__init__(
            task_name="multiple_go",
            description="Multiple-Go: Multi-language Go tasks",
            categories=["coding", "reasoning", "go", "multi-language"],
        )


class CodexglueCodeToTextPythonTask(LMEvalTask):
    """CodexGlue Code-to-Text Python task implementation."""

    def __init__(self):
        super().__init__(
            task_name="codexglue_code_to_text_python",
            description="CodexGlue Code-to-Text Python: Python code summarization",
            categories=["coding", "reasoning", "python", "code-to-text"],
        )


class CodexglueCodeToTextGoTask(LMEvalTask):
    """CodexGlue Code-to-Text Go task implementation."""

    def __init__(self):
        super().__init__(
            task_name="codexglue_code_to_text_go",
            description="CodexGlue Code-to-Text Go: Go code summarization",
            categories=["coding", "reasoning", "go", "code-to-text"],
        )


class CodexglueCodeToTextRubyTask(LMEvalTask):
    """CodexGlue Code-to-Text Ruby task implementation."""

    def __init__(self):
        super().__init__(
            task_name="codexglue_code_to_text_ruby",
            description="CodexGlue Code-to-Text Ruby: Ruby code summarization",
            categories=["coding", "reasoning", "ruby", "code-to-text"],
        )


class CodexglueCodeToTextJavaTask(LMEvalTask):
    """CodexGlue Code-to-Text Java task implementation."""

    def __init__(self):
        super().__init__(
            task_name="codexglue_code_to_text_java",
            description="CodexGlue Code-to-Text Java: Java code summarization",
            categories=["coding", "reasoning", "java", "code-to-text"],
        )


class CodexglueCodeToTextJavascriptTask(LMEvalTask):
    """CodexGlue Code-to-Text JavaScript task implementation."""

    def __init__(self):
        super().__init__(
            task_name="codexglue_code_to_text_javascript",
            description="CodexGlue Code-to-Text JavaScript: JavaScript code summarization",
            categories=["coding", "reasoning", "javascript", "code-to-text"],
        )


class CodexglueCodeToTextPhpTask(LMEvalTask):
    """CodexGlue Code-to-Text PHP task implementation."""

    def __init__(self):
        super().__init__(
            task_name="codexglue_code_to_text_php",
            description="CodexGlue Code-to-Text PHP: PHP code summarization",
            categories=["coding", "reasoning", "php", "code-to-text"],
        )


class RecodeTask(LMEvalTask):
    """Recode task implementation."""

    def __init__(self):
        super().__init__(
            task_name="recode",
            description="Recode: Perturbed HumanEval natural generation",
            categories=["coding", "reasoning", "python", "perturbation"],
        )


class Squad2Task(LMEvalTask):
    """SQuAD2 task implementation. Uses unified split strategy."""

    def __init__(self):
        super().__init__(
            task_name="squadv2",
            description="SQuAD2: Stanford Question Answering Dataset 2.0",
            categories=["reading-comprehension", "qa", "natural-language"],
        )
    # No longer needs special handling - unified split combines all available docs


class ArcEasyTask(LMEvalTask):
    """ARC-Easy task implementation."""

    def __init__(self):
        super().__init__(
            task_name="arc_easy",
            description="ARC-Easy: AI2 Reasoning Challenge (Easy Set)",
            categories=["reasoning", "science", "multiple-choice"],
        )


class ArcChallengeTask(LMEvalTask):
    """ARC-Challenge task implementation."""

    def __init__(self):
        super().__init__(
            task_name="arc_challenge",
            description="ARC-Challenge: AI2 Reasoning Challenge (Challenge Set)",
            categories=["reasoning", "science", "multiple-choice"],
        )


class HellaswagTask(LMEvalTask):
    """HellaSwag task implementation."""

    def __init__(self):
        super().__init__(
            task_name="hellaswag",
            description="HellaSwag: Commonsense NLI for sentence completion",
            categories=["reasoning", "commonsense", "multiple-choice"],
        )


class WinograndeTask(LMEvalTask):
    """WinoGrande task implementation."""

    def __init__(self):
        super().__init__(
            task_name="winogrande",
            description="WinoGrande: Large-scale Winograd Schema Challenge",
            categories=["reasoning", "commonsense", "coreference"],
        )


class PiqaTask(LMEvalTask):
    """PIQA task implementation."""

    def __init__(self):
        super().__init__(
            task_name="piqa",
            description="PIQA: Physical Interaction Question Answering",
            categories=["reasoning", "commonsense", "physical"],
        )


class BoolqTask(LMEvalTask):
    """BoolQ task implementation."""

    def __init__(self):
        super().__init__(
            task_name="boolq",
            description="BoolQ: Boolean Questions reading comprehension",
            categories=["reading-comprehension", "qa", "boolean"],
        )


class OpenbookqaTask(LMEvalTask):
    """OpenBookQA task implementation."""

    def __init__(self):
        super().__init__(
            task_name="openbookqa",
            description="OpenBookQA: Open-book science question answering",
            categories=["reasoning", "science", "multiple-choice"],
        )
