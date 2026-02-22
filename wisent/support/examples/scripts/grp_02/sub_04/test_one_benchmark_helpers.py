"""Helpers for test_one_benchmark."""


class MockModel:
    """Mock model that returns predictable outputs without actual inference.

    This mock ensures that:
    - For log_likelihoods: first choice always has higher log prob
    - For perplexity: returns low perplexity for first choice
    - For generation: returns empty (not used in contrastive evaluation)
    """

    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name

    def get_log_probs(self, prompt: str, choices: list[str]) -> list[float]:
        """Return mock log probabilities - first choice always has higher probability."""
        return [-0.5] + [-2.0] * (len(choices) - 1) if len(choices) >= 1 else []

    def loglikelihood(self, context: str, continuation: str) -> float:
        """Return mock log likelihood for perplexity evaluator."""
        return -len(continuation) * 0.1

    def generate(self, prompt: str, **kwargs) -> str:
        """Mock generation - returns empty as we use choices for evaluation."""
        return "mock generation"


HF_TASKS = [
    "math", "math_500", "aime", "hmmt", "polymath", "livemathbench",
    "humaneval", "humaneval_plus",
    "instruct_humaneval", "apps", "conala", "concode",
    "ds", "ds1000", "ds_1000", "mercury", "recode",
    "multipl", "multiple_", "multipl_e",
    "codexglue", "livecodebench",
    "super_gpqa", "supergpqa", "hle",
    "tag",
    "meddialog",
    "mmlusr",
    "iwslt2017",
    "stsb",
    "babilong", "bangla_mmlu",
    "bhtc_v2", "basque-glue", "basqueglue",
    "flan_held_in",
    "gpt3_translation_benchmarks",
    "penn_treebank", "ptb",
    "self_consistency", "t0_eval",
    "wikitext103",
]

LM_EVAL_ONLY_TASKS = [
    "minerva_math", "code_x_glue", "humaneval_infilling", "mathqa",
    "multiple_choice",
    "vaxx_stance", "wiceu",
]


def detect_loader_type(task_name: str) -> str:
    """Auto-detect loader type for a task name."""
    if any(
        task_name.lower() == t or task_name.lower().startswith(t + "_")
        for t in LM_EVAL_ONLY_TASKS
    ):
        return "lm_eval"
    elif any(task_name.lower().startswith(t) for t in HF_TASKS):
        return "huggingface"
    else:
        return "lm_eval"
