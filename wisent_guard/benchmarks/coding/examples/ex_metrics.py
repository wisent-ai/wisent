from wisent_guard.benchmarks.coding.providers.livecodebench.provider import LiveCodeBenchProvider
from wisent_guard.benchmarks.coding.metrics.evaluator import CodingEvaluator, EvaluatorConfig
from wisent_guard.benchmarks.coding.metrics.passk import PassAtK
from typing import Dict
from wisent_guard.benchmarks.coding.providers.core.atoms import CodingTask

def model_stub(task: CodingTask) -> Dict[str,str]:
    # leave the provider's files as-is (some contain a bug to trigger repair)
    return {k: v for k, v in task.files.items() if "solution" in k.lower() or "solution" in k}

def simple_repair(lang: str, prev_files: Dict[str,str], feedback: str) -> Dict[str,str]:
    # tiny heuristic just for demo
    if lang == "python": return {"solution.py": prev_files["solution.py"].replace("a - b","a + b")}
    if lang == "cpp":    return {"solution.cpp": prev_files["solution.cpp"].replace("a-b","a+b")}
    return {"Solution.java": prev_files["Solution.java"].replace("a-b","a+b")}

for lang in ("python","cpp","java"):
    provider = LiveCodeBenchProvider(language=lang)
    evaluator = CodingEvaluator(provider, model_fn=model_stub, repair_fn=simple_repair,
                                cfg=EvaluatorConfig(image="coding/sandbox:polyglot-1.0", self_repair=True))
    outcomes = list(evaluator.evaluate())
    score = PassAtK(1).compute(outcomes)
    print(f"{provider.name}[{lang}] Pass@1-after-repair = {score:.3f}")
