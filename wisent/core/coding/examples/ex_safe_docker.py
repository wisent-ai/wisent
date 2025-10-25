from __future__ import annotations
from typing import Dict
from wisent.benchmarks.coding.providers.livecodebench.provider import LiveCodeBenchProvider
from wisent.benchmarks.coding.metrics.evaluator import CodingEvaluator, EvaluatorConfig
from wisent.benchmarks.coding.metrics.passk import PassAtK
from wisent.benchmarks.coding.providers.core.atoms import CodingTask

def model_stub(task: CodingTask) -> Dict[str,str]:
    if task.language == "python":
        return {"solution.py": task.files.get("solution.py", "def add(a,b): return a-b")}
    elif task.language == "cpp":
        return {"solution.cpp": task.files.get("solution.cpp", "int add(int a,int b){return a-b;}")}
    else:
        return {"Solution.java": task.files.get("Solution.java","public class Solution{public static int add(int a,int b){return a-b;}}")}

def simple_repair(lang: str, prev_files: Dict[str,str], feedback: str) -> Dict[str,str]:
    if lang == "python":
        fixed = prev_files["solution.py"].replace("a - b", "a + b")
        return {"solution.py": fixed}
    if lang == "cpp":
        fixed = prev_files["solution.cpp"].replace("a-b", "a+b")
        return {"solution.cpp": fixed}
    fixed = prev_files["Solution.java"].replace("a-b", "a+b")
    return {"Solution.java": fixed}

def run_all():
    cfg = EvaluatorConfig(image="coding/sandbox:polyglot-1.0", self_repair=True)

    providers = [
        LiveCodeBenchProvider(language="python"),
        LiveCodeBenchProvider(language="cpp"),
        LiveCodeBenchProvider(language="java"),
    ]

    for p in providers:
        print(f"\n=== Evaluating provider={p.name} language={getattr(p,'language','?')} ===")
        evaluator = CodingEvaluator(provider=p, model_fn=model_stub, repair_fn=simple_repair, cfg=cfg)
        outcomes = list(evaluator.evaluate())
        metric = PassAtK(k=1)
        score = metric.compute(outcomes)
        for o in outcomes:
            print(o)
        print(f"Pass@1 after repair: {score:.3f}")

if __name__ == "__main__":
    run_all()