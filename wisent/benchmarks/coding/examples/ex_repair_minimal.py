# Minimal example of repair loop using wisent-guard's safe-docker coding sandbox.
from __future__ import annotations
from typing import Dict
from wisent.benchmarks.coding.safe_docker.core.runtime import DockerSandboxExecutor
from wisent.benchmarks.coding.safe_docker.recipes import RECIPE_REGISTRY

def run_once(lang: str, files: Dict[str,str]):
    job = RECIPE_REGISTRY[lang].make_job(time_limit_s=6)
    return DockerSandboxExecutor(image="coding/sandbox:polyglot-1.0").run(files, job)

def feedback(res) -> str:
    if res.status == "timeout": return f"Timeout after {res.elapsed:.2f}s."
    return ("Compilation failed:\n" if res.status=="compile_error" else "Tests failed:\n") + (res.stdout or "") + ("\n"+res.stderr if res.stderr else "")

def repair(lang: str, f: Dict[str,str], fb: str) -> Dict[str,str]:
    if lang == "python":
        return {**f, "solution.py": f["solution.py"].replace("a - b", "a + b")}
    if lang == "cpp":
        return {**f, "solution.cpp": f["solution.cpp"].replace("a-b", "a+b")}
    return {**f, "Solution.java": f["Solution.java"].replace("a-b", "a+b")}

cases = {
    "python": {
        "solution.py": "def add(a,b): return a - b  # BUG",
        "tests.py":    "from solution import add\n"
                       "def test_ok(): assert add(1,2)==3\n"
    },
    "cpp": {
        "solution.cpp":  "int add(int a,int b){return a-b;}",
        "test_main.cpp": "#include <cassert>\nint add(int,int);\nint main(){assert(add(1,2)==3);return 0;}\n",
    },
    "java": {
        "Solution.java": "public class Solution{public static int add(int a,int b){return a-b;}}",
        "MainTest.java": "public class MainTest{public static void main(String[]a){"
                         "if(Solution.add(1,2)!=3)throw new RuntimeException(\"f\");}}",
    },
}

for lang, files in cases.items():
    print(f"\n== {lang.upper()} initial run ==")
    r0 = run_once(lang, files)
    print(r0.status, r0.stdout, r0.stderr)
    if r0.status == "ok":
        print("Pass@1-after-repair: True"); continue
    fb = feedback(lang, r0)
    files2 = repair(lang, files, fb)
    print(f"== {lang.upper()} repaired run ==")
    r1 = run_once(lang, files2)
    print(r1.status, r1.stdout, r1.stderr)
    print("Pass@1-after-repair:", r1.status == "ok")
