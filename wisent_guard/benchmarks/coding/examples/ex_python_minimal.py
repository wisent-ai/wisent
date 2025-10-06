from wisent_guard.benchmarks.coding.safe_docker.core.runtime import DockerSandboxExecutor
from wisent_guard.benchmarks.coding.safe_docker.recipes import RECIPE_REGISTRY

files = {
    "solution.py": "def add(a,b): return a+b",
    "tests.py":    "from solution import add\n"
                   "def test_ok(): assert add(2,2)==4\n"
}
job = RECIPE_REGISTRY["python"].make_job(files, time_limit_s=6, cpu_limit_s=3, mem_limit_mb=256)
res = DockerSandboxExecutor(image="coding/sandbox:polyglot-1.0").run(files, job)
print(res)
