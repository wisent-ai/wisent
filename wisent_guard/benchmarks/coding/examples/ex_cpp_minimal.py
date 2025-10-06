from wisent_guard.benchmarks.coding.safe_docker.core.runtime import DockerSandboxExecutor
from wisent_guard.benchmarks.coding.safe_docker.recipes import RECIPE_REGISTRY

files = {
    "solution.cpp":  "int add(int a,int b){return a+b;}",
    "test_main.cpp": "#include <cassert>\nint add(int,int);\nint main(){assert(add(1,2)==3);return 0;}\n",
}
job = RECIPE_REGISTRY["cpp"].make_job(files, cxx_std="c++17", time_limit_s=6)
res = DockerSandboxExecutor(image="coding/sandbox:polyglot-1.0").run(files, job)
print(res)
