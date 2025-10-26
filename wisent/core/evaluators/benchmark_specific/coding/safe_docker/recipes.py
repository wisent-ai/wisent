from __future__ import annotations
from typing import Dict
from wisent.core.evaluators.benchmark_specific.coding.safe_docker.core.atoms import Job, LanguageRecipe

class PythonRecipe(LanguageRecipe):
    """
     Recipe for running Python code in a sandboxed environment.
    """
    language = "python"
    def make_job(self, **options) -> Job:
        run = ["pytest", "-q", "--maxfail=1", "--tb=short", "-rA", "tests.py"]
        return Job(language="python", compile_argv=None, run_argv=run,
                   cpu_limit_s=options.get("cpu_limit_s",3),
                   wall_timeout_s=options.get("time_limit_s",8),
                   mem_limit_mb=options.get("mem_limit_mb",768))

class CppRecipe(LanguageRecipe):
    language = "cpp"
    def make_job(self, **options) -> Job:
        std = options.get("cxx_std", "c++17")
        compile_cmd = ["bash","-lc", f"g++ -std={std} -O2 -pipe -o program solution.cpp test_main.cpp"]
        run_cmd = ["bash","-lc","./program"]
        return Job(language="cpp", compile_argv=compile_cmd, run_argv=run_cmd,
                   cpu_limit_s=options.get("cpu_limit_s",3),
                   wall_timeout_s=options.get("time_limit_s",8),
                   mem_limit_mb=options.get("mem_limit_mb",768))

class JavaRecipe:
    language = "java"
    def make_job(self, **options) -> Job:
        main = options.get("java_main", "MainTest")

        java_opts = options.get(
            "java_opts",
            "-Xms32m -Xmx256m -Xss512k "
            "-XX:CompressedClassSpaceSize=64m "
            "-XX:MaxMetaspaceSize=128m "
            "-XX:ReservedCodeCacheSize=64m "
            "-XX:MaxDirectMemorySize=64m "
            "-XX:+UseSerialGC -XX:+ExitOnOutOfMemoryError"
        )

        compile_cmd = ["bash", "-lc", "javac *.java"]
        run_cmd = ["bash", "-lc", f"java {java_opts} {main}"]

        return Job(
            language="java",
            compile_argv=compile_cmd,
            run_argv=run_cmd,
            cpu_limit_s=options.get("cpu_limit_s", 3),
            wall_timeout_s=options.get("time_limit_s", 8),
            mem_limit_mb=options.get("mem_limit_mb", 768),
        )


RECIPE_REGISTRY = {
    "python": PythonRecipe(),
    "cpp": CppRecipe(),
    "java": JavaRecipe(),
}