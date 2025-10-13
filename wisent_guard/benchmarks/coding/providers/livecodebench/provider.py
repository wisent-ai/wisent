# coding/providers/livecodebench/provider.py
from __future__ import annotations
from typing import Iterable
from ..core.atoms import CodingTask, Language

class LiveCodeBenchProvider:
    """
    Sketch adapter: load LiveCodeBench (code_generation_lite) and render tasks.
    Note: HF card states it's used for self-repair with test case feedback too.
    """
    name = "livecodebench"

    def __init__(self, language: Language = "python"):
        self.language = language

    def iter_tasks(self, split: str = "test") -> Iterable[CodingTask]:
        # placeholder: integrate HF datasets on your host and transform each row
        # according to `self.language` into {files} + options.
        # HF dataset card: "also used for self-repair using test case feedback".
        # https://huggingface.co/datasets/livecodebench/code_generation_lite
        # (Keep this stub lean; real impl will map test templates per language.)
        # Yield a toy one so examples work:
        if self.language == "python":
            yield CodingTask(
                language="python",
                files={
                    "solution.py": "def add(a,b): return a - b  # BUG",
                    "tests.py": "from solution import add\n"
                                "def test_ok(): assert add(1,2)==3\n"
                                "def test_neg(): assert add(-5,2)==-3\n"
                },
                options={}
            )
        elif self.language == "cpp":
            yield CodingTask(
                language="cpp",
                files={
                    "solution.cpp":"int add(int a,int b){return a-b;}", 
                    "test_main.cpp":"#include <cassert>\nint add(int,int);\nint main(){assert(add(1,2)==3);assert(add(-5,2)==-3);return 0;}"
                },
                options={"cxx_std":"c++17"}
            )
        else:  # java
            yield CodingTask(
                language="java",
                files={
                    "Solution.java":"public class Solution{public static int add(int a,int b){return a-b;}}",
                    "MainTest.java":"public class MainTest{public static void main(String[]a){"
                                    "if(Solution.add(1,2)!=3)throw new RuntimeException(\"f1\");"
                                    "if(Solution.add(-5,2)!=-3)throw new RuntimeException(\"f2\");}}"
                },
                options={"java_main":"MainTest"}
            )
