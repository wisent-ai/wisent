from wisent_guard.benchmarks.coding.output_sanitizer.core.atoms import TaskSchema
from wisent_guard.benchmarks.coding.output_sanitizer.python_sanitizer import PythonStandardizer
from wisent_guard.benchmarks.coding.output_sanitizer.cpp_sanitizer import CppStandardizer
from wisent_guard.benchmarks.coding.output_sanitizer.java_sanitizer import JavaStandardizer

def demo_python():
    messy = '''\
    Here is your code:

    ```python
    class Solution:
        def add(self, a, b):
            return a + b

    '''
    schema = TaskSchema(language="python", file_name="solution.py", entry_point="add", prefer_rename=False)
    out = PythonStandardizer().normalize(messy, schema)
    print("PY ok:", out.ok, "\nnotes:\n", out.notes, "\n---\n", out.files["solution.py"])

def demo_cpp():
    messy = '''\

#include <bits/stdc++.h>
using namespace std;
int sum(int a,int b){ return a + b; }

'''
    schema = TaskSchema(language="cpp", file_name="solution.cpp", entry_point="add", prefer_rename=True)
    out = CppStandardizer().normalize(messy, schema)
    print("CPP ok:", out.ok, "\nnotes:\n", out.notes, "\n---\n", out.files["solution.cpp"])

def demo_java():
    messy = '''\

class MyClass {
    int add(int a, int b){ return a + b; }
}

'''
    schema = TaskSchema(language="java", file_name="Solution.java", entry_point="add", java_class="Solution")
    out = JavaStandardizer().normalize(messy, schema)
    print("JAVA ok:", out.ok, "\nnotes:\n", out.notes, "\n---\n", out.files["Solution.java"])

if __name__ == "__main__":
    demo_python()
    demo_cpp()
    demo_java()