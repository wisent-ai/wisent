from wisent_guard.benchmarks.coding.safe_docker.core.runtime import DockerSandboxExecutor
from wisent_guard.benchmarks.coding.safe_docker.recipes import RECIPE_REGISTRY

# We need to load model
from wisent_guard.core.models.wisent_model import WisentModel
PATH_MODEL = "Llama-2-7b-hf"  # add your model path here
model = WisentModel(
    model_name=PATH_MODEL,
    layers={},
    device="cuda"  
)

# we need to define TaskSchema
from wisent_guard.benchmarks.coding.output_sanitizer.core.atoms import TaskSchema
schema = TaskSchema(
    language="python",
    file_name="solution.py",
    entry_point="add",
    prefer_rename=False,
    allow_wrapper=True
)
# we need to load sanitizer
from wisent_guard.benchmarks.coding.output_sanitizer.python_sanitizer import PythonStandardizer
sanitizer = PythonStandardizer()

# We need to define a programming problem (problem_prompt)

problem_prompt = """
### Problem
Write a Python function `add(a, b)` that returns the sum of two numbers.
Your solution must be in the format:
```python
def add(a, b):
    # your code here
```
DO NOT include any test code or main function.
"""
# generate solution
response = model.generate(
    inputs=[[{"role": "user", "content": problem_prompt}]],
    max_new_tokens=200,
    temperature=0.01,
    top_p=0.9,
    use_steering=False  
)

# we need to extract the code from the response (stronger models may return more structured output)
solution_code = response[0].split("function.assistant")[1].strip()

print("Generated solution code:\n", solution_code)

# we need to sanitize the code
normalized = sanitizer.normalize(solution_code, schema)
print("Sanitization notes:\n", normalized.notes)
if not normalized.ok:
    print("Sanitization failed, cannot proceed to execution.")
    exit(1)

solution_code = normalized.files["solution.py"]

# Extract the code from the response
files = {
    "solution.py": solution_code,
    "tests.py":    "from solution import add\n"
                   "def test_ok(): assert add(2,2)==4\n"
}
job = RECIPE_REGISTRY["python"].make_job(time_limit_s=6, cpu_limit_s=3, mem_limit_mb=256)
res = DockerSandboxExecutor(image="coding/sandbox:polyglot-1.0").run(files, job)
print(res)
