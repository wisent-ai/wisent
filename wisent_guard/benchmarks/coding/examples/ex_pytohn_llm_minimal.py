from wisent_guard.benchmarks.coding.safe_docker.core.runtime import DockerSandboxExecutor
from wisent_guard.benchmarks.coding.safe_docker.recipes import RECIPE_REGISTRY

# We need to load model
from wisent_guard.core.models.wisent_model import WisentModel
PATH_MODEL = "meta-llama/Llama-3.2-1B-Instruct"  
model = WisentModel(
    model_name=PATH_MODEL,  # Example model name
    layers={},  # No steering vectors
    device="cuda"  # or "cpu"
)

# We need to define a programming problem (promplem_prompt)

problem_prompt = """
### Problem
Write a Python function `add(a, b)` that returns the sum of two numbers.
Your solution must be in the format:
```python
def add(a, b):
    # your code here
```
"""
# generate solution
response = model.generate(
    inputs=[[{"role": "user", "content": problem_prompt}]],
    max_new_tokens=200,
    temperature=0.01,
    top_p=0.9,
    use_steering=False  
)

solution_code = response[0]
print("Generated solution code:\n", solution_code)

# Extract the code from the response
files = {
    "solution.py": solution_code,
    "tests.py":    "from solution import add\n"
                   "def test_ok(): assert add(2,2)==4\n"
}
job = RECIPE_REGISTRY["python"].make_job(files, time_limit_s=6, cpu_limit_s=3, mem_limit_mb=256)
res = DockerSandboxExecutor(image="coding/sandbox:polyglot-1.0").run(files, job)
print(res)
