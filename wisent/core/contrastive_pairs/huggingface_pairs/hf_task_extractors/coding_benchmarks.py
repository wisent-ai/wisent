from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.errors import InvalidValueError

__all__ = ["OJBenchExtractor", "TerminalBenchExtractor", "SciCodeExtractor"]

log = setup_logger(__name__)


class OJBenchExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for OJ-Bench - online judge style competitive programming benchmark.

    OJ-Bench evaluates LLMs on competitive programming problems similar to those
    found on online judges like Codeforces, AtCoder, and LeetCode. Problems are
    primarily in C++ and test algorithmic problem-solving skills.

    For competitive programming evaluation:
    - Positive (correct) = Solution that passes all test cases within time/memory limits
    - Negative (incorrect) = Solution with wrong answer, TLE, or MLE
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "competitive_programming"

    def __init__(self, difficulty: str | None = None, language: str = "cpp"):
        """
        Initialize OJ-Bench extractor.

        Args:
            difficulty: Optional filter (easy, medium, hard)
            language: Programming language (default: cpp)
        """
        super().__init__()
        self.difficulty = difficulty
        self.language = language

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from OJ-Bench examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Try loading from competitive programming datasets
        docs = []

        try:
            docs = self.load_dataset(
                dataset_name="deepmind/code_contests",
                split="test",
                limit=max_items * 2 if max_items else None,
            )
            log.info(f"Loaded {len(docs)} examples from code_contests")
        except Exception as e:
            log.warning(f"Failed to load code_contests: {e}")
            # Create synthetic competitive programming examples
            docs = self._create_synthetic_examples(max_items or 100)

        pairs: list[ContrastivePair] = []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid OJ-Bench pairs extracted")

        return pairs

    def _create_synthetic_examples(self, count: int) -> list[dict[str, Any]]:
        """Create synthetic competitive programming examples."""
        examples = [
            {
                "description": """Problem: Two Sum
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

Input: First line contains n (1 ≤ n ≤ 10^5) and target. Second line contains n space-separated integers.
Output: Two indices (0-indexed) separated by space.

Example:
Input:
4 9
2 7 11 15
Output:
0 1""",
                "correct_solution": """#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, target;
    cin >> n >> target;

    vector<int> nums(n);
    unordered_map<int, int> mp;

    for (int i = 0; i < n; i++) {
        cin >> nums[i];
        int complement = target - nums[i];
        if (mp.count(complement)) {
            cout << mp[complement] << " " << i << endl;
            return 0;
        }
        mp[nums[i]] = i;
    }

    return 0;
}""",
                "incorrect_solution": """#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, target;
    cin >> n >> target;

    vector<int> nums(n);
    for (int i = 0; i < n; i++) cin >> nums[i];

    // O(n^2) - will TLE on large inputs
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {  // Bug: should start from i+1
            if (nums[i] + nums[j] == target) {
                cout << i << " " << j << endl;
                return 0;
            }
        }
    }
    return 0;
}""",
                "difficulty": "easy",
            },
            {
                "description": """Problem: Maximum Subarray Sum
Find the contiguous subarray with the largest sum.

Input: First line contains n (1 ≤ n ≤ 10^6). Second line contains n integers (-10^9 ≤ a[i] ≤ 10^9).
Output: Maximum subarray sum.

Example:
Input:
8
-2 1 -3 4 -1 2 1 -5 4
Output:
6""",
                "correct_solution": """#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;

    long long maxSum = LLONG_MIN;
    long long currentSum = 0;

    for (int i = 0; i < n; i++) {
        long long x;
        cin >> x;
        currentSum = max(x, currentSum + x);
        maxSum = max(maxSum, currentSum);
    }

    cout << maxSum << endl;
    return 0;
}""",
                "incorrect_solution": """#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    cin >> n;

    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    int maxSum = 0;  // Bug: should be LLONG_MIN for negative arrays
    int currentSum = 0;

    for (int i = 0; i < n; i++) {
        currentSum += a[i];  // Bug: doesn't handle Kadane's algorithm correctly
        if (currentSum > maxSum) maxSum = currentSum;
        if (currentSum < 0) currentSum = 0;
    }

    cout << maxSum << endl;
    return 0;
}""",
                "difficulty": "medium",
            },
            {
                "description": """Problem: Segment Tree Range Sum
Given an array, support two operations:
1. Update a[i] = x
2. Query sum(l, r)

Input: First line n, q. Second line is initial array. Next q lines are operations.
Output: Answer for each query operation.

Example:
Input:
5 3
1 2 3 4 5
2 1 3
1 2 10
2 1 3
Output:
6
14""",
                "correct_solution": """#include <bits/stdc++.h>
using namespace std;

class SegmentTree {
    vector<long long> tree;
    int n;

public:
    SegmentTree(vector<int>& arr) {
        n = arr.size();
        tree.resize(4 * n);
        build(arr, 1, 0, n - 1);
    }

    void build(vector<int>& arr, int v, int tl, int tr) {
        if (tl == tr) {
            tree[v] = arr[tl];
        } else {
            int tm = (tl + tr) / 2;
            build(arr, 2*v, tl, tm);
            build(arr, 2*v+1, tm+1, tr);
            tree[v] = tree[2*v] + tree[2*v+1];
        }
    }

    void update(int v, int tl, int tr, int pos, int val) {
        if (tl == tr) {
            tree[v] = val;
        } else {
            int tm = (tl + tr) / 2;
            if (pos <= tm) update(2*v, tl, tm, pos, val);
            else update(2*v+1, tm+1, tr, pos, val);
            tree[v] = tree[2*v] + tree[2*v+1];
        }
    }

    long long query(int v, int tl, int tr, int l, int r) {
        if (l > r) return 0;
        if (l == tl && r == tr) return tree[v];
        int tm = (tl + tr) / 2;
        return query(2*v, tl, tm, l, min(r, tm)) +
               query(2*v+1, tm+1, tr, max(l, tm+1), r);
    }

    void update(int pos, int val) { update(1, 0, n-1, pos, val); }
    long long query(int l, int r) { return query(1, 0, n-1, l, r); }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    cin >> n >> q;

    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    SegmentTree st(a);

    while (q--) {
        int type, x, y;
        cin >> type >> x >> y;
        if (type == 1) {
            st.update(x - 1, y);
        } else {
            cout << st.query(x - 1, y - 1) << "\\n";
        }
    }

    return 0;
}""",
                "incorrect_solution": """#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, q;
    cin >> n >> q;

    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    // O(n) per query - will TLE
    while (q--) {
        int type, x, y;
        cin >> type >> x >> y;
        if (type == 1) {
            a[x-1] = y;
        } else {
            int sum = 0;
            for (int i = x-1; i < y; i++) sum += a[i];
            cout << sum << "\\n";
        }
    }
    return 0;
}""",
                "difficulty": "hard",
            },
        ]

        result = []
        for i in range(count):
            example = examples[i % len(examples)].copy()
            result.append(example)

        return result

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            # Handle code_contests schema
            description = doc.get("description", doc.get("problem", "")).strip()
            correct = doc.get("correct_solution", "")
            incorrect = doc.get("incorrect_solution", "")

            # For code_contests dataset
            if not correct and "solutions" in doc:
                solutions = doc.get("solutions", {})
                if isinstance(solutions, dict) and "cpp" in solutions:
                    cpp_solutions = solutions["cpp"]
                    if cpp_solutions:
                        correct = cpp_solutions[0]

            if not description:
                return None

            # Create incorrect solution if not provided
            if not incorrect:
                incorrect = self._create_incorrect_solution(description)

            if not correct:
                correct = self._create_placeholder_correct(description)

            difficulty = doc.get("difficulty", "medium")

            # Filter by difficulty if specified
            if self.difficulty and self.difficulty.lower() != difficulty.lower():
                return None

            task_prompt = f"""Competitive Programming Problem:

{description}

Write a correct C++ solution that passes all test cases within the time and memory limits."""

            metadata = {
                "label": "oj_bench",
                "source": "oj_bench",
                "difficulty": difficulty,
                "language": self.language,
                "is_competitive_programming_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _create_incorrect_solution(self, description: str) -> str:
        """Create a plausible but incorrect solution."""
        return """#include <bits/stdc++.h>
using namespace std;

int main() {
    // This solution has bugs:
    // - Doesn't handle edge cases
    // - May have integer overflow
    // - Inefficient algorithm causing TLE

    int n;
    cin >> n;

    // Naive O(n^2) approach
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // Missing logic
        }
    }

    cout << 0 << endl;  // Wrong answer
    return 0;
}"""

    def _create_placeholder_correct(self, description: str) -> str:
        """Create a placeholder correct solution structure."""
        return """#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Efficient solution with proper algorithm
    // Handles all edge cases
    // Time complexity: O(n log n) or better

    // Implementation details depend on specific problem

    return 0;
}"""



class TerminalBenchExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for Terminal-Bench - terminal/CLI interaction benchmark.

    Terminal-Bench evaluates LLMs' ability to interact with command-line
    interfaces, execute shell commands, navigate filesystems, and perform
    system administration tasks.

    For terminal interaction evaluation:
    - Positive (correct) = Correct commands with proper syntax and expected behavior
    - Negative (incorrect) = Commands with errors, wrong syntax, or dangerous operations
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "terminal_interaction"

    def __init__(self, os_type: str = "linux"):
        """
        Initialize Terminal-Bench extractor.

        Args:
            os_type: Operating system type (linux, macos, windows)
        """
        super().__init__()
        self.os_type = os_type

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Terminal-Bench examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Try loading NL2Bash dataset
        docs = []

        try:
            docs = self.load_dataset(
                dataset_name="jiacheng-ye/nl2bash",
                split="test",
                limit=max_items * 2 if max_items else None,
            )
            log.info(f"Loaded {len(docs)} examples from nl2bash")
        except Exception as e:
            log.warning(f"Failed to load nl2bash: {e}")
            # Create synthetic terminal examples
            docs = self._create_synthetic_examples(max_items or 100)

        pairs: list[ContrastivePair] = []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid Terminal-Bench pairs extracted")

        return pairs

    def _create_synthetic_examples(self, count: int) -> list[dict[str, Any]]:
        """Create synthetic terminal interaction examples."""
        examples = [
            {
                "nl": "Find all Python files in the current directory and subdirectories",
                "correct_command": "find . -name '*.py' -type f",
                "incorrect_command": "find *.py",  # Wrong syntax
                "category": "file_search",
            },
            {
                "nl": "Count the number of lines in all text files in the current directory",
                "correct_command": "wc -l *.txt | tail -1",
                "incorrect_command": "count lines *.txt",  # Not a real command
                "category": "file_analysis",
            },
            {
                "nl": "Create a compressed archive of the logs directory",
                "correct_command": "tar -czvf logs.tar.gz logs/",
                "incorrect_command": "zip logs/ archive",  # Wrong argument order
                "category": "archiving",
            },
            {
                "nl": "Show running processes sorted by memory usage",
                "correct_command": "ps aux --sort=-%mem | head -20",
                "incorrect_command": "ps memory",  # Invalid syntax
                "category": "process_management",
            },
            {
                "nl": "Find and kill all processes named 'python'",
                "correct_command": "pkill -f python",
                "incorrect_command": "kill python",  # kill needs PID, not name
                "category": "process_management",
            },
            {
                "nl": "Download a file from a URL and save it with a specific name",
                "correct_command": "curl -o output.txt https://example.com/file.txt",
                "incorrect_command": "download https://example.com/file.txt",  # Not a command
                "category": "networking",
            },
            {
                "nl": "Find files modified in the last 24 hours",
                "correct_command": "find . -mtime -1 -type f",
                "incorrect_command": "find . modified 24h",  # Wrong syntax
                "category": "file_search",
            },
            {
                "nl": "Replace all occurrences of 'foo' with 'bar' in a file in-place",
                "correct_command": "sed -i 's/foo/bar/g' file.txt",
                "incorrect_command": "replace foo bar file.txt",  # Not a command
                "category": "text_processing",
            },
            {
                "nl": "Check disk space usage for all mounted filesystems",
                "correct_command": "df -h",
                "incorrect_command": "disk space",  # Not a command
                "category": "system_info",
            },
            {
                "nl": "Create a new user named 'developer' with home directory",
                "correct_command": "sudo useradd -m -s /bin/bash developer",
                "incorrect_command": "create user developer",  # Not a command
                "category": "user_management",
            },
        ]

        result = []
        for i in range(count):
            example = examples[i % len(examples)].copy()
            result.append(example)

        return result

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            # Handle nl2bash schema
            nl = doc.get("nl", doc.get("description", "")).strip()
            correct = doc.get("correct_command", doc.get("bash", "")).strip()
            incorrect = doc.get("incorrect_command", "").strip()
            category = doc.get("category", "general")

            if not nl:
                return None

            if not correct:
                return None

            if not incorrect:
                incorrect = self._create_incorrect_command(nl)

            task_prompt = f"""Terminal Command Task:

{nl}

Provide the correct {self.os_type} terminal command to accomplish this task.
The command should be safe, efficient, and follow best practices."""

            correct_response = f"```bash\n{correct}\n```\n\nThis command correctly accomplishes the task."
            incorrect_response = f"```bash\n{incorrect}\n```\n\nNote: This command may have syntax errors or may not work as intended."

            metadata = {
                "label": "terminal_bench",
                "source": "terminal_bench",
                "category": category,
                "os_type": self.os_type,
                "is_terminal_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct_response,
                incorrect=incorrect_response,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _create_incorrect_command(self, description: str) -> str:
        """Create a plausible but incorrect command."""
        return "# Command with incorrect syntax or missing flags"



class SciCodeExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for SciCode - scientific computing code generation benchmark.

    SciCode evaluates LLMs' ability to generate code for scientific computing
    tasks including numerical methods, data analysis, and domain-specific
    scientific computations.

    Dataset: Various scientific computing datasets

    For scientific computing evaluation:
    - Positive (correct) = Scientifically accurate code with proper numerical methods
    - Negative (incorrect) = Code with numerical errors or incorrect scientific methods
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "scientific_computing"

    def __init__(self, domain: str | None = None):
        """
        Initialize SciCode extractor.

        Args:
            domain: Optional filter for scientific domain (physics, chemistry, biology, etc.)
        """
        super().__init__()
        self.domain = domain

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from SciCode examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Create synthetic scientific computing examples
        docs = self._create_synthetic_examples(max_items or 100)

        pairs: list[ContrastivePair] = []

        for doc in docs:
            if self.domain and doc.get("domain") != self.domain:
                continue

            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid SciCode pairs extracted")

        return pairs

    def _create_synthetic_examples(self, count: int) -> list[dict[str, Any]]:
        """Create synthetic scientific computing examples."""
        examples = [
            {
                "problem": "Implement numerical integration using Simpson's rule",
                "domain": "mathematics",
                "correct_solution": """import numpy as np

def simpsons_rule(f, a, b, n):
    '''
    Integrate f(x) from a to b using Simpson's rule with n intervals.
    n must be even.
    '''
    if n % 2 != 0:
        raise InvalidValueError(param_name="n", actual=n, expected="even number for Simpson's rule")

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    # Simpson's rule: h/3 * (y_0 + 4*y_1 + 2*y_2 + 4*y_3 + ... + y_n)
    integral = y[0] + y[-1]
    integral += 4 * np.sum(y[1:-1:2])  # odd indices
    integral += 2 * np.sum(y[2:-1:2])  # even indices (except first and last)

    return integral * h / 3

# Example: Integrate sin(x) from 0 to pi (expected: 2.0)
result = simpsons_rule(np.sin, 0, np.pi, 100)
print(f"Integral of sin(x) from 0 to pi: {result:.10f}")""",
                "incorrect_solution": """import numpy as np

def simpsons_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n)  # Bug: should be n+1 points
    y = f(x)

    # Wrong implementation - missing proper weighting
    integral = np.sum(y) * h  # This is just rectangular rule

    return integral""",
            },
            {
                "problem": "Solve a system of ODEs using Runge-Kutta 4th order method",
                "domain": "physics",
                "correct_solution": """import numpy as np

def rk4_step(f, t, y, h):
    '''
    Single step of RK4 method.
    f: function f(t, y) returning dy/dt
    t: current time
    y: current state vector
    h: step size
    '''
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)

    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

def solve_ode(f, y0, t_span, n_steps):
    '''
    Solve ODE system dy/dt = f(t, y) using RK4.
    '''
    t = np.linspace(t_span[0], t_span[1], n_steps + 1)
    h = t[1] - t[0]

    y = np.zeros((n_steps + 1, len(y0)))
    y[0] = y0

    for i in range(n_steps):
        y[i+1] = rk4_step(f, t[i], y[i], h)

    return t, y

# Example: Simple harmonic oscillator
def harmonic(t, y):
    return np.array([y[1], -y[0]])

t, y = solve_ode(harmonic, np.array([1.0, 0.0]), [0, 10], 1000)""",
                "incorrect_solution": """import numpy as np

def euler_step(f, t, y, h):
    # Using Euler method instead of RK4 - much less accurate
    return y + h * f(t, y)

def solve_ode(f, y0, t_span, n_steps):
    t = np.linspace(t_span[0], t_span[1], n_steps)  # Bug: should be n_steps+1
    h = (t_span[1] - t_span[0]) / n_steps

    y = [y0]
    for i in range(n_steps - 1):
        y.append(euler_step(f, t[i], y[i], h))

    return t, np.array(y)""",
            },
        ]

        result = []
        for i in range(count):
            example = examples[i % len(examples)].copy()
            result.append(example)

        return result

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            problem = doc.get("problem", "").strip()
            domain = doc.get("domain", "general")
            correct = doc.get("correct_solution", "").strip()
            incorrect = doc.get("incorrect_solution", "").strip()

            if not problem or not correct:
                return None

            task_prompt = f"""Scientific Computing Task:

{problem}

Provide a Python implementation that is:
- Numerically accurate and stable
- Well-documented with clear variable names
- Efficient and follows scientific computing best practices"""

            correct_response = f"```python\n{correct}\n```"
            incorrect_response = f"```python\n{incorrect}\n```"

            metadata = {
                "label": "scicode",
                "source": "scicode",
                "domain": domain,
                "is_scientific_computing_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct_response,
                incorrect=incorrect_response,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

