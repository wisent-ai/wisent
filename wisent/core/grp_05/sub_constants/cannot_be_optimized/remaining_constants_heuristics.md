# Decision Rules for Constant X

Given a constant X in `cannot_be_optimized`, apply these rules top-to-bottom. The first matching rule determines the action.

---

## Rule 1: Zero references outside definition → DELETE

Grep for X across the codebase excluding the `_fixed_*.py` definition file. If zero hits, X is dead code. Delete it.

---

## Rule 2: X depends on CPU count, RAM, or GPU memory → RUNTIME FUNCTION

If the correct value of X changes when the machine changes, X should not be a constant. Replace it with a function in `hardware.py` that computes the value from `detect_system_resources()`.

**Test:** "If I move this code from an A100 server to a Raspberry Pi, should X change?" If yes, make it a function.

Examples: batch sizes, memory caps, worker counts, Docker resource reservations, subprocess durations that scale with CPU speed.

---

## Rule 3: X depends on the loaded model → DERIVE FROM MODEL

If the correct value of X changes when the model changes (different context window, different hidden size, different number of layers), X should be computed from model config at load-time.

**Test:** "If I switch from Llama-3.2-1B to Qwen3-8B, should X change?" If yes, derive it from `model.config` or `tokenizer.model_max_length`.

Examples: tokenization max lengths, layer indices, attention head counts.

---

## Rule 4: X is a definition, not a choice → KEEP

If changing X would change the meaning of the concept it represents, keep it. This includes display/formatting constants (font sizes, marker sizes, separator widths, truncation lengths, DPI, alpha values, figure dimensions) and mathematical/semantic definitions (50% = majority, 2 = minimum for a pair, 1 = single). The half/double test does not apply because V/2 or 2*V would describe a different concept, not a worse value for the same concept.

---

## Rule 5: Nothing above matched → KEEP

If no rule above applies, X stays. The default action is to leave constants alone. Adding dynamic computation has a cost (complexity, testing, debugging) and should only happen when there is a concrete benefit.

---

## Rule 6: Half/Double Test — every constant must be justified

For every constant X with value V, consider: would X = V/2 or X = 2*V also be acceptable? If yes, the current value is arbitrary and must be explicitly justified.

Every constant in `cannot_be_optimized` must have exactly one of:

1. **A written justification** — a comment explaining why V and not V/2 or 2*V. The justification must reference something concrete: a spec, a measurement, a constraint, or a design decision with stated reasoning.

2. **An experiment** — a recorded test showing that V outperforms V/2 and 2*V on a relevant metric (accuracy, throughput, crash rate, visual quality, etc.). The experiment result should be referenced in a comment next to the constant.

If neither a justification nor an experiment exists for a constant, the system must raise an error at definition time. No constant may exist in `cannot_be_optimized` without one of these two backing it. If a new constant is added and the author does not provide a justification or experiment, it is rejected.
