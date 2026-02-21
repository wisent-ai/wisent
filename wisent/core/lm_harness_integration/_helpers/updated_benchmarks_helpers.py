"""Extracted from updated_benchmarks.py - tail of CORE_BENCHMARKS dict."""


# Additional benchmark entries that extend CORE_BENCHMARKS
EXTRA_BENCHMARKS = {
    # Linguistic Understanding
    "blimp": {
        "task": "blimp",
        "tags": ["long context", "reasoning", "general knowledge"]
    },
    "unscramble": {
        "task": "unscramble",
        "tags": ["long context", "reasoning", "general knowledge"]
    },

    # Translation
    "wmt": {
        "task": "wmt",
        "tags": ["reasoning", "general knowledge", "science"]
    },

    # Comprehensive Suites
    "big_bench": {
        "task": "big_bench",
        "tags": ["reasoning", "general knowledge", "science"]
    },

    # Dialogue and Conversation
    "babi": {
        "task": "babi",
        "tags": ["reasoning", "general knowledge", "long context"]
    },
}
