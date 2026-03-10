"""Count tokens per message type in a Claude Code JSONL session transcript."""
import importlib.util
import json
import os
import sys
from collections import defaultdict

_CONST_PATH = os.path.join(
    os.path.dirname(__file__),
    "config_tools", "constants",
    "cannot_be_optimized", "_infrastructure.py",
)
_spec = importlib.util.spec_from_file_location(
    "_infrastructure", _CONST_PATH,
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CHARS_PER_TOKEN_ESTIMATE = _mod.CHARS_PER_TOKEN_ESTIMATE
COMBO_OFFSET = _mod.COMBO_OFFSET
RECURSION_INITIAL_DEPTH = _mod.RECURSION_INITIAL_DEPTH
REPORT_COL_COUNT_WIDTH = _mod.REPORT_COL_COUNT_WIDTH
REPORT_COL_NUMERIC_WIDTH = _mod.REPORT_COL_NUMERIC_WIDTH
REPORT_COL_TYPE_WIDTH = _mod.REPORT_COL_TYPE_WIDTH

JSONL_IDX = COMBO_OFFSET

_SEP_WIDTH = (
    REPORT_COL_TYPE_WIDTH
    + REPORT_COL_COUNT_WIDTH
    + REPORT_COL_NUMERIC_WIDTH
    + REPORT_COL_NUMERIC_WIDTH
    + COMBO_OFFSET + COMBO_OFFSET + COMBO_OFFSET
)


def _extract_text_length(obj):
    """Extract total character length of text content from a message."""
    total = RECURSION_INITIAL_DEPTH
    msg = obj.get("message", {})
    content = msg.get("content", "")

    if isinstance(content, str):
        total += len(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                if item_type == "text":
                    total += len(item.get("text", ""))
                elif item_type == "tool_result":
                    sub_content = item.get("content", "")
                    if isinstance(sub_content, str):
                        total += len(sub_content)
                    elif isinstance(sub_content, list):
                        for sub in sub_content:
                            if isinstance(sub, dict):
                                total += len(
                                    sub.get("text", "")
                                )
                elif item_type == "tool_use":
                    inp = item.get("input", {})
                    total += len(json.dumps(inp))
            elif isinstance(item, str):
                total += len(item)

    return total


def _fmt_header():
    """Format the report header row."""
    tw = REPORT_COL_TYPE_WIDTH
    cw = REPORT_COL_COUNT_WIDTH
    nw = REPORT_COL_NUMERIC_WIDTH
    return (
        f"{'Type':<{tw}} {'Lines':>{cw}} "
        f"{'Chars':>{nw}} {'Est Tokens':>{nw}}"
    )


def _fmt_row(label, lines, chars, tokens):
    """Format a single report row."""
    tw = REPORT_COL_TYPE_WIDTH
    cw = REPORT_COL_COUNT_WIDTH
    nw = REPORT_COL_NUMERIC_WIDTH
    return (
        f"{label:<{tw}} {lines:>{cw}} "
        f"{chars:>{nw},} {tokens:>{nw},}"
    )


def count_tokens(jsonl_path):
    """Count tokens per message type in a JSONL session."""
    with open(jsonl_path) as f:
        lines = f.readlines()

    counts = defaultdict(lambda: {
        "lines": RECURSION_INITIAL_DEPTH,
        "chars": RECURSION_INITIAL_DEPTH,
    })

    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_type = obj.get("type", "unknown")
        text_len = _extract_text_length(obj)

        counts[msg_type]["lines"] += COMBO_OFFSET
        counts[msg_type]["chars"] += text_len

    print()
    print(_fmt_header())
    print("-" * _SEP_WIDTH)

    total_chars = RECURSION_INITIAL_DEPTH
    total_tokens = RECURSION_INITIAL_DEPTH
    total_lines = RECURSION_INITIAL_DEPTH

    for msg_type in sorted(
        counts.keys(),
        key=lambda k: counts[k]["chars"],
        reverse=True,
    ):
        data = counts[msg_type]
        est_tokens = int(
            data["chars"] / CHARS_PER_TOKEN_ESTIMATE
        )
        total_chars += data["chars"]
        total_tokens += est_tokens
        total_lines += data["lines"]
        print(_fmt_row(
            msg_type, data["lines"],
            data["chars"], est_tokens,
        ))

    print("-" * _SEP_WIDTH)
    print(_fmt_row(
        "TOTAL", total_lines, total_chars, total_tokens,
    ))
    print()


if __name__ == "__main__":
    count_tokens(sys.argv[JSONL_IDX])
