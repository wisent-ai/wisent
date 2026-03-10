"""Extract readable conversation from JSONL session transcript."""
import json
import sys

from wisent.core.utils.config_tools.constants import (
    COMBO_OFFSET,
    RECURSION_INITIAL_DEPTH,
)

JSONL_IDX = COMBO_OFFSET
OUTPUT_IDX = COMBO_OFFSET + COMBO_OFFSET

JSONL_PATH = sys.argv[JSONL_IDX]
OUTPUT_PATH = sys.argv[OUTPUT_IDX]

with open(JSONL_PATH) as f:
    lines = f.readlines()

output_lines = []

for line in lines:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        continue

    msg_type = obj.get("type")

    if msg_type == "user":
        msg = obj.get("message", {})
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            output_lines.append(f"USER: {content.strip()}\n\n")
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text = item.get("text", "").strip()
                        if text:
                            output_lines.append(
                                f"USER: {text}\n\n"
                            )

    elif msg_type == "assistant":
        msg = obj.get("message", {})
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text = item.get("text", "").strip()
                        if text:
                            output_lines.append(
                                f"ASSISTANT: {text}\n\n"
                            )

with open(OUTPUT_PATH, "w") as f:
    f.writelines(output_lines)

print(
    f"Wrote {len(output_lines)} messages to {OUTPUT_PATH}"
)
