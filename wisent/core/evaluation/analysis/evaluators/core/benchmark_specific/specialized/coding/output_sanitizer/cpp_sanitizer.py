from __future__ import annotations
import re
from typing import List
from wisent.core.evaluators.benchmark_specific.coding.output_sanitizer.core.atoms import TaskSchema, NormalizeResult, CodeStandardizer
from wisent.core.evaluators.benchmark_specific.coding.output_sanitizer.utils import extract_code_block, normalize_whitespace

FUNC_RE = re.compile(r"^\s*(?:template<[^>]+>\s*)?(?:[\w:\s*&<>,]+)\s+(\w+)\s*\(", re.MULTILINE)
CLASS_RE = re.compile(r"^\s*class\s+(\w+)\s*[{:]", re.MULTILINE)

class CppStandardizer(CodeStandardizer):
    language = "cpp"

    def normalize(self, raw: str, schema: TaskSchema) -> NormalizeResult:
        notes: List[str] = []
        code = normalize_whitespace(extract_code_block(raw, prefer_langs=("cpp","c++","cc","c")))
        code = re.sub(r"^```.*?\n|\n```$", "", code, flags=re.DOTALL)

        if re.search(rf"\b{re.escape(schema.entry_point)}\s*\(", code):
            notes.append(f"found function '{schema.entry_point}'")
            return NormalizeResult(files={schema.file_name: code}, notes="\n".join(notes), ok=True)

        classes = CLASS_RE.findall(code)
        for cls in classes:
            if re.search(rf"\b{re.escape(cls)}\s*::\s*{re.escape(schema.entry_point)}\s*\(", code) or \
               re.search(rf"class\s+{re.escape(cls)}.*?\b{re.escape(schema.entry_point)}\s*\(", code, flags=re.S):
                notes.append(f"found {cls}::{schema.entry_point}; adding free-function shim")
                shim = (
                    f"\n\ntemplate <typename... Args>\n"
                    f"auto {schema.entry_point}(Args&&... args)\n"
                    f"    -> decltype({cls}().{schema.entry_point}(std::forward<Args>(args)...)) {{\n"
                    f"    return {cls}().{schema.entry_point}(std::forward<Args>(args)...);\n"
                    f"}}\n"
                )
                if "#include <utility>" not in code:
                    code = "#include <utility>\n" + code
                return NormalizeResult(files={schema.file_name: code + shim}, notes="\n".join(notes), ok=True)

        candidates = [m.group(1) for m in FUNC_RE.finditer(code)]
        if schema.prefer_rename and len(candidates) == 1:
            old = candidates[0]
            if old != schema.entry_point:
                notes.append(f"renaming free function '{old}' -> '{schema.entry_point}'")
                code2 = re.sub(rf"(\b){re.escape(old)}(\s*\()", rf"\1{schema.entry_point}\2", code)
                return NormalizeResult(files={schema.file_name: code2}, notes="\n".join(notes), ok=True)

        if candidates:
            target = candidates[0]
            if target != schema.entry_point:
                notes.append(f"adding forwarding wrapper {schema.entry_point} -> {target}")
                shim = (
                    f"\n\ntemplate <typename... Args>\n"
                    f"auto {schema.entry_point}(Args&&... args)\n"
                    f"    -> decltype({target}(std::forward<Args>(args)...)) {{\n"
                    f"    return {target}(std::forward<Args>(args)...);\n"
                    f"}}\n"
                )
                if "#include <utility>" not in code:
                    code = "#include <utility>\n" + code
                return NormalizeResult(files={schema.file_name: code + shim}, notes="\n".join(notes), ok=True)

        notes.append("no obvious function; returned normalized source only")
        return NormalizeResult(files={schema.file_name: code}, notes="\n".join(notes), ok=False)
