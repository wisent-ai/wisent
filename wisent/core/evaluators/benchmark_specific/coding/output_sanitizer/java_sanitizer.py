# coding/llm_sanitizer/java_sanitizer.py
from __future__ import annotations
import re
from typing import List
from wisent.core.evaluators.benchmark_specific.coding.output_sanitizer.core.atoms import TaskSchema, NormalizeResult, CodeStandardizer
from wisent.core.evaluators.benchmark_specific.coding.output_sanitizer.utils import extract_code_block, normalize_whitespace

CLASS_RE = re.compile(r"\bclass\s+([A-Za-z_]\w*)")
METHOD_RE = re.compile(r"(public\s+static\s+[\w\<\>\[\]]+\s+)(\w+)\s*\(")

class JavaStandardizer(CodeStandardizer):
    language = "java"

    def normalize(self, raw: str, schema: TaskSchema) -> NormalizeResult:
        notes: List[str] = []
        code = normalize_whitespace(extract_code_block(raw, prefer_langs=("java")))
        code = re.sub(r"^```.*?\n|\n```$", "", code, flags=re.DOTALL)

        m = CLASS_RE.search(code)
        if m:
            found = m.group(1)
            if found != schema.java_class:
                notes.append(f"renaming class '{found}' -> '{schema.java_class}'")
                code = re.sub(rf"\bclass\s+{re.escape(found)}\b", f"class {schema.java_class}", code, count=1)

        if not CLASS_RE.search(code):
            notes.append(f"wrapping code in class {schema.java_class}")
            code = f"public class {schema.java_class} {{\n{indent(code)}\n}}\n"

        static_methods = list(METHOD_RE.finditer(code))
        if any(m.group(2) == schema.entry_point for m in static_methods):
            notes.append(f"found public static '{schema.entry_point}'")
            return NormalizeResult(files={schema.file_name: code}, notes="\n".join(notes), ok=True)

        if len(static_methods) == 1 and schema.prefer_rename:
            old = static_methods[0].group(2)
            if old != schema.entry_point:
                notes.append(f"renaming static method '{old}' -> '{schema.entry_point}'")
                code = re.sub(rf"(\bpublic\s+static\s+[\w\<\>\[\]]+\s+){re.escape(old)}(\s*\()",
                              rf"\1{schema.entry_point}\2", code, count=1)
                return NormalizeResult(files={schema.file_name: code}, notes="\n".join(notes), ok=True)

        if re.search(rf"\b{schema.entry_point}\s*\(", code):
            notes.append(f"adding static wrapper for instance method '{schema.entry_point}'")
            wrapper = (
                f"\n    public static <T> Object {schema.entry_point}(Object... args) {{\n"
                f"        {schema.java_class} _x = new {schema.java_class}();\n"
                f"        try {{\n"
                f"            // attempt reflective dispatch to instance method\n"
                f"            Class<?>[] types = new Class<?>[args.length];\n"
                f"            for (int i=0;i<args.length;i++) types[i] = args[i].getClass();\n"
                f"            return {schema.java_class}.class.getMethod(\"{schema.entry_point}\", types).invoke(_x, args);\n"
                f"        }} catch (Exception ex) {{ throw new RuntimeException(ex); }}\n"
                f"    }}\n"
            )
            code = re.sub(rf"(class\s+{schema.java_class}\s*{{)", r"\1" + wrapper, code, count=1)
            return NormalizeResult(files={schema.file_name: code}, notes="\n".join(notes), ok=True)

        notes.append("no suitable method; adding delegating static method to first public static or instance method via reflection")
        fallback = (
            f"\n    public static Object {schema.entry_point}(Object... args) {{\n"
            f"        try {{\n"
            f"            // try any public method first\n"
            f"            for (var m : {schema.java_class}.class.getMethods()) {{\n"
            f"                if (m.getName().equals(\"{schema.entry_point}\")) continue;\n"
            f"                try {{ return m.invoke(m.getParameterCount()==0? new {schema.java_class}(): new {schema.java_class}(), args); }}\n"
            f"                catch (Exception ignored) {{}}\n"
            f"            }}\n"
            f"        }} catch (Exception e) {{ throw new RuntimeException(e); }}\n"
            f"        throw new RuntimeException(\"No suitable method for entry point\");\n"
            f"    }}\n"
        )
        code = re.sub(rf"(class\s+{schema.java_class}\s*{{)", r"\1" + fallback, code, count=1)
        return NormalizeResult(files={schema.file_name: code}, notes="\n".join(notes), ok=True)

def indent(s: str, n: int = 4) -> str:
    pad = " " * n
    return "\n".join(pad + line if line.strip() else line for line in s.splitlines())
